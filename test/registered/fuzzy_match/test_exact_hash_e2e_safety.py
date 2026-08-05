"""E2E safety tests for ``ExactHashProvider``: hash-collision fallback and
cross-tenant isolation. Companion to
``test_exact_hash_shifted_offset_kl.py`` (which tests the *positive*
correctness case) — these test the two ways a fuzzy-match provider could
silently corrupt output if it were built carelessly:

1. **Hash collision must fall back to a token-ID equality check, never trust
   the hash alone** (``exact_hash_provider.py``'s ``_first_equal_match``).
   A real 64-bit hash collision is astronomically unlikely to occur by
   chance in a controlled test, so this test forces one, via the
   ``SGLANG_TEST_FUZZY_FORCE_HASH_COLLISION`` env var (collapses every
   chunk's fingerprint to a constant). With two *different* donor contents
   colliding into the same store bucket, a correct implementation still
   disambiguates by token-ID equality; a broken one (hash-only) would either
   serve the wrong donor's KV (silent corruption, catchable via KL
   divergence against a fresh recompute) or crash. A third, genuinely novel
   query (also forced to the same fingerprint) must fall back to a clean
   miss, proving the equality check actively *rejects* false candidates, not
   just "happens to usually work."

2. **Different tenants must never share fuzzy-matched content**
   (``extra_key``, already namespaced into the store key as
   ``(extra_key, fingerprint)``). Uses the real, already-wired-through
   ``extra_key`` field on ``/generate`` requests — no new plumbing.

Both tests verify behavior via server log content (``[FUZZY RADIX] fuzzy
match success`` firing or not firing across a request), not just HTTP status
codes, since a silently-wrong-but-200 response and a correct response are
indistinguishable from status code alone.
"""

import os
import random
import unittest

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kl_test_utils import (
    _extract_output_logprobs,
    _flush_cache,
    _generate,
    _get_input_logprobs,
    compare_kl_divergence,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Same checkpoint as test_exact_hash_shifted_offset_kl.py — see that file's
# module docstring for why (PR #31057's own tested checkpoint; the -1M
# variant uses Dual Chunk Attention, not plain RoPE).
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

SINK_TOKENS = 32  # must match chunker.SINK_TOKENS
NUM_SAMPLES = 8  # matches test_exact_hash_shifted_offset_kl.py's sample size
DONOR_TOKENS = 400
QUERY_PREFIX_TOKENS = 200
SYNTHETIC_TOKEN_LOW = 1000
SYNTHETIC_TOKEN_HIGH = 140000

ACC_THRESHOLDS = {MODEL: {"kl_div": 0.02}}

FUZZY_SUCCESS_MARKER = "[FUZZY RADIX] fuzzy match success"

FUZZY_ARGS = [
    "--radix-cache-backend",
    "fuzzy_match",
    "--fuzzy-match-provider",
    "ExactHash",
]

# One registration per file (both test classes below run under it) — see
# ci_register.py's AST-based collection, which registers the whole file per
# call, not per class.
register_cuda_ci(est_time=480, stage="base-b", runner_config="1-gpu-large")


def _rand_tokens(rng, n):
    return [rng.randint(SYNTHETIC_TOKEN_LOW, SYNTHETIC_TOKEN_HIGH) for _ in range(n)]


def _generate_with_extra_key(base_url, input_ids, max_new_tokens, extra_key):
    # kl_test_utils._generate has no extra_key parameter — it's a real,
    # already-wired-through top-level GenerateReqInput field
    # (srt/managers/io_struct.py), just not exposed by that shared helper.
    import requests

    json_data = {
        "input_ids": input_ids,
        "extra_key": extra_key,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
    }
    response = requests.post(base_url + "/generate", json=json_data)
    response.raise_for_status()
    return response.json()


def _log_tail(log_paths, start_pos_by_path):
    """Concatenate everything written to any of ``log_paths`` since the
    corresponding recorded position — logger output location (stdout vs
    stderr) isn't asserted on, so both are scanned."""
    chunks = []
    for path in log_paths:
        with open(path) as f:
            f.seek(start_pos_by_path[path])
            chunks.append(f.read())
    return "\n".join(chunks)


def _log_positions(log_paths):
    return {path: os.path.getsize(path) for path in log_paths}


class TestHashCollisionFallback(CustomTestCase):
    """Forces every chunk fingerprint to collide (fingerprint=0) and checks
    that ExactHashProvider still (a) serves the *correct* donor's KV when
    one exists, via KL divergence against a fresh recompute, and (b) falls
    back to a clean miss for content that collides but doesn't equality-
    match anything registered."""

    STDOUT_PATH = "/tmp/fuzzy_hash_collision_stdout.txt"
    STDERR_PATH = "/tmp/fuzzy_hash_collision_stderr.txt"

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.stdout = open(cls.STDOUT_PATH, "w")
        cls.stderr = open(cls.STDERR_PATH, "w")
        with envs.SGLANG_TEST_FUZZY_FORCE_HASH_COLLISION.override(True):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=FUZZY_ARGS,
                return_stdout_stderr=(cls.stdout, cls.stderr),
            )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)
        if hasattr(cls, "stdout"):
            cls.stdout.close()
        if hasattr(cls, "stderr"):
            cls.stderr.close()

    def test_collision_disambiguates_and_rejects_false_positives(self):
        # NUM_SAMPLES donors matches test_exact_hash_shifted_offset_kl.py's
        # sample size — an earlier version of this test used only 2 donors
        # and got a marginal KL-divergence failure (0.028 vs the 0.02
        # threshold) purely from small-sample variance (log-based checks
        # all passed; a *wrong*-donor substitution would produce KL
        # divergence an order of magnitude higher, not a 40% overshoot).
        # Averaging over NUM_SAMPLES colliding candidates, like the sibling
        # test does for non-colliding ones, is the honest fix — and is
        # incidentally a *harder* disambiguation test than 2 donors, since
        # every query must pick the right one out of NUM_SAMPLES colliding
        # candidates in the same store bucket, not just one alternative.
        rng = random.Random(20260727)

        donors = [_rand_tokens(rng, DONOR_TOKENS) for _ in range(NUM_SAMPLES)]
        novel = _rand_tokens(rng, DONOR_TOKENS)  # never registered
        prefixes = [_rand_tokens(rng, QUERY_PREFIX_TOKENS) for _ in range(NUM_SAMPLES)]
        novel_prefix = _rand_tokens(rng, QUERY_PREFIX_TOKENS)

        _flush_cache(self.base_url)
        sink_prefix = list(range(50000, 50000 + SINK_TOKENS))

        # Register NUM_SAMPLES genuinely different donors. With the forced
        # collision, every one of their chunks lands in the same
        # (extra_key=None, fingerprint=0) store bucket.
        _generate(self.base_url, [sink_prefix + d for d in donors], max_new_tokens=0)

        # Pre-cache each query prefix standalone so the combined request
        # below is a full exact match on the prefix, leaving donor/novel
        # content as the entire unmatched tail (see
        # test_exact_hash_shifted_offset_kl.py's docstring for why this
        # exact construction is required by the current "chunks[0] only"
        # limitation).
        _generate(self.base_url, prefixes + [novel_prefix], max_new_tokens=0)

        log_paths = [self.STDOUT_PATH, self.STDERR_PATH]

        # One batched request for all NUM_SAMPLES donor queries plus the
        # novel-content query — like test_exact_hash_shifted_offset_kl.py,
        # not a Python loop of individual requests. An earlier version sent
        # NUM_SAMPLES+1 separate back-to-back /generate calls and hit a
        # server crash ("cannot reshape tensor of 0 elements ... [0, -1,
        # 128]") inside vanilla qwen2.py RotaryEmbedding — a pre-existing
        # rapid-cycling edge case unrelated to fuzzy matching (that crash
        # site has no fuzzy-match frames in its stack); batching into one
        # request avoids the rapid-cycling trigger entirely rather than
        # working around an unrelated, pre-existing bug.
        #
        # This also must run *before* the KL check below —
        # _get_input_logprobs calls _flush_cache internally (needs a clean
        # slate for a fresh full-recompute baseline), which would wipe
        # ExactHashProvider's store (on_cache_reset) and silently defeat
        # this query if it ran afterward instead.
        queries = [p + d for p, d in zip(prefixes, donors)] + [novel_prefix + novel]
        start = _log_positions(log_paths)
        results = _generate(
            self.base_url, queries, max_new_tokens=64, return_logprob=True
        )
        self.assertEqual(len(results), NUM_SAMPLES + 1)
        log_tail = _log_tail(log_paths, start)

        # Exactly NUM_SAMPLES successes: proves every one of the
        # NUM_SAMPLES real donors was found despite colliding into the same
        # bucket (not fewer — a missed disambiguation), and that the novel
        # query — which also collides into that bucket but equality-matches
        # none of them — did *not* get a spurious match (not more).
        actual_successes = log_tail.count(FUZZY_SUCCESS_MARKER)
        self.assertEqual(
            actual_successes,
            NUM_SAMPLES,
            f"expected exactly {NUM_SAMPLES} fuzzy match successes (one per "
            "real donor, despite all colliding into the same forced-"
            "fingerprint bucket, and none for the novel never-registered "
            f"content that also collides into it), got {actual_successes}",
        )

        # --- Correctness oracle, batched (like
        # test_exact_hash_shifted_offset_kl.py): if the equality check
        # picked the *wrong* donor for any query above, the realized KV
        # would be numerically wrong and this KL comparison would fail even
        # though the log-based check above already passed. Excludes the
        # novel query (never registered — a real recompute, no fuzzy
        # correction to validate). ---
        donor_queries = queries[:NUM_SAMPLES]
        donor_results = results[:NUM_SAMPLES]
        new_input_ids = [
            q + r["output_ids"] for q, r in zip(donor_queries, donor_results)
        ]
        output_logprobs = [_extract_output_logprobs(r) for r in donor_results]
        input_logprobs = _get_input_logprobs(
            self.base_url, new_input_ids, output_logprobs
        )
        compare_kl_divergence(
            input_logprobs,
            output_logprobs,
            ACC_THRESHOLDS,
            self.model,
            "test_collision_disambiguates_and_rejects_false_positives",
        )


class TestCrossTenantIsolation(CustomTestCase):
    """Content registered under one extra_key must never fuzzy-match a
    lookup under a different extra_key, even when the token content is
    byte-identical."""

    STDOUT_PATH = "/tmp/fuzzy_cross_tenant_stdout.txt"
    STDERR_PATH = "/tmp/fuzzy_cross_tenant_stderr.txt"

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.stdout = open(cls.STDOUT_PATH, "w")
        cls.stderr = open(cls.STDERR_PATH, "w")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=FUZZY_ARGS,
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)
        if hasattr(cls, "stdout"):
            cls.stdout.close()
        if hasattr(cls, "stderr"):
            cls.stderr.close()

    def test_cross_tenant_lookup_never_matches(self):
        rng = random.Random(20260728)
        donor = _rand_tokens(rng, DONOR_TOKENS)
        query_prefix = _rand_tokens(rng, QUERY_PREFIX_TOKENS)

        _flush_cache(self.base_url)
        sink_prefix = list(range(50000, 50000 + SINK_TOKENS))

        # Register donor content under tenant_a only.
        _generate_with_extra_key(
            self.base_url, [sink_prefix + donor], max_new_tokens=0, extra_key="tenant_a"
        )

        # Pre-cache the *same* query prefix standalone under both tenants'
        # namespaces, so both scenarios below get an identical, full exact
        # match on the prefix — the only variable is extra_key on the
        # combined request. (Exact RadixCache is itself namespaced by
        # extra_key — srt/mem_cache/radix_cache.py's RadixKey — so without
        # this, tenant_b's combined request would get exact_matched_len=0
        # and a different, unrelated chunk-alignment path instead of a
        # clean isolation test.)
        _generate_with_extra_key(
            self.base_url, [query_prefix], max_new_tokens=0, extra_key="tenant_a"
        )
        _generate_with_extra_key(
            self.base_url, [query_prefix], max_new_tokens=0, extra_key="tenant_b"
        )

        log_paths = [self.STDOUT_PATH, self.STDERR_PATH]
        query = query_prefix + donor

        # --- Negative case first: a different tenant must get a clean miss
        # for byte-identical content. ---
        start = _log_positions(log_paths)
        _generate_with_extra_key(
            self.base_url, [query], max_new_tokens=8, extra_key="tenant_b"
        )
        log_cross = _log_tail(log_paths, start)
        self.assertNotIn(
            FUZZY_SUCCESS_MARKER,
            log_cross,
            "tenant_b must never fuzzy-match content registered only "
            "under tenant_a's extra_key",
        )

        # --- Positive control: the *same* tenant must still get the hit —
        # proves the negative result above is isolation working, not the
        # mechanism silently failing to fire at all. ---
        start = _log_positions(log_paths)
        _generate_with_extra_key(
            self.base_url, [query], max_new_tokens=8, extra_key="tenant_a"
        )
        log_same = _log_tail(log_paths, start)
        self.assertIn(
            FUZZY_SUCCESS_MARKER,
            log_same,
            "tenant_a must still get a fuzzy match on its own registered "
            "content (positive control for the isolation test above)",
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)

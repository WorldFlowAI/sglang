"""E2E correctness test: does ExactHashProvider's shifted-offset reuse
produce numerically consistent output vs. a fresh full recompute?

This is deliberately *not* a reuse of
``test_input_output_logprobs_match_prefill_cache_hit_helper`` — that helper
tests exact-prefix content reused at the *same* offset (today's ordinary
RadixCache behavior). This test constructs the scenario ``ExactHashProvider``
exists for: content registered as a donor at one absolute position, found
and reused when it reappears at a *different* position in a later,
unrelated prompt.

Scenario construction is deliberate, not arbitrary, because of a real
current limitation: ``ExactHashProvider`` only checks the *first* chunk of
the unmatched tail. For that chunk to align with a registered chunk, the
content immediately after the exact-matched prefix must be chunked
identically on both the registration and query side — which requires the
registration prefix to be exactly ``SINK_TOKENS`` long (so
``chunk_region_start`` lands exactly at the donor content's start) and the
query prefix to be *fully* exact-matched beforehand (so the unmatched tail
is exactly the donor content, chunked fresh from its own start). Real,
un-arranged traffic won't reliably line up this way yet — multi-chunk /
N:M segment matching is a natural follow-up (see the provider's own
docstring).

Donor/query content is **synthetic random token IDs**, not real dataset text
(e.g. LongBench, used by the sibling KL tests). An earlier version used real
LongBench-v2 samples and got spurious `cached_tokens` hits with no
`[FUZZY RADIX]`/`[EXACT_HASH]` log line ever firing — LongBench-v2 entries
frequently share source documents across different questions, so two
samples drawn from "different" indices in the pool aren't guaranteed to be
non-overlapping content, which is exactly what this test needs to control
for precisely. Random token IDs make non-overlap essentially certain and
are fine for a pure numerical self-consistency check — the KL comparison is
between two computations of the same thing, not a judgment about text
quality.
"""

import random
import unittest

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

# Qwen2.5-7B-Instruct-AWQ: the exact checkpoint PR #31057 was tested
# against. NOTE: Qwen2.5-7B-Instruct-1M (already cached locally) was tried
# first as a same-family substitute to avoid a download, on the assumption
# that it's architecturally identical modulo rope_theta — that assumption
# was wrong. The -1M variant uses Dual Chunk Attention for its extended
# context (a genuinely different attention mechanism, requiring a specific
# dual_chunk_flash_attn backend), not plain RoPE-theta scaling. Confirmed
# by actually trying to launch it, not by re-reading the config more
# carefully beforehand — worth being honest about in case this recurs.
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

SINK_TOKENS = 32  # must match chunker.SINK_TOKENS
NUM_SAMPLES = 8
DONOR_TOKENS = 400
QUERY_PREFIX_TOKENS = 200
# Safe ordinary-token range for Qwen2.5's ~151.6k vocab — avoids the
# special-token IDs clustered near the top of the vocab and near 0.
SYNTHETIC_TOKEN_LOW = 1000
SYNTHETIC_TOKEN_HIGH = 140000

ACC_THRESHOLDS = {MODEL: {"kl_div": 0.02}}

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-large")


class TestExactHashShiftedOffset(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--radix-cache-backend",
                "fuzzy_match",
                "--fuzzy-match-provider",
                "ExactHash",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_shifted_offset_reuse_matches_full_recompute(self):
        # Synthetic, guaranteed-non-overlapping content — see module
        # docstring for why real dataset text doesn't work for this.
        rng = random.Random(20260726)

        def rand_tokens(n):
            return [
                rng.randint(SYNTHETIC_TOKEN_LOW, SYNTHETIC_TOKEN_HIGH) for _ in range(n)
            ]

        donor_contents = [rand_tokens(DONOR_TOKENS) for _ in range(NUM_SAMPLES)]
        query_prefixes = [rand_tokens(QUERY_PREFIX_TOKENS) for _ in range(NUM_SAMPLES)]

        _flush_cache(self.base_url)

        # Fixed SINK_TOKENS-length prefix: makes chunk_region_start land
        # exactly at donor content's start during registration (see module
        # docstring). Distinct constant range from the donor/query pools,
        # still within the safe ordinary-token window (not near 0, where
        # low IDs risk colliding with special/reserved tokens).
        sink_prefix = list(range(50000, 50000 + SINK_TOKENS))
        registration_prompts = [sink_prefix + d for d in donor_contents]
        _generate(self.base_url, registration_prompts, max_new_tokens=0)

        # Pre-cache each query prefix standalone, so the combined request
        # below gets a full *exact* match on it — leaving donor content as
        # the entire unmatched tail, chunked fresh from its own start.
        _generate(self.base_url, query_prefixes, max_new_tokens=0)

        query_prompts = [
            query_prefixes[i] + donor_contents[i] for i in range(NUM_SAMPLES)
        ]
        results = _generate(
            self.base_url,
            query_prompts,
            max_new_tokens=64,
            return_logprob=True,
        )
        self.assertEqual(len(results), NUM_SAMPLES)

        new_input_ids = []
        output_logprobs = []
        for i, result in enumerate(results):
            new_input_ids.append(query_prompts[i] + result["output_ids"])
            output_logprobs.append(_extract_output_logprobs(result))

        input_logprobs = _get_input_logprobs(
            self.base_url, new_input_ids, output_logprobs
        )

        compare_kl_divergence(
            input_logprobs,
            output_logprobs,
            ACC_THRESHOLDS,
            self.model,
            "test_shifted_offset_reuse_matches_full_recompute",
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)

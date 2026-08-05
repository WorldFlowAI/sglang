"""Unit tests for srt/mem_cache/fuzzy_match/chunker.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import random
import unittest

from sglang.srt.environ import envs
from sglang.srt.mem_cache.fuzzy_match.chunker import (
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_TOKENS,
    chunk_tokens,
)
from sglang.test.test_utils import CustomTestCase


class TestChunkTokens(CustomTestCase):
    def test_content_defined_not_position_defined(self):
        """Identical content at a different absolute offset must still
        produce at least one matching chunk (same fingerprint, same token
        IDs). This is the entire property CDC exists for — the property
        a fixed-boundary / position-defined chunker provably lacks.
        Regressing to fixed-boundary chunking would silently defeat the
        whole mechanism without failing any obviously-related test, since
        fixed boundaries still "work" in isolation.
        """
        rng = random.Random(42)
        shared_content = [rng.randint(1000, 50000) for _ in range(1500)]
        prefix_a = [rng.randint(1000, 50000) for _ in range(50)]
        prefix_b = [rng.randint(1000, 50000) for _ in range(137)]

        chunks_a = chunk_tokens(prefix_a + shared_content)
        chunks_b = chunk_tokens(prefix_b + shared_content)

        by_fp_a = {c.fingerprint: c.token_ids for c in chunks_a}
        by_fp_b = {c.fingerprint: c.token_ids for c in chunks_b}
        shared_fingerprints = set(by_fp_a) & set(by_fp_b)

        self.assertGreater(
            len(shared_fingerprints),
            0,
            "expected at least one chunk boundary to align across two "
            "sequences sharing 1500 tokens of content behind different-"
            "length prefixes",
        )
        for fp in shared_fingerprints:
            self.assertEqual(
                by_fp_a[fp],
                by_fp_b[fp],
                "same fingerprint but different token IDs — the fingerprint "
                "itself must never be trusted without this equality holding "
                "for the CDC layer to be sound",
            )

    def test_chunk_sizes_bounded(self):
        """Every emitted chunk must respect the [MIN, MAX] clamp — an
        off-by-one in the boundary/clamp bookkeeping could otherwise emit
        pathologically tiny (defeats amortizing the hash lookup) or huge
        (defeats fine-grained reuse) chunks silently.
        """
        rng = random.Random(7)
        tokens = [rng.randint(0, 100000) for _ in range(5000)]
        chunks = chunk_tokens(tokens)
        self.assertGreater(len(chunks), 0)
        for c in chunks:
            size = c.end - c.start
            self.assertGreaterEqual(size, MIN_CHUNK_TOKENS)
            self.assertLessEqual(size, MAX_CHUNK_TOKENS)
            self.assertEqual(size, len(c.token_ids))

    def test_chunks_tile_without_gaps_or_overlap(self):
        """Chunks must exactly tile the input in order (chunk[i].end ==
        chunk[i+1].start), with only a final under-MIN_CHUNK_TOKENS
        remainder legitimately dropped. A bug here (double-counting or
        skipping tokens at a boundary) would silently register or match
        content that doesn't correspond to what's actually at that offset.
        """
        rng = random.Random(99)
        tokens = [rng.randint(0, 100000) for _ in range(3000)]
        chunks = chunk_tokens(tokens)

        self.assertEqual(chunks[0].start, 0)
        for prev, nxt in zip(chunks, chunks[1:]):
            self.assertEqual(prev.end, nxt.start)
        remainder = len(tokens) - chunks[-1].end
        self.assertLess(remainder, MIN_CHUNK_TOKENS)

    def test_force_hash_collision_env_var_collapses_fingerprints(self):
        """The E2E hash-collision-fallback test
        (test_exact_hash_e2e_safety.py) relies entirely on
        SGLANG_TEST_FUZZY_FORCE_HASH_COLLISION making two genuinely
        different chunks collide by construction. If that wiring silently
        broke (renamed env var, hook removed), the E2E test would degrade
        into exercising nothing while still passing — this pins the hook's
        actual effect at the unit level, independent of any server.
        """
        rng = random.Random(1234)
        tokens_a = [rng.randint(0, 100000) for _ in range(200)]
        tokens_b = [rng.randint(0, 100000) for _ in range(200)]
        self.assertNotEqual(tokens_a, tokens_b)

        with envs.SGLANG_TEST_FUZZY_FORCE_HASH_COLLISION.override(True):
            fp_a = chunk_tokens(tokens_a)[0].fingerprint
            fp_b = chunk_tokens(tokens_b)[0].fingerprint
        self.assertEqual(fp_a, fp_b, "override(True) must force a collision")

        fp_a_real = chunk_tokens(tokens_a)[0].fingerprint
        fp_b_real = chunk_tokens(tokens_b)[0].fingerprint
        self.assertNotEqual(
            fp_a_real,
            fp_b_real,
            "override must not leak past its `with` block",
        )


if __name__ == "__main__":
    unittest.main()

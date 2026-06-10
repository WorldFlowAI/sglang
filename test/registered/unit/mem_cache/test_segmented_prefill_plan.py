"""Unit tests for segmented semantic prefill planning."""

from __future__ import annotations

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.srt.mem_cache.fuzzy_match.config import FuzzyMatchConfig
from sglang.srt.mem_cache.fuzzy_match.fuzzy_match_provider import (
    FuzzyMatchResult,
    FuzzyMatchSegment,
)
from sglang.srt.mem_cache.fuzzy_match.segmented_prefill import (
    build_segmented_prefill_plan,
)

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


def _segment(target_positions, donor_positions=None, donor_kv_indices=None):
    donor_positions = (
        donor_positions
        if donor_positions is not None
        else list(range(100, 100 + len(target_positions)))
    )
    donor_kv_indices = (
        donor_kv_indices
        if donor_kv_indices is not None
        else list(range(200, 200 + len(target_positions)))
    )
    return FuzzyMatchSegment(
        target_positions=torch.tensor(target_positions, dtype=torch.int64),
        donor_positions=torch.tensor(donor_positions, dtype=torch.int64),
        donor_kv_indices=torch.tensor(donor_kv_indices, dtype=torch.int64),
    )


class TestSegmentedPrefillPlan(unittest.TestCase):
    def test_multi_segment_plan_preserves_fresh_holes(self):
        plan = build_segmented_prefill_plan(
            prompt_token_count=12,
            exact_prefix_len=2,
            segments=[
                _segment([5, 6, 7]),
                _segment([10, 11], donor_positions=[150, 151]),
            ],
        )

        self.assertEqual(
            [(span.kind, span.start_pos, span.end_pos) for span in plan.spans],
            [
                ("exact_prefix", 0, 2),
                ("fresh", 2, 5),
                ("donor", 5, 8),
                ("fresh", 8, 10),
                ("donor", 10, 12),
            ],
        )
        self.assertEqual(plan.donor_token_count, 5)
        self.assertEqual(plan.fresh_token_count, 5)
        self.assertEqual(plan.donor_target_positions, (5, 6, 7, 10, 11))
        self.assertEqual(plan.fresh_target_positions, (2, 3, 4, 8, 9))
        self.assertFalse(plan.prefix_contract_compatible)
        self.assertEqual(
            [
                (step.kind, step.start_pos, step.end_pos, step.prefix_len_before)
                for step in plan.phased_dense_steps
            ],
            [
                ("compute_fresh", 2, 5, 2),
                ("realize_donor", 5, 8, 5),
                ("compute_fresh", 8, 10, 8),
                ("realize_donor", 10, 12, 10),
            ],
        )
        self.assertEqual(plan.next_required_fresh_chunk_end(2), 5)
        self.assertEqual(plan.leading_donor_end(5), 8)
        self.assertEqual(plan.next_required_fresh_chunk_end(8), 10)
        self.assertEqual(plan.leading_donor_end(10), 12)
        self.assertIsNone(plan.next_required_fresh_chunk_end(12))
        self.assertEqual(
            (
                plan.next_backend_step(2).kind,
                plan.next_backend_step(2).start_pos,
                plan.next_backend_step(2).end_pos,
            ),
            ("compute_fresh", 2, 5),
        )
        self.assertEqual(
            (
                plan.next_backend_step(5).kind,
                plan.next_backend_step(5).start_pos,
                plan.next_backend_step(5).end_pos,
            ),
            ("realize_donor", 5, 8),
        )

    def test_contiguous_donor_spans_are_one_backend_step(self):
        plan = build_segmented_prefill_plan(
            prompt_token_count=10,
            exact_prefix_len=2,
            segments=[
                _segment([2, 3], donor_positions=[20, 21], donor_kv_indices=[30, 31]),
                _segment([4, 5], donor_positions=[22, 23], donor_kv_indices=[32, 33]),
                _segment([8, 9], donor_positions=[40, 41], donor_kv_indices=[50, 51]),
            ],
        )

        step = plan.next_backend_step(2)
        self.assertEqual(
            (step.kind, step.start_pos, step.end_pos), ("realize_donor", 2, 6)
        )
        self.assertEqual(step.donor_positions, (20, 21, 22, 23))
        self.assertEqual(step.donor_kv_indices, (30, 31, 32, 33))
        self.assertEqual(
            (plan.next_backend_step(6).kind, plan.next_backend_step(6).end_pos),
            ("compute_fresh", 8),
        )

    def test_current_prefix_compression_skips_prompt_holes(self):
        plan = build_segmented_prefill_plan(
            prompt_token_count=12,
            exact_prefix_len=2,
            segments=[
                _segment([5, 6, 7]),
                _segment([10, 11]),
            ],
        )

        # Current fuzzy reuse appends donor KV to prefix_indices and then uses
        # fill_ids[len(prefix_indices):]. With exact=2 and five donor tokens,
        # the scheduler starts fresh prefill at position 7, silently dropping
        # fresh prompt positions 2, 3, and 4.
        self.assertEqual(plan.skipped_by_prefix_compression, (2, 3, 4))

    def test_sparse_prefill_metadata_uses_only_fresh_queries(self):
        plan = build_segmented_prefill_plan(
            prompt_token_count=12,
            exact_prefix_len=2,
            segments=[
                _segment(
                    [5, 6, 7],
                    donor_positions=[20, 21, 22],
                    donor_kv_indices=[30, 31, 32],
                ),
                _segment(
                    [10, 11],
                    donor_positions=[40, 41],
                    donor_kv_indices=[50, 51],
                ),
            ],
        )

        sparse = plan.sparse_prefill_metadata()

        self.assertEqual(sparse.fresh_positions, (2, 3, 4, 8, 9))
        self.assertEqual(sparse.donor_positions, (5, 6, 7, 10, 11))
        self.assertEqual(sparse.donor_kv_indices, (30, 31, 32, 50, 51))
        self.assertEqual(
            sparse.input_token_ids(list(range(100, 112))), (102, 103, 104, 108, 109)
        )

        mask = sparse.token_level_causal_mask()
        self.assertEqual(len(mask), 5)
        self.assertEqual(len(mask[0]), 12)
        self.assertEqual(
            mask[0],
            (
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ),
        )
        self.assertEqual(
            mask[-1],
            (True, True, True, True, True, True, True, True, True, True, False, False),
        )

    def test_sparse_prefill_metadata_supports_three_donors_and_fresh_tail(self):
        plan = build_segmented_prefill_plan(
            prompt_token_count=20,
            exact_prefix_len=0,
            segments=[
                _segment(
                    [3, 4, 5],
                    donor_positions=[30, 31, 32],
                    donor_kv_indices=[300, 301, 302],
                ),
                _segment(
                    [8, 9, 10],
                    donor_positions=[80, 81, 82],
                    donor_kv_indices=[800, 801, 802],
                ),
                _segment(
                    [13, 14, 15],
                    donor_positions=[130, 131, 132],
                    donor_kv_indices=[1300, 1301, 1302],
                ),
            ],
        )

        self.assertEqual(
            [(span.kind, span.start_pos, span.end_pos) for span in plan.spans],
            [
                ("fresh", 0, 3),
                ("donor", 3, 6),
                ("fresh", 6, 8),
                ("donor", 8, 11),
                ("fresh", 11, 13),
                ("donor", 13, 16),
                ("fresh", 16, 20),
            ],
        )
        self.assertEqual(plan.donor_target_positions, (3, 4, 5, 8, 9, 10, 13, 14, 15))
        self.assertEqual(
            plan.fresh_target_positions, (0, 1, 2, 6, 7, 11, 12, 16, 17, 18, 19)
        )
        self.assertFalse(plan.prefix_contract_compatible)
        self.assertEqual(plan.skipped_by_prefix_compression, (0, 1, 2, 6, 7))

        sparse = plan.sparse_prefill_metadata()
        self.assertEqual(sparse.fresh_token_count, 11)
        self.assertEqual(sparse.donor_token_count, 9)
        self.assertEqual(
            sparse.donor_kv_indices,
            (300, 301, 302, 800, 801, 802, 1300, 1301, 1302),
        )
        self.assertEqual(
            sparse.input_token_ids(list(range(100, 120))),
            (100, 101, 102, 106, 107, 111, 112, 116, 117, 118, 119),
        )

        mask = sparse.token_level_causal_mask()
        self.assertEqual(len(mask), 11)
        self.assertEqual(len(mask[0]), 20)
        self.assertEqual(mask[0], (True,) + (False,) * 19)
        self.assertEqual(mask[3], (True,) * 7 + (False,) * 13)
        self.assertEqual(mask[-1], (True,) * 20)

    def test_prefix_shaped_segment_is_geometrically_compatible(self):
        plan = build_segmented_prefill_plan(
            prompt_token_count=9,
            exact_prefix_len=2,
            segments=[_segment([2, 3, 4])],
        )

        self.assertEqual(
            [(span.kind, span.start_pos, span.end_pos) for span in plan.spans],
            [
                ("exact_prefix", 0, 2),
                ("donor", 2, 5),
                ("fresh", 5, 9),
            ],
        )
        self.assertTrue(plan.prefix_contract_compatible)
        self.assertEqual(plan.skipped_by_prefix_compression, ())
        self.assertEqual(plan.leading_donor_end(2), 5)
        self.assertIsNone(plan.next_required_fresh_chunk_end(5))
        self.assertEqual(
            [
                (step.kind, step.start_pos, step.end_pos, step.prefix_len_before)
                for step in plan.phased_dense_steps
            ],
            [
                ("realize_donor", 2, 5, 2),
                ("compute_fresh", 5, 9, 5),
            ],
        )

    def test_rejects_overlapping_segments(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            build_segmented_prefill_plan(
                prompt_token_count=12,
                exact_prefix_len=2,
                segments=[_segment([5, 6, 7]), _segment([7, 8])],
            )

    def test_rejects_segments_that_overlap_exact_prefix(self):
        with self.assertRaisesRegex(ValueError, "exact prefix"):
            build_segmented_prefill_plan(
                prompt_token_count=8,
                exact_prefix_len=3,
                segments=[_segment([2, 3, 4])],
            )

    def test_rejects_non_contiguous_segment(self):
        with self.assertRaisesRegex(ValueError, "contiguous"):
            build_segmented_prefill_plan(
                prompt_token_count=12,
                exact_prefix_len=2,
                segments=[_segment([5, 7])],
            )


class TestSegmentedPrefillRadixContract(unittest.TestCase):
    def test_non_prefix_segment_is_not_appended_to_prefix_indices(self):
        try:
            from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
            from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
        except TypeError as exc:
            self.skipTest(f"local torch custom-op registration unavailable: {exc}")

        cache = RadixCache.create_simulated()

        class FakeProvider:
            def cache_on_request_finished(self, *args, **kwargs):
                return False

            def on_cache_reset(self):
                return None

            def match_on_prefix_miss(
                self,
                prompt_token_ids,
                already_matched_len,
                request=None,
                extra_key=None,
            ):
                return FuzzyMatchResult(
                    cached_token_count=3,
                    cached_token_ids=[50, 51, 52],
                    prompt_token_count=len(prompt_token_ids),
                    kv_cache_indices=torch.tensor([20, 21, 22], dtype=torch.int64),
                    position_offset=5,
                    cached_start_pos=5,
                    segments=[_segment([5, 6, 7])],
                )

        config = FuzzyMatchConfig(
            enable_fuzzy_match=True,
            cache_fuzzy_results=False,
            fuzzy_min_match_length=1,
        )
        cache.init_fuzzy_match(config, FakeProvider())

        class Req:
            rid = "segmented-req"
            segmented_prefill_plan = None
            requires_segmented_prefill_backend = False

        req = Req()
        result = cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey([1, 2, 3, 4, 5, 6, 7, 8]),
                req=req,
            )
        )

        self.assertEqual(len(result.device_indices), 0)
        self.assertIsNone(result.fuzzy_matched_len)
        self.assertIsNotNone(req.segmented_prefill_plan)
        self.assertTrue(req.requires_segmented_prefill_backend)
        self.assertEqual(
            req.segmented_prefill_plan.skipped_by_prefix_compression, (0, 1, 2)
        )

    def test_short_exact_prefix_still_calls_semantic_provider(self):
        try:
            from sglang.srt.mem_cache.base_prefix_cache import (
                InsertParams,
                MatchPrefixParams,
            )
            from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
        except TypeError as exc:
            self.skipTest(f"local torch custom-op registration unavailable: {exc}")

        cache = RadixCache.create_simulated()
        cache.insert(
            InsertParams(
                key=RadixKey([1, 2, 3]),
                value=torch.tensor([10, 11, 12], dtype=torch.int64),
            )
        )

        class FakeProvider:
            called = False
            already_matched_len = None

            def cache_on_request_finished(self, *args, **kwargs):
                return False

            def on_cache_reset(self):
                return None

            def match_on_prefix_miss(
                self,
                prompt_token_ids,
                already_matched_len,
                request=None,
                extra_key=None,
            ):
                self.called = True
                self.already_matched_len = already_matched_len
                return None

        provider = FakeProvider()
        config = FuzzyMatchConfig(
            enable_fuzzy_match=True,
            cache_fuzzy_results=False,
            fuzzy_min_match_length=8,
        )
        cache.init_fuzzy_match(config, provider)

        cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey([1, 2, 3, 4, 5, 6]),
            )
        )

        self.assertTrue(provider.called)
        self.assertEqual(provider.already_matched_len, 3)


if __name__ == "__main__":
    unittest.main()

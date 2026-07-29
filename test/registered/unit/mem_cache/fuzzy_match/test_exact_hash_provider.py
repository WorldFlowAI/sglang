"""Unit tests for srt/mem_cache/fuzzy_match/exact_hash_provider.py.

No server, no model loading — pure provider logic against a fake Req.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.fuzzy_match.chunker import SINK_TOKENS, Chunk
from sglang.srt.mem_cache.fuzzy_match.config import FuzzyMatchConfig
from sglang.srt.mem_cache.fuzzy_match.exact_hash_provider import ExactHashProvider
from sglang.test.test_utils import CustomTestCase


def _fake_req(rid: str, extra_key=None):
    return types.SimpleNamespace(rid=rid, extra_key=extra_key)


def _provider() -> ExactHashProvider:
    return ExactHashProvider(
        FuzzyMatchConfig(enable_fuzzy_match=True, fuzzy_match_provider="ExactHash")
    )


class TestExactHashProviderMatching(CustomTestCase):
    def test_exact_match_at_shifted_offset(self):
        """The entire point of the mechanism: content registered at one
        absolute position must be found and returned (with the correct
        p_src/position_offset) when the same content later appears at a
        different offset.
        """
        provider = _provider()
        content = list(range(2000, 2000 + 200))  # well past SINK_TOKENS
        donor_tokens = [0] * SINK_TOKENS + content
        kv = torch.arange(len(donor_tokens))

        ok = provider.cache_on_request_finished(
            request=_fake_req("donor-1"),
            token_ids=donor_tokens,
            kv_cache=kv,
            cache_start_pos=0,
            cache_end_pos=len(donor_tokens),
        )
        self.assertTrue(ok)
        provider.on_donor_inserted(_fake_req("donor-1"), donor_last_node_id=42)

        # Same content, arriving at a different offset in a new prompt.
        already_matched_len = 500
        prompt = list(range(9999, 9999 + already_matched_len)) + content
        result = provider.match_on_prefix_miss(
            prompt_token_ids=prompt,
            already_matched_len=already_matched_len,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.cached_start_pos, SINK_TOKENS)
        self.assertEqual(result.position_offset, already_matched_len - SINK_TOKENS)
        self.assertEqual(result.donor_last_node_id, 42)
        self.assertEqual(result.cached_token_ids, content[: result.cached_token_count])

    def test_sink_region_never_registered(self):
        """Content whose original occurrence started inside the
        attention-sink zone must never be registered as a donor: early
        positions absorb a disproportionate, content-independent share of
        attention regardless of what's actually there, so a content hash
        match in that zone isn't a trustworthy signal. A regression here
        would silently start serving sink-adjacent content as if it were a
        reliable, content-only signal.
        """
        provider = _provider()
        # Entirely within the sink zone.
        donor_tokens = list(range(SINK_TOKENS))
        kv = torch.arange(len(donor_tokens))

        ok = provider.cache_on_request_finished(
            request=_fake_req("donor-sink"),
            token_ids=donor_tokens,
            kv_cache=kv,
            cache_start_pos=0,
            cache_end_pos=len(donor_tokens),
        )
        self.assertFalse(ok)
        self.assertEqual(len(provider._store), 0)

    def test_cross_tenant_isolation(self):
        """Identical content registered under one tenant's extra_key must
        never match a lookup under a different tenant's extra_key — this
        provider must enforce multi-tenant isolation unconditionally. A
        regression here is a real cross-tenant KV leak.
        """
        provider = _provider()
        content = list(range(3000, 3000 + 200))
        donor_tokens = [0] * SINK_TOKENS + content
        kv = torch.arange(len(donor_tokens))

        provider.cache_on_request_finished(
            request=_fake_req("donor-2", extra_key="tenant_a"),
            token_ids=donor_tokens,
            kv_cache=kv,
            cache_start_pos=0,
            cache_end_pos=len(donor_tokens),
        )

        already_matched_len = 10
        prompt = list(range(10)) + content
        result = provider.match_on_prefix_miss(
            prompt_token_ids=prompt,
            already_matched_len=already_matched_len,
            extra_key="tenant_b",
        )
        self.assertIsNone(result)

        # Sanity: the same lookup under the correct tenant does match.
        result_same_tenant = provider.match_on_prefix_miss(
            prompt_token_ids=prompt,
            already_matched_len=already_matched_len,
            extra_key="tenant_a",
        )
        self.assertIsNotNone(result_same_tenant)

    def test_hash_collision_falls_back_to_miss(self):
        """Never trust the hash alone: two different chunks forced to share
        a fingerprint must not produce a match — the mandatory token-ID
        equality check is what actually guards correctness here, not the
        fingerprint's collision odds.
        """
        provider = _provider()
        donor_content = [111] * 200
        query_content = [222] * 200  # different content, forced same fingerprint

        def fake_chunks(tokens):
            return [
                Chunk(start=0, end=len(tokens), token_ids=list(tokens), fingerprint=1)
            ]

        with patch(
            "sglang.srt.mem_cache.fuzzy_match.exact_hash_provider.chunk_tokens",
            side_effect=fake_chunks,
        ):
            donor_tokens = [0] * SINK_TOKENS + donor_content
            provider.cache_on_request_finished(
                request=_fake_req("donor-3"),
                token_ids=donor_tokens,
                kv_cache=torch.arange(len(donor_tokens)),
                cache_start_pos=0,
                cache_end_pos=len(donor_tokens),
            )

            result = provider.match_on_prefix_miss(
                prompt_token_ids=[0] * 10 + query_content,
                already_matched_len=10,
            )
        self.assertIsNone(
            result,
            "fingerprint collided by construction but content differs — "
            "must fall back to miss, not serve the wrong donor",
        )


if __name__ == "__main__":
    unittest.main()

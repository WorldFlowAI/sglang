"""Unit tests for fuzzy match provider integration edges."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sys
import types
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.srt.mem_cache.fuzzy_match.config import FuzzyMatchConfig
from sglang.srt.mem_cache.fuzzy_match.fuzzy_match_provider import FuzzyMatchResult

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class TestSemanticEmbeddingProvider(unittest.TestCase):
    def test_match_decodes_query_text_and_passes_extra_key(self):
        class FakeSemBlendConfig:
            @classmethod
            def from_dict(cls, values):
                inst = cls()
                inst.values = dict(values)
                return inst

        class FakeAdapter:
            instances = []

            def __init__(self, config):
                self.config = config
                self.match_calls = []
                FakeAdapter.instances.append(self)

            def match(
                self,
                prompt_token_ids,
                already_matched_len,
                *,
                prompt_text=None,
                extra_key=None,
            ):
                self.match_calls.append(
                    {
                        "prompt_token_ids": list(prompt_token_ids),
                        "already_matched_len": already_matched_len,
                        "prompt_text": prompt_text,
                        "extra_key": extra_key,
                    }
                )
                return None

        fake_semblend = types.ModuleType("semblend")
        fake_semblend.__version__ = "0.3.12"
        fake_config_mod = types.ModuleType("semblend.integration.sglang.config")
        fake_config_mod.SemBlendProviderConfig = FakeSemBlendConfig
        fake_provider_mod = types.ModuleType("semblend.integration.sglang.provider")
        fake_provider_mod.SemBlendProviderAdapter = FakeAdapter

        saved = {
            name: sys.modules.get(name)
            for name in (
                "semblend",
                "semblend.integration",
                "semblend.integration.sglang",
                "semblend.integration.sglang.config",
                "semblend.integration.sglang.provider",
            )
        }
        try:
            sys.modules["semblend"] = fake_semblend
            sys.modules["semblend.integration"] = types.ModuleType(
                "semblend.integration"
            )
            sys.modules["semblend.integration.sglang"] = types.ModuleType(
                "semblend.integration.sglang"
            )
            sys.modules["semblend.integration.sglang.config"] = fake_config_mod
            sys.modules["semblend.integration.sglang.provider"] = fake_provider_mod

            from sglang.srt.mem_cache.fuzzy_match.semantic_embedding import (
                SemanticEmbeddingProvider,
            )

            config = FuzzyMatchConfig(
                enable_fuzzy_match=True,
                fuzzy_match_provider="SemanticEmbedding",
                fuzzy_min_match_length=1,
            )
            provider = SemanticEmbeddingProvider(config)

            class Tokenizer:
                def decode(self, token_ids, skip_special_tokens=False):
                    return "decoded:" + ",".join(map(str, token_ids))

            class Request:
                tokenizer = Tokenizer()

            provider.match_on_prefix_miss(
                prompt_token_ids=[1, 2, 3, 4],
                already_matched_len=1,
                request=Request(),
                extra_key="tenant-a",
            )

            call = FakeAdapter.instances[-1].match_calls[-1]
            self.assertEqual(call["prompt_token_ids"], [1, 2, 3, 4])
            self.assertEqual(call["already_matched_len"], 1)
            self.assertEqual(call["prompt_text"], "decoded:2,3,4")
            self.assertEqual(call["extra_key"], "tenant-a")
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module


class TestRadixFuzzyConcurrency(unittest.TestCase):
    def test_concurrent_matches_lock_same_donor_node(self):
        try:
            from sglang.srt.mem_cache.base_prefix_cache import (
                InsertParams,
                MatchPrefixParams,
            )
            from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
        except TypeError as e:
            self.skipTest(f"local torch custom-op registration unavailable: {e}")

        cache = RadixCache.create_simulated()
        donor_insert = cache.insert(
            InsertParams(
                key=RadixKey([1, 2, 3, 4]),
                value=torch.tensor([10, 11, 12, 13], dtype=torch.int64),
            )
        )
        donor = cache._node_registry[donor_insert.last_node_id]

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
                    cached_token_count=2,
                    cached_token_ids=[1, 2],
                    prompt_token_count=2,
                    kv_cache_indices=torch.tensor([10, 11], dtype=torch.int64),
                    position_offset=already_matched_len,
                    cached_start_pos=already_matched_len,
                    donor_last_node_id=donor_insert.last_node_id,
                )

        config = FuzzyMatchConfig(
            enable_fuzzy_match=True,
            cache_fuzzy_results=False,
            fuzzy_min_match_length=1,
        )
        cache.init_fuzzy_match(config, FakeProvider())

        class Req:
            def __init__(self, rid):
                self.rid = rid

        reqs = [Req(f"req-{i}") for i in range(8)]

        def run_match(req):
            result = cache.match_prefix(
                MatchPrefixParams(
                    key=RadixKey([90, 91, 92]),
                    req=req,
                )
            )
            self.assertEqual(result.fuzzy_matched_len, 2)
            self.assertIs(req.fuzzy_donor_node, donor)

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(run_match, reqs))

        self.assertEqual(donor.lock_ref, len(reqs))

        for req in reqs:
            cache.dec_lock_ref(req.fuzzy_donor_node)
            req.fuzzy_donor_node = None

        self.assertEqual(donor.lock_ref, 0)


if __name__ == "__main__":
    unittest.main()

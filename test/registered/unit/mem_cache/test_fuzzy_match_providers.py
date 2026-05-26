"""Unit tests for fuzzy match provider integration edges."""

from __future__ import annotations

import sys
import types
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.srt.mem_cache.fuzzy_match.config import FuzzyMatchConfig
from sglang.srt.mem_cache.fuzzy_match.non_prefix_store import (
    NodeRef,
    NonPrefixKVStore,
)
from sglang.srt.mem_cache.fuzzy_match.token_block_match import TokenBlockMatchProvider

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class TestNonPrefixKVStore(unittest.TestCase):
    def test_eviction_keeps_block_index_entry_ids_valid(self):
        store = NonPrefixKVStore(max_entries=2, block_size=2)
        store.insert([0, 1, 2, 3], [NodeRef(1, 0, 4)], extra_key=None)
        store.insert([10, 11, 12, 13], [NodeRef(2, 0, 4)], extra_key=None)
        store.insert([20, 21, 22, 23], [NodeRef(3, 0, 4)], extra_key=None)

        self.assertEqual([entry.id for entry in store.entries], [1, 2])

        matches = store.find_by_block_hash(
            query_tokens=[20, 21, 22, 23],
            min_length=2,
        )

        self.assertTrue(matches)
        self.assertEqual(matches[0][1].id, 2)

    def test_clear_removes_entries_and_indexes(self):
        store = NonPrefixKVStore(max_entries=2, block_size=2)
        store.insert([0, 1, 2, 3], [NodeRef(1, 0, 4)], extra_key="tenant")

        store.clear()

        self.assertEqual(store.entries, [])
        self.assertEqual(dict(store.block_index), {})
        self.assertEqual(store.total_entries, 0)


class TestTokenBlockMatchProvider(unittest.TestCase):
    def _config(self):
        return FuzzyMatchConfig(
            enable_fuzzy_match=True,
            fuzzy_match_provider="TokenBlockMatch",
            fuzzy_min_match_length=2,
            fuzzy_block_size=2,
            fuzzy_non_prefix_max_entries=8,
        )

    def test_on_cache_reset_clears_non_prefix_store(self):
        provider = TokenBlockMatchProvider(self._config())
        provider.non_prefix_store.insert(
            [0, 1, 2, 3],
            [NodeRef(1, 0, 4)],
            extra_key="tenant",
        )

        provider.on_cache_reset()

        self.assertEqual(provider.non_prefix_store.total_entries, 0)
        self.assertEqual(dict(provider.non_prefix_store.block_index), {})

    def test_match_passes_extra_key_to_store(self):
        provider = TokenBlockMatchProvider(self._config())
        seen = {}

        def fake_find_by_block_hash(query_tokens, min_length, extra_key=None):
            seen["query_tokens"] = list(query_tokens)
            seen["min_length"] = min_length
            seen["extra_key"] = extra_key
            return []

        provider.non_prefix_store.find_by_block_hash = fake_find_by_block_hash

        result = provider.match_on_prefix_miss(
            prompt_token_ids=[99, 0, 1, 2, 3],
            already_matched_len=1,
            extra_key="tenant-a",
        )

        self.assertIsNone(result)
        self.assertEqual(seen["query_tokens"], [0, 1, 2, 3])
        self.assertEqual(seen["min_length"], 2)
        self.assertEqual(seen["extra_key"], "tenant-a")


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


if __name__ == "__main__":
    unittest.main()

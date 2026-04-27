"""Unit tests for SGLang's FuzzyMatchProvider extensions.

Covers:
* The optional ``segments`` / ``layer_recompute_mask`` / ``quality_signals``
  fields on ``FuzzyMatchResult`` (back-compat with TokenBlockMatchProvider).
* ``FuzzyMatchSegment`` dataclass shape.
* ``create_fuzzy_match_provider`` factory routing.
* ``FuzzyMatchConfig`` validation for the new SemanticEmbedding fields.
* Forward-batch plumbing of segments / layer_recompute_mask.
* ``ModelRunner._copy_kv_with_rope_correction`` honors layer_recompute_mask
  using a hand-rolled mock pool (no GPU / model).

These tests do NOT require sglang_serve / a real GPU. They run under plain
``pytest`` and exercise only the Python-level data flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pytest
import torch

from sglang.srt.mem_cache.fuzzy_match.config import FuzzyMatchConfig
from sglang.srt.mem_cache.fuzzy_match.fuzzy_match_provider import (
    FuzzyMatchProvider,
    FuzzyMatchResult,
    FuzzyMatchSegment,
    QualitySignals,
    create_fuzzy_match_provider,
)


# ----------------------------------------------------------------------
# FuzzyMatchResult / FuzzyMatchSegment / QualitySignals dataclasses
# ----------------------------------------------------------------------


class TestFuzzyMatchResultLegacyShape:
    """TokenBlockMatchProvider's contract: optional fields default to None."""

    def test_legacy_fields_only(self):
        r = FuzzyMatchResult(
            cached_token_count=8,
            cached_token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            prompt_token_count=8,
            kv_cache_indices=torch.zeros(8, dtype=torch.int64),
            position_offset=0,
            cached_start_pos=4,
        )
        assert r.segments is None
        assert r.layer_recompute_mask is None
        assert r.quality_signals is None
        assert r._match_entry is None


class TestFuzzyMatchResultExtensions:
    def test_segments_carry_donor_id_and_per_segment_mask(self):
        seg = FuzzyMatchSegment(
            donor_kv_indices=torch.tensor([10, 11, 12], dtype=torch.int64),
            target_positions=torch.tensor([0, 1, 2], dtype=torch.int64),
            donor_positions=torch.tensor([5, 6, 7], dtype=torch.int64),
            donor_req_id="donor-A",
            layer_recompute_mask=[True, False, False, True],
        )
        assert seg.donor_req_id == "donor-A"
        assert seg.layer_recompute_mask == [True, False, False, True]

    def test_quality_signals_carry_rejection_reason(self):
        qs = QualitySignals(
            cosine_similarity=0.72,
            reuse_ratio=0.45,
            confidence_tier="fuzzy",
            passed_quality_gate=False,
            rejection_reason="below_min_reuse_ratio",
        )
        assert qs.passed_quality_gate is False
        assert qs.rejection_reason == "below_min_reuse_ratio"

    def test_full_semantic_result(self):
        r = FuzzyMatchResult(
            cached_token_count=6,
            cached_token_ids=[1, 2, 3, 4, 5, 6],
            prompt_token_count=6,
            kv_cache_indices=torch.empty(0, dtype=torch.int64),
            position_offset=0,
            segments=[
                FuzzyMatchSegment(
                    donor_kv_indices=torch.tensor([10, 11, 12], dtype=torch.int64),
                    target_positions=torch.tensor([0, 1, 2], dtype=torch.int64),
                    donor_positions=torch.tensor([5, 6, 7], dtype=torch.int64),
                    donor_req_id="donor-A",
                ),
                FuzzyMatchSegment(
                    donor_kv_indices=torch.tensor([20, 21, 22], dtype=torch.int64),
                    target_positions=torch.tensor([3, 4, 5], dtype=torch.int64),
                    donor_positions=torch.tensor([0, 1, 2], dtype=torch.int64),
                    donor_req_id="donor-A",
                ),
            ],
            layer_recompute_mask=[True, False, False, False, False, False, False, True],
            quality_signals=QualitySignals(
                cosine_similarity=0.92,
                reuse_ratio=0.85,
                confidence_tier="exact",
                passed_quality_gate=True,
            ),
        )
        assert len(r.segments) == 2
        assert sum(r.layer_recompute_mask) == 2  # only first/last recompute


# ----------------------------------------------------------------------
# Factory routing
# ----------------------------------------------------------------------


class TestProviderFactory:
    def test_disabled_returns_none(self):
        assert create_fuzzy_match_provider(FuzzyMatchConfig(enable_fuzzy_match=False)) is None

    def test_token_block_match_routes_to_correct_class(self):
        cfg = FuzzyMatchConfig(
            enable_fuzzy_match=True,
            fuzzy_match_provider="TokenBlockMatch",
        )
        provider = create_fuzzy_match_provider(cfg)
        assert provider is not None
        assert provider.__class__.__name__ == "TokenBlockMatchProvider"

    def test_semantic_embedding_imports_optional_dep(self, monkeypatch):
        """Without semblend installed, SemanticEmbedding raises ImportError clearly."""
        # Force the lazy import to fail. We do this by monkey-patching the
        # constructed class's import path.
        import sys
        # If semblend is installed in this env, skip — the test only proves the
        # error path. The success path is exercised in semblend's own tests.
        if "semblend" in sys.modules or _semblend_importable():
            pytest.skip("semblend importable in this env; skipping import-failure path")

        cfg = FuzzyMatchConfig(
            enable_fuzzy_match=True,
            fuzzy_match_provider="SemanticEmbedding",
            model_arch="llama",
        )
        with pytest.raises(ImportError):
            create_fuzzy_match_provider(cfg)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError):
            FuzzyMatchConfig(
                enable_fuzzy_match=True,
                fuzzy_match_provider="DoesNotExist",
            )


def _semblend_importable() -> bool:
    try:
        import semblend  # noqa: F401
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------
# FuzzyMatchConfig validation
# ----------------------------------------------------------------------


class TestFuzzyMatchConfigValidation:
    def test_gateway_requires_url(self):
        with pytest.raises(ValueError, match="gateway_url is required"):
            FuzzyMatchConfig(
                enable_fuzzy_match=True,
                fuzzy_match_provider="SemanticEmbedding",
                embedding_backend="gateway",
                gateway_url=None,
            )

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError, match="embedding_backend"):
            FuzzyMatchConfig(
                enable_fuzzy_match=True,
                fuzzy_match_provider="SemanticEmbedding",
                embedding_backend="bogus",
            )

    def test_min_reuse_ratio_bounds(self):
        with pytest.raises(ValueError, match="fuzzy_min_reuse_ratio"):
            FuzzyMatchConfig(
                enable_fuzzy_match=True,
                fuzzy_match_provider="SemanticEmbedding",
                fuzzy_min_reuse_ratio=0.0,  # must be > 0
            )
        with pytest.raises(ValueError, match="fuzzy_min_reuse_ratio"):
            FuzzyMatchConfig(
                enable_fuzzy_match=True,
                fuzzy_match_provider="SemanticEmbedding",
                fuzzy_min_reuse_ratio=1.5,
            )

    def test_defaults_pass_validation(self):
        # No exception
        FuzzyMatchConfig(enable_fuzzy_match=True)


# ----------------------------------------------------------------------
# ForwardBatch plumbing
# ----------------------------------------------------------------------


class TestForwardBatchPlumbing:
    def test_optional_fields_default_to_none(self):
        try:
            from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        except Exception as e:  # pragma: no cover — heavy import in some envs
            pytest.skip(f"forward_batch_info import unavailable: {e}")

        fields = {f.name: f for f in ForwardBatch.__dataclass_fields__.values()}
        assert "fuzzy_segments" in fields
        assert "fuzzy_layer_recompute_mask" in fields
        # Defaults are explicitly set to None in the dataclass body.
        assert fields["fuzzy_segments"].default is None
        assert fields["fuzzy_layer_recompute_mask"].default is None


# ----------------------------------------------------------------------
# ModelRunner._copy_kv_with_rope_correction with a fake pool
# ----------------------------------------------------------------------


class _FakePool:
    """Minimal stand-in for SGLang's KV pool — supports k_buffer/v_buffer indexing."""

    def __init__(self, layer_num: int, num_slots: int, num_heads: int, head_dim: int):
        self.layer_num = layer_num
        self.k_buffer = [
            torch.zeros(num_slots, num_heads, head_dim, dtype=torch.float32)
            for _ in range(layer_num)
        ]
        self.v_buffer = [
            torch.zeros(num_slots, num_heads, head_dim, dtype=torch.float32)
            for _ in range(layer_num)
        ]


class _FakeRotaryEmb:
    """Identity rotary: cos=1, sin=0 so RoPE correction is the identity transform."""

    def __init__(self, max_pos: int, head_dim: int):
        self.is_neox_style = True
        self.rotary_dim = head_dim
        # cos_sin_cache layout: [max_pos, 2 * (head_dim // 2)] per SGLang
        # convention -> cos in first half, sin in second half.
        self.cos_sin_cache = torch.cat(
            [
                torch.ones(max_pos, head_dim // 2, dtype=torch.float32),  # cos
                torch.zeros(max_pos, head_dim // 2, dtype=torch.float32),  # sin
            ],
            dim=-1,
        )


def _identity_apply_rotary(x, cos, sin, is_neox_style):
    """Stub that bypasses RoPE entirely (returns x unchanged)."""
    return x


def _identity_reverse_rotary(x, cos, sin, is_neox_style):
    return x


class TestCopyKVWithRopeCorrection:
    """Behavioral check on the model_runner helper, sans GPU.

    Uses an identity rotary so correction is the identity: K_new == K_old.
    Verifies layer_recompute_mask zeroes the flagged layer and copies the
    others.
    """

    def _fixture(self, layer_num=4, num_slots=32, num_heads=2, head_dim=8):
        pool = _FakePool(layer_num, num_slots, num_heads, head_dim)
        rotary = _FakeRotaryEmb(max_pos=64, head_dim=head_dim)
        # Pre-populate donor slots 10..13 with distinguishable values.
        for layer_id in range(layer_num):
            pool.k_buffer[layer_id][10:14] = layer_id + 1.0
            pool.v_buffer[layer_id][10:14] = -(layer_id + 1.0)
        return pool, rotary

    def test_copies_kv_when_no_mask(self):
        pool, rotary = self._fixture()

        from sglang.srt.mem_cache.fuzzy_match.rope_correction import (
            copy_kv_with_rope_correction,
        )

        old_locs = torch.tensor([10, 11, 12, 13], dtype=torch.long)
        new_locs = torch.tensor([20, 21, 22, 23], dtype=torch.long)
        old_pos = torch.tensor([5, 6, 7, 8], dtype=torch.long)
        new_pos = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        copy_kv_with_rope_correction(
            pool=pool,
            rotary_emb=rotary,
            old_locs=old_locs,
            new_locs=new_locs,
            old_positions=old_pos,
            new_positions=new_pos,
            layer_recompute_mask=None,
            apply_rotary_emb=_identity_apply_rotary,
            reverse_rotary_emb=_identity_reverse_rotary,
        )

        # Identity rotary means K is unchanged; V is a direct copy.
        for layer_id in range(pool.layer_num):
            torch.testing.assert_close(
                pool.k_buffer[layer_id][20:24],
                pool.k_buffer[layer_id][10:14],
            )
            torch.testing.assert_close(
                pool.v_buffer[layer_id][20:24],
                pool.v_buffer[layer_id][10:14],
            )

    def test_layer_mask_zeroes_flagged_layers(self):
        pool, rotary = self._fixture(layer_num=4)
        from sglang.srt.mem_cache.fuzzy_match.rope_correction import (
            copy_kv_with_rope_correction,
        )

        old_locs = torch.tensor([10, 11, 12, 13], dtype=torch.long)
        new_locs = torch.tensor([20, 21, 22, 23], dtype=torch.long)
        old_pos = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        new_pos = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        # Recompute layers 0 and 3 (bathtub-curve early/late).
        mask = [True, False, False, True]

        copy_kv_with_rope_correction(
            pool=pool,
            rotary_emb=rotary,
            old_locs=old_locs,
            new_locs=new_locs,
            old_positions=old_pos,
            new_positions=new_pos,
            layer_recompute_mask=mask,
            apply_rotary_emb=_identity_apply_rotary,
            reverse_rotary_emb=_identity_reverse_rotary,
        )

        # Layer 0: zeroed (recompute requested)
        assert torch.all(pool.k_buffer[0][20:24] == 0)
        assert torch.all(pool.v_buffer[0][20:24] == 0)
        # Layer 1: copied
        torch.testing.assert_close(
            pool.k_buffer[1][20:24], pool.k_buffer[1][10:14]
        )
        # Layer 2: copied
        torch.testing.assert_close(
            pool.v_buffer[2][20:24], pool.v_buffer[2][10:14]
        )
        # Layer 3: zeroed (recompute requested)
        assert torch.all(pool.k_buffer[3][20:24] == 0)
        assert torch.all(pool.v_buffer[3][20:24] == 0)

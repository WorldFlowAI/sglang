# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exact-content fuzzy-match provider.

Finds donor KV for content that is byte-identical to the current prompt's
unmatched tail but sits at a *different* offset than where it was originally
computed — as opposed to ``SemanticEmbeddingProvider``, which matches
merely-similar content and is explicitly not lossless. This provider's own
correctness bar is exact reuse: the token-ID equality check on every hit is
mandatory, not a nice-to-have — a fingerprint collision must never be trusted
without confirming the underlying token IDs actually match.

Current limitation: single-chunk, non-segmented matches only
(``FuzzyMatchResult.segments=None``, realized via
``FuzzyKVRealizer._realize_contiguous``). Multi-chunk / N:M segment matches
are a natural follow-up, not required for the mechanism to be correct.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import msgspec
import torch

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.mem_cache.fuzzy_match.chunker import SINK_TOKENS, chunk_tokens
from sglang.srt.mem_cache.fuzzy_match.config import FuzzyMatchConfig
from sglang.srt.mem_cache.fuzzy_match.fuzzy_match_provider import (
    FuzzyMatchProvider,
    FuzzyMatchResult,
)

logger = logging.getLogger(__name__)

# (extra_key, fingerprint) -> candidates; a list because two different
# chunks can share a fingerprint (rare, but the point of the equality check
# below is to never trust the hash alone regardless of how rare).
_StoreKey = Tuple[Optional[str], int]


class _ChunkEntry(msgspec.Struct):
    """One registered donor chunk."""

    token_ids: List[int]
    kv_indices: torch.Tensor
    start_pos: int  # p_src: absolute position this chunk was computed at
    donor_last_node_id: Optional[int] = None


class ExactHashProvider(FuzzyMatchProvider):
    """Content-defined-chunking + exact-hash donor matching."""

    def __init__(self, config: FuzzyMatchConfig):
        super().__init__(config)
        self._store: Dict[_StoreKey, List[_ChunkEntry]] = {}
        # request_id -> keys registered by that request's own
        # cache_on_request_finished call, so the immediately-following
        # on_donor_inserted callback can attach the real TreeNode id.
        self._pending_by_request: Dict[str, List[_StoreKey]] = {}
        logger.info("ExactHashProvider initialized")

    # ------------------------------------------------------------------
    # FuzzyMatchProvider contract
    # ------------------------------------------------------------------

    def cache_on_request_finished(
        self,
        request,
        token_ids: List[int],
        kv_cache: torch.Tensor,
        cache_start_pos: int,
        cache_end_pos: int,
        radix_tree=None,
    ) -> bool:
        request_id = _request_id(request)
        if request_id is None or request_id.startswith(HEALTH_CHECK_RID_PREFIX):
            return False
        if cache_end_pos <= cache_start_pos:
            return False

        # Never register content whose original occurrence started inside
        # the attention-sink zone — sink behavior is position-driven, not
        # reliably content-only, so a hash match there isn't trustworthy.
        chunk_region_start = max(cache_start_pos, SINK_TOKENS)
        if chunk_region_start >= cache_end_pos:
            return False

        extra_key = getattr(request, "extra_key", None)
        region = token_ids[chunk_region_start:cache_end_pos]
        chunks = chunk_tokens(region)

        registered_keys: List[_StoreKey] = []
        for c in chunks:
            abs_start = chunk_region_start + c.start
            abs_end = chunk_region_start + c.end
            entry = _ChunkEntry(
                token_ids=c.token_ids,
                kv_indices=kv_cache[abs_start:abs_end].detach().clone(),
                start_pos=abs_start,
            )
            key = (extra_key, c.fingerprint)
            self._store.setdefault(key, []).append(entry)
            registered_keys.append(key)

        if registered_keys:
            self._pending_by_request[request_id] = registered_keys
        logger.info(
            "[EXACT_HASH] cache_on_request_finished: rid=%s tokens=%d chunks=%d",
            request_id,
            cache_end_pos - cache_start_pos,
            len(chunks),
        )
        return bool(registered_keys)

    def match_on_prefix_miss(
        self,
        prompt_token_ids: List[int],
        already_matched_len: int,
        request=None,
        extra_key=None,
    ) -> Optional[FuzzyMatchResult]:
        tail = prompt_token_ids[already_matched_len:]
        chunks = chunk_tokens(tail)
        if not chunks:
            return None

        # Only the first chunk of the unmatched tail. It always starts at
        # offset 0 of `tail`, so its target position in the current prompt
        # is exactly `already_matched_len` — no extra offset math needed
        # for the single-chunk case.
        c = chunks[0]
        candidates = self._store.get((extra_key, c.fingerprint))
        if not candidates:
            return None

        entry = _first_equal_match(candidates, c.token_ids)
        if entry is None:
            logger.debug(
                "[EXACT_HASH] fingerprint hit but token-ID mismatch — "
                "hash collision, never trusting the hash alone"
            )
            return None

        return FuzzyMatchResult(
            cached_token_count=len(entry.token_ids),
            cached_token_ids=entry.token_ids,
            prompt_token_count=len(prompt_token_ids),
            kv_cache_indices=entry.kv_indices,
            position_offset=already_matched_len - entry.start_pos,
            cached_start_pos=entry.start_pos,
            segments=None,
            donor_last_node_id=entry.donor_last_node_id,
        )

    def on_donor_inserted(self, request, donor_last_node_id: int) -> None:
        request_id = _request_id(request)
        if request_id is None:
            return
        keys = self._pending_by_request.pop(request_id, None)
        if not keys:
            return
        for key in keys:
            for entry in self._store.get(key, []):
                if entry.donor_last_node_id is None:
                    entry.donor_last_node_id = donor_last_node_id

    def on_cache_reset(self) -> None:
        self._store.clear()
        self._pending_by_request.clear()
        logger.info("[EXACT_HASH] cleared store on cache reset")


def _request_id(request) -> Optional[str]:
    rid = getattr(request, "rid", None) or getattr(request, "request_id", None)
    return str(rid) if rid is not None else None


def _first_equal_match(
    candidates: List[_ChunkEntry], token_ids: List[int]
) -> Optional[_ChunkEntry]:
    """Mandatory token-ID equality check — never trust the hash alone."""
    for entry in candidates:
        if entry.token_ids == token_ids:
            return entry
    return None

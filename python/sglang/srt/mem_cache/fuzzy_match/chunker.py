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

"""Content-defined chunking (CDC) over token-ID sequences.

Boundaries depend only on the last ``WINDOW_SIZE`` tokens, not on absolute
position or anything upstream — the property RadixAttention's chained hash
lacks by design. Used by ``ExactHashProvider`` to find donor spans
independent of where they sit in the current prompt.

The chunking parameters below (window size, boundary mask, chunk-size
clamp, sink carve-out) follow the methodology in Ma, Eitzinger, and
Köstler, "Irminsul: MLA-Native Position-Independent Caching for Agentic
LLM Serving" (arXiv:2605.05696), Section 4 and Appendix B. That paper's
own reuse mechanism (RoPE delta-rotation over MLA's decomposed KV) is
unrelated to this module — only its CDC/chunking layer is used here.
"""

from __future__ import annotations

import random
from typing import List, NamedTuple, Sequence

_MASK64 = (1 << 64) - 1

# Gear-hash rolling window, in tokens. A boundary decision only ever depends
# on the last WINDOW_SIZE tokens (Gear hash's left-shift makes older tokens'
# contributions fall off the top of a 64-bit accumulator automatically —
# no explicit "remove the oldest byte" step needed, unlike Rabin-Karp).
WINDOW_SIZE = 64

# Tuned so a boundary fires with probability 2^-BOUNDARY_BITS per token,
# i.e. an expected chunk size of 2**BOUNDARY_BITS tokens. k=7 is the value
# the cited paper's mask-exponent ablation (Appendix B) settled on — wider
# masks push expected chunk length above MAX_CHUNK_TOKENS, which clamps
# chunks to fixed-size and degenerates back to the fixed-block hashing this
# design exists to avoid.
BOUNDARY_BITS = 7  # 2**7 = 128
_BOUNDARY_MASK = (1 << BOUNDARY_BITS) - 1

MIN_CHUNK_TOKENS = 32
MAX_CHUNK_TOKENS = 512

# Content before this position is never chunked/cached — attention-sink
# territory: early positions absorb a disproportionate, content-independent
# share of attention regardless of what's actually there, so a content hash
# match in that zone isn't a trustworthy signal. 32 tokens matches the
# cited paper's own sink-extent measurement (Section 7.1).
SINK_TOKENS = 32

# Fixed seed: the Gear table must be identical across the write side
# (cache_on_request_finished) and the read side (match_on_prefix_miss),
# and across process restarts, or chunk boundaries silently stop lining up.
_GEAR_SEED = 0x516D5A4F2F3A7C5C
_GEAR_TABLE_SIZE = 1 << 16  # token_id % this size indexes the table


def _build_gear_table() -> List[int]:
    rng = random.Random(_GEAR_SEED)
    return [rng.getrandbits(64) for _ in range(_GEAR_TABLE_SIZE)]


_GEAR_TABLE = _build_gear_table()

try:
    import xxhash

    def _real_fingerprint(token_ids: Sequence[int]) -> int:
        h = xxhash.xxh64(seed=0)
        h.update(_tokens_to_bytes(token_ids))
        return h.intdigest()

except ImportError:
    import hashlib

    def _real_fingerprint(token_ids: Sequence[int]) -> int:
        # xxhash isn't installed in this environment; blake2b is stdlib,
        # fast enough for a fingerprint (not a security boundary), and
        # gives the same "cheap, non-cryptographic-role" fingerprint
        # xxHash64 would — the token-ID equality check downstream is what
        # actually guards correctness, not this hash's collision odds.
        digest = hashlib.blake2b(
            _tokens_to_bytes(token_ids), digest_size=8
        ).digest()
        return int.from_bytes(digest, "little")


def _fingerprint(token_ids: Sequence[int]) -> int:
    # Test-only: force every chunk to the same fingerprint so a real E2E
    # test can exercise the mandatory token-ID equality-check fallback
    # (never trust the hash alone) without needing an astronomically
    # unlikely real collision. Off by default in production.
    from sglang.srt.environ import envs

    if envs.SGLANG_TEST_FUZZY_FORCE_HASH_COLLISION.get():
        return 0
    return _real_fingerprint(token_ids)


def _tokens_to_bytes(token_ids: Sequence[int]) -> bytes:
    return b"".join(t.to_bytes(4, "little", signed=False) for t in token_ids)


class Chunk(NamedTuple):
    """One content-defined chunk: token IDs plus its offset within the
    sequence that was chunked (not an absolute sequence position — callers
    add their own base offset)."""

    start: int  # offset into the chunked sequence, inclusive
    end: int  # offset into the chunked sequence, exclusive
    token_ids: List[int]
    fingerprint: int


def chunk_tokens(token_ids: Sequence[int]) -> List[Chunk]:
    """Split ``token_ids`` into content-defined chunks.

    Boundaries are declared where a Gear-hash rolling hash over the last
    ``WINDOW_SIZE`` tokens has its low ``BOUNDARY_BITS`` bits all zero,
    clamped to ``[MIN_CHUNK_TOKENS, MAX_CHUNK_TOKENS]``. The same content at
    a different position in a different call produces the same boundaries
    and the same per-chunk fingerprint, as long as ``WINDOW_SIZE`` tokens of
    identical context precede each boundary.
    """
    n = len(token_ids)
    chunks: List[Chunk] = []
    chunk_start = 0
    h = 0
    for i, tok in enumerate(token_ids):
        h = ((h << 1) + _GEAR_TABLE[tok % _GEAR_TABLE_SIZE]) & _MASK64
        chunk_len = i - chunk_start + 1
        if chunk_len < MIN_CHUNK_TOKENS:
            continue
        at_boundary = (i - chunk_start + 1 >= WINDOW_SIZE) and (
            h & _BOUNDARY_MASK == 0
        )
        if at_boundary or chunk_len >= MAX_CHUNK_TOKENS:
            span = token_ids[chunk_start : i + 1]
            chunks.append(
                Chunk(
                    start=chunk_start,
                    end=i + 1,
                    token_ids=list(span),
                    fingerprint=_fingerprint(span),
                )
            )
            chunk_start = i + 1
            h = 0

    # Trailing remainder: the rolling hash may never hit a boundary before
    # the input runs out, regardless of how much is left — this is not
    # bounded to be small. If what's left is at least MIN_CHUNK_TOKENS,
    # emit it as a final chunk (standard CDC practice: the last chunk of
    # any given input is "whatever's left" once you hit the end); only drop
    # it when it's genuinely too small to amortize a lookup. This only
    # affects the *final* chunk of whatever span is being chunked — content
    # that isn't at the tail end still gets purely content-defined
    # boundaries, unaffected by this.
    if len(token_ids) - chunk_start >= MIN_CHUNK_TOKENS:
        span = token_ids[chunk_start:]
        chunks.append(
            Chunk(
                start=chunk_start,
                end=len(token_ids),
                token_ids=list(span),
                fingerprint=_fingerprint(span),
            )
        )

    return chunks

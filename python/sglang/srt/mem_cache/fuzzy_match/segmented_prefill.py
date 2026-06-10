"""Segmented semantic prefill planning primitives.

This module intentionally does not mutate scheduler state. It describes the
contract needed by an execution backend that can mix exact prefix hits, donor
KV spans, and freshly computed prompt holes without pretending the donor spans
are a longer exact prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

from sglang.srt.mem_cache.fuzzy_match.fuzzy_match_provider import FuzzyMatchSegment


SpanKind = Literal["exact_prefix", "fresh", "donor"]
BackendStepKind = Literal["compute_fresh", "realize_donor"]


@dataclass(frozen=True)
class SegmentedPrefillSpan:
    """One logical span in a segmented prefill request."""

    kind: SpanKind
    start_pos: int
    end_pos: int
    donor_positions: tuple[int, ...] = ()
    donor_kv_indices: tuple[int, ...] = ()
    donor_req_id: Optional[str] = None
    layer_recompute_mask: Optional[tuple[bool, ...]] = None

    @property
    def length(self) -> int:
        return self.end_pos - self.start_pos


@dataclass(frozen=True)
class SegmentedPrefillBackendStep:
    """One left-to-right operation for a segmented prefill backend."""

    kind: BackendStepKind
    start_pos: int
    end_pos: int
    prefix_len_before: int
    donor_positions: tuple[int, ...] = ()
    donor_kv_indices: tuple[int, ...] = ()

    @property
    def length(self) -> int:
        return self.end_pos - self.start_pos


@dataclass(frozen=True)
class SegmentedSparsePrefillMetadata:
    """Single-pass sparse-query metadata for arbitrary-hole semantic prefill."""

    prompt_token_count: int
    exact_prefix_len: int
    fresh_positions: tuple[int, ...]
    donor_positions: tuple[int, ...]
    donor_kv_indices: tuple[int, ...]

    @property
    def fresh_token_count(self) -> int:
        return len(self.fresh_positions)

    @property
    def donor_token_count(self) -> int:
        return len(self.donor_positions)

    def input_token_ids(self, fill_ids: Sequence[int]) -> tuple[int, ...]:
        """Return the non-contiguous fresh input IDs for this sparse prefill."""
        if len(fill_ids) < self.prompt_token_count:
            raise ValueError("fill_ids is shorter than sparse prompt length")
        return tuple(int(fill_ids[pos]) for pos in self.fresh_positions)

    def token_level_causal_mask(self) -> tuple[tuple[bool, ...], ...]:
        """Return an exact fresh-query x logical-KV causal mask.

        Rows are fresh query target positions. Columns are logical prompt
        positions. This is intentionally token-level so arbitrary fresh/donor
        holes are exact even when they do not align with future block sizes.
        """
        return tuple(
            tuple(col_pos <= query_pos for col_pos in range(self.prompt_token_count))
            for query_pos in self.fresh_positions
        )


@dataclass(frozen=True)
class SegmentedPrefillPlan:
    """Execution plan for non-prefix semantic KV reuse."""

    prompt_token_count: int
    exact_prefix_len: int
    spans: tuple[SegmentedPrefillSpan, ...]

    @property
    def donor_token_count(self) -> int:
        return sum(span.length for span in self.spans if span.kind == "donor")

    @property
    def fresh_token_count(self) -> int:
        return sum(span.length for span in self.spans if span.kind == "fresh")

    @property
    def donor_target_positions(self) -> tuple[int, ...]:
        positions: list[int] = []
        for span in self.spans:
            if span.kind == "donor":
                positions.extend(range(span.start_pos, span.end_pos))
        return tuple(positions)

    @property
    def fresh_target_positions(self) -> tuple[int, ...]:
        positions: list[int] = []
        for span in self.spans:
            if span.kind == "fresh":
                positions.extend(range(span.start_pos, span.end_pos))
        return tuple(positions)

    @property
    def prefix_contract_compatible(self) -> bool:
        """Whether donor spans can be represented as appended prefix indices.

        The current fuzzy path appends donor KV to ``prefix_indices`` and then
        computes ``fill_ids[len(prefix_indices):]``. That is geometrically safe
        only when donated target positions are exactly the next contiguous
        positions after the exact prefix. It is not sufficient for semantic
        correctness; it only means the scheduler will not skip fresh holes.
        """
        donor_count = self.donor_token_count
        if donor_count == 0:
            return True
        return self.donor_target_positions == tuple(
            range(self.exact_prefix_len, self.exact_prefix_len + donor_count)
        )

    @property
    def skipped_by_prefix_compression(self) -> tuple[int, ...]:
        """Fresh prompt positions lost by the current prefix-shaped shortcut."""
        compressed_prefix_end = min(
            self.prompt_token_count, self.exact_prefix_len + self.donor_token_count
        )
        compressed_range = set(range(self.exact_prefix_len, compressed_prefix_end))
        donor_positions = set(self.donor_target_positions)
        return tuple(sorted(compressed_range - donor_positions))

    @property
    def phased_dense_steps(self) -> tuple[SegmentedPrefillBackendStep, ...]:
        """Backend contract for dense segmented prefill.

        A dense MVP can consume these steps from left to right: run prefill for
        fresh spans, realize donor KV into recipient-owned slots, and advance
        the logical prefix length after each step. Sparse backends can consume
        the same spans with different metadata.
        """
        steps: list[SegmentedPrefillBackendStep] = []
        cursor = self.exact_prefix_len
        for span in self.spans:
            if span.kind == "exact_prefix":
                continue
            if span.start_pos != cursor:
                raise ValueError("segmented prefill spans must cover positions in order")
            if span.kind == "fresh":
                steps.append(
                    SegmentedPrefillBackendStep(
                        kind="compute_fresh",
                        start_pos=span.start_pos,
                        end_pos=span.end_pos,
                        prefix_len_before=cursor,
                    )
                )
            elif span.kind == "donor":
                steps.append(
                    SegmentedPrefillBackendStep(
                        kind="realize_donor",
                        start_pos=span.start_pos,
                        end_pos=span.end_pos,
                        prefix_len_before=cursor,
                        donor_positions=span.donor_positions,
                        donor_kv_indices=span.donor_kv_indices,
                    )
                )
            cursor = span.end_pos
        return tuple(steps)

    def next_backend_step(
        self, prefix_len: int
    ) -> Optional[SegmentedPrefillBackendStep]:
        """Return the next dense backend step at the current logical prefix."""
        for idx, span in enumerate(self.spans):
            if span.end_pos <= prefix_len:
                continue
            if span.kind == "exact_prefix":
                if prefix_len < span.end_pos:
                    raise ValueError("prefix_len is inside the exact prefix span")
                continue
            if span.start_pos > prefix_len:
                raise ValueError("segmented prefill prefix is before the next span")
            if span.kind == "fresh":
                return SegmentedPrefillBackendStep(
                    kind="compute_fresh",
                    start_pos=prefix_len,
                    end_pos=span.end_pos,
                    prefix_len_before=prefix_len,
                )
            if span.start_pos != prefix_len:
                raise ValueError("cannot partially realize a donor span")
            donor_positions = list(span.donor_positions)
            donor_kv_indices = list(span.donor_kv_indices)
            cursor = span.end_pos
            for later in self.spans[idx + 1 :]:
                if later.kind != "donor" or later.start_pos != cursor:
                    break
                donor_positions.extend(later.donor_positions)
                donor_kv_indices.extend(later.donor_kv_indices)
                cursor = later.end_pos
            return SegmentedPrefillBackendStep(
                kind="realize_donor",
                start_pos=span.start_pos,
                end_pos=cursor,
                prefix_len_before=prefix_len,
                donor_positions=tuple(donor_positions),
                donor_kv_indices=tuple(donor_kv_indices),
            )
        return None

    def has_donor_after(self, pos: int) -> bool:
        """Whether any donor span remains after ``pos``."""
        return any(span.kind == "donor" and span.start_pos >= pos for span in self.spans)

    def next_required_fresh_chunk_end(self, prefix_len: int) -> Optional[int]:
        """Return the next fresh-span end when a future donor requires a cut.

        Dense phased prefill can run normal prefill over fresh tokens until the
        next donor span starts. If there is no future donor after that fresh
        span, no forced cut is needed because the backend can finish the prompt
        normally.
        """
        for idx, span in enumerate(self.spans):
            if span.kind != "fresh":
                continue
            if span.start_pos <= prefix_len < span.end_pos:
                has_future_donor = any(
                    later.kind == "donor" for later in self.spans[idx + 1 :]
                )
                return span.end_pos if has_future_donor else None
        return None

    def leading_donor_end(self, prefix_len: int) -> Optional[int]:
        """Return the end of donor spans contiguous at ``prefix_len``."""
        cursor = prefix_len
        found = False
        for span in self.spans:
            if span.end_pos <= cursor:
                continue
            if span.start_pos != cursor:
                break
            if span.kind != "donor":
                break
            found = True
            cursor = span.end_pos
        return cursor if found else None

    def sparse_prefill_metadata(self) -> SegmentedSparsePrefillMetadata:
        """Return metadata for one-pass sparse-query segmented prefill."""
        donor_positions: list[int] = []
        donor_kv_indices: list[int] = []
        for span in self.spans:
            if span.kind != "donor":
                continue
            donor_positions.extend(range(span.start_pos, span.end_pos))
            if not span.donor_kv_indices:
                raise ValueError("donor span is missing donor_kv_indices")
            donor_kv_indices.extend(span.donor_kv_indices)

        return SegmentedSparsePrefillMetadata(
            prompt_token_count=self.prompt_token_count,
            exact_prefix_len=self.exact_prefix_len,
            fresh_positions=self.fresh_target_positions,
            donor_positions=tuple(donor_positions),
            donor_kv_indices=tuple(donor_kv_indices),
        )


def build_segmented_prefill_plan(
    *,
    prompt_token_count: int,
    exact_prefix_len: int,
    segments: Sequence[FuzzyMatchSegment],
) -> SegmentedPrefillPlan:
    """Build a prompt-position plan from provider-supplied donor segments."""
    if prompt_token_count < 0:
        raise ValueError("prompt_token_count must be non-negative")
    if exact_prefix_len < 0:
        raise ValueError("exact_prefix_len must be non-negative")
    if exact_prefix_len > prompt_token_count:
        raise ValueError("exact_prefix_len cannot exceed prompt_token_count")

    donor_spans = [
        _segment_to_span(segment, exact_prefix_len, prompt_token_count)
        for segment in segments
    ]
    donor_spans.sort(key=lambda span: span.start_pos)
    _validate_non_overlapping(donor_spans)

    spans: list[SegmentedPrefillSpan] = []
    if exact_prefix_len > 0:
        spans.append(
            SegmentedPrefillSpan(
                kind="exact_prefix",
                start_pos=0,
                end_pos=exact_prefix_len,
            )
        )

    cursor = exact_prefix_len
    for donor_span in donor_spans:
        if donor_span.start_pos > cursor:
            spans.append(
                SegmentedPrefillSpan(
                    kind="fresh",
                    start_pos=cursor,
                    end_pos=donor_span.start_pos,
                )
            )
        spans.append(donor_span)
        cursor = donor_span.end_pos

    if cursor < prompt_token_count:
        spans.append(
            SegmentedPrefillSpan(
                kind="fresh",
                start_pos=cursor,
                end_pos=prompt_token_count,
            )
        )

    return SegmentedPrefillPlan(
        prompt_token_count=prompt_token_count,
        exact_prefix_len=exact_prefix_len,
        spans=tuple(spans),
    )


def _segment_to_span(
    segment: FuzzyMatchSegment,
    exact_prefix_len: int,
    prompt_token_count: int,
) -> SegmentedPrefillSpan:
    target_positions = _as_int_tuple(segment.target_positions)
    if not target_positions:
        raise ValueError("fuzzy match segments must include target_positions")
    if not _is_contiguous(target_positions):
        raise ValueError("target_positions must be contiguous for one donor span")

    start_pos = target_positions[0]
    end_pos = target_positions[-1] + 1
    if start_pos < exact_prefix_len:
        raise ValueError("donor segment cannot overlap the exact prefix")
    if end_pos > prompt_token_count:
        raise ValueError("donor segment cannot exceed prompt_token_count")

    donor_positions = _as_int_tuple(segment.donor_positions)
    if donor_positions and len(donor_positions) != len(target_positions):
        raise ValueError("donor_positions length must match target_positions")

    donor_kv_indices = _as_int_tuple(segment.donor_kv_indices)
    if donor_kv_indices and len(donor_kv_indices) != len(target_positions):
        raise ValueError("donor_kv_indices length must match target_positions")

    layer_recompute_mask = (
        tuple(segment.layer_recompute_mask)
        if segment.layer_recompute_mask is not None
        else None
    )
    return SegmentedPrefillSpan(
        kind="donor",
        start_pos=start_pos,
        end_pos=end_pos,
        donor_positions=donor_positions,
        donor_kv_indices=donor_kv_indices,
        donor_req_id=segment.donor_req_id,
        layer_recompute_mask=layer_recompute_mask,
    )


def _validate_non_overlapping(spans: Sequence[SegmentedPrefillSpan]) -> None:
    last_end = -1
    for span in spans:
        if span.start_pos < last_end:
            raise ValueError("donor segments cannot overlap")
        last_end = span.end_pos


def _as_int_tuple(value) -> tuple[int, ...]:
    if value is None:
        return ()
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    return tuple(int(item) for item in value)


def _is_contiguous(values: Sequence[int]) -> bool:
    return all(right == left + 1 for left, right in zip(values, values[1:]))

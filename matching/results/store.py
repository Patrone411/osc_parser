from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from .types import CallKey, Interval, PerCallSignal
from .ops import coalesce_intervals, intervals_to_mask

class ResultStore:
    """
    Group hits by (block_label, call_index) -> list of PerCallSignal.
    Each PerCallSignal is for one (segment, binding) for that call.
    """
    def __init__(self) -> None:
        self.by_call: Dict[CallKey, List[PerCallSignal]] = {}

    def add_hits(
        self,
        *,
        call_key: CallKey,
        seg_id: str,
        binding: Dict[str, str],
        hits: List[Dict],
        T: Optional[int],
        coalesce: bool = True,
        build_mask: bool = False,
    ) -> None:
        """
        hits: list of { 't_start', 't_end', ... } for exactly this (segment, binding, call).
        """
        if not hits:
            return
        ivs = [Interval(int(h["t_start"]), int(h["t_end"])) for h in hits]
        ivs = coalesce_intervals(ivs) if coalesce else ivs

        # Guard on T (try to infer from hits if missing)
        t_max = max((iv.t1 for iv in ivs), default=-1)
        if T is None:
            T = t_max + 1 if t_max >= 0 else 0

        mask = intervals_to_mask(ivs, T) if build_mask else None
        sig = PerCallSignal(segment_id=seg_id, roles=dict(binding), T=int(T), intervals=ivs, mask=mask)
        self.by_call.setdefault(call_key, []).append(sig)

    def signals(self, call_key: CallKey) -> List[PerCallSignal]:
        return self.by_call.get(call_key, [])

    def ensure_masks(self) -> None:
        for bucket in self.by_call.values():
            for sig in bucket:
                if sig.mask is None:
                    sig.mask = intervals_to_mask(sig.intervals, sig.T)
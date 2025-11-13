from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict

from .types import BindingKey, PerCallSignal, BlockSignal, ResultsStore
from .interval_ops import intervals_to_mask, mask_to_intervals

def evaluate_parallel_block(
    store: ResultsStore, call_keys: List[Tuple[str, int]]
) -> Dict[BindingKey, BlockSignal]:
    """AND over calls: keep times where all calls are true for the same (segment,binding)."""
    bucket: Dict[BindingKey, List[PerCallSignal]] = defaultdict(list)
    for ck in call_keys:
        for s in store.signals(ck):
            bkey: BindingKey = (s.segment_id, tuple(sorted(s.roles.items())))
            bucket[bkey].append(s)

    out: Dict[BindingKey, BlockSignal] = {}
    for bkey, sigs in bucket.items():
        T = max(s.T for s in sigs)
        masks = []
        for s in sigs:
            masks.append(s.mask if s.mask is not None else intervals_to_mask(s.intervals, T))
        m = masks[0].copy()
        for mm in masks[1:]:
            m &= mm
        intervals = mask_to_intervals(m)
        seg_id, roles_tup = bkey
        out[bkey] = BlockSignal(
            segment_id=seg_id, roles=dict(roles_tup), T=T,
            mask=m, intervals=intervals
        )
    return out

def evaluate_serial_block(
    store: ResultsStore, call_keys: List[Tuple[str, int]], *, max_gap: int = 0
) -> Dict[BindingKey, BlockSignal]:
    """
    A then B then C… (same binding). Simple temporal sequencing:
    we dilate each previous mask forward by `max_gap` and intersect with the next.
    """
    if not call_keys:
        return {}

    # Seeds: first call’s union per binding
    seeds: Dict[BindingKey, List[PerCallSignal]] = {}
    for s in store.signals(call_keys[0]):
        bkey: BindingKey = (s.segment_id, tuple(sorted(s.roles.items())))
        seeds.setdefault(bkey, []).append(s)

    out: Dict[BindingKey, BlockSignal] = {}
    for bkey, sigs in seeds.items():
        T = max(s.T for s in sigs)
        m = np.zeros(T, dtype=bool)
        for s in sigs:
            m |= (s.mask if s.mask is not None else intervals_to_mask(s.intervals, T))

        ok = True
        for ck in call_keys[1:]:
            m_next = np.zeros(T, dtype=bool)
            for s in store.signals(ck):
                if (s.segment_id, tuple(sorted(s.roles.items()))) == bkey:
                    m_next |= (s.mask if s.mask is not None else intervals_to_mask(s.intervals, T))
            if not m_next.any():
                ok = False; break

            # "dilate forward" by max_gap (one simple shift)
            if max_gap > 0:
                shifted = np.zeros(T, dtype=bool)
                shifted[max_gap:] = m[:-max_gap]
                m = m | shifted

            m = m & m_next
            if not m.any():
                ok = False; break

        if ok and m.any():
            seg_id, roles_tup = bkey
            out[bkey] = BlockSignal(
                segment_id=seg_id, roles=dict(roles_tup), T=T,
                mask=m, intervals=mask_to_intervals(m)
            )
    return out
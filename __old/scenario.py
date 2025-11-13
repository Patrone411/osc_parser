from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from .types import BindingKey, BlockSignal, Interval

class Trace:
    def __init__(self, segment_id: str, roles: Dict[str,str],
                 block_intervals: List[Tuple[str, List[Interval]]]):
        self.segment_id = segment_id
        self.roles = roles
        self.block_intervals = block_intervals

def stitch_blocks_in_order(
    blocks_in_order: List[Tuple[str, Dict[BindingKey, BlockSignal]]],
    *, max_gap_between_blocks: int = 0
) -> List[Trace]:
    """
    Compose block results in order for consistent (segment,binding).
    Each step requires temporal progress within an optional max gap.
    """
    if not blocks_in_order:
        return []

    traces: List[Trace] = []
    first_label, first_map = blocks_in_order[0]
    for bkey, bsig in first_map.items():
        seg_id, roles_tup = bkey
        T = bsig.T
        m = bsig.mask
        if m is None:
            m = np.zeros(T, dtype=bool)
            for it in bsig.intervals:
                m[it.t0:it.t1+1] = True

        stitched = [(first_label, bsig.intervals)]
        ok = True

        for label, bmap in blocks_in_order[1:]:
            nxt = bmap.get(bkey)
            if nxt is None:
                ok = False; break

            mn = nxt.mask
            if mn is None:
                mn = np.zeros(T, dtype=bool)
                for it in nxt.intervals:
                    mn[it.t0:it.t1+1] = True

            if max_gap_between_blocks > 0:
                shifted = np.zeros_like(m)
                shifted[max_gap_between_blocks:] = m[:-max_gap_between_blocks]
                m = m | shifted

            m = m & mn
            if not m.any():
                ok = False; break

            # compress for display
            idx = np.where(m)[0]
            splits = np.where(np.diff(idx) != 1)[0] + 1
            groups = np.split(idx, splits)
            intervals = [Interval(int(g[0]), int(g[-1])) for g in groups if len(g) > 0]
            stitched.append((label, intervals))

        if ok:
            traces.append(Trace(seg_id, dict(roles_tup), stitched))
    return traces
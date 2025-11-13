from __future__ import annotations
from typing import List
import numpy as np
from .types import Interval

def coalesce_intervals(intervals: List[Interval]) -> List[Interval]:
    if not intervals:
        return []
    ivs = sorted(intervals, key=lambda x: (x.t0, x.t1))
    out = [ivs[0]]
    for cur in ivs[1:]:
        last = out[-1]
        if cur.t0 <= last.t1 + 1:  # touch/overlap -> merge
            out[-1] = Interval(last.t0, max(last.t1, cur.t1))
        else:
            out.append(cur)
    return out

def intervals_to_mask(intervals: List[Interval], T: int) -> np.ndarray:
    m = np.zeros((T,), dtype=bool)
    for iv in intervals:
        a = max(0, iv.t0); b = min(T - 1, iv.t1)
        if a <= b:
            m[a:b+1] = True
    return m

def mask_to_intervals(mask: np.ndarray) -> List[Interval]:
    mask = mask.astype(bool)
    if mask.size == 0:
        return []
    dif = np.diff(mask.astype(int), prepend=0, append=0)
    starts = list((dif == 1).nonzero()[0])
    ends   = list((dif == -1).nonzero()[0] - 1)
    return [Interval(int(s), int(e)) for s, e in zip(starts, ends)]
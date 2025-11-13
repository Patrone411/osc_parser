from __future__ import annotations
from typing import List, Tuple
import numpy as np
from .types import Interval

def merge_windows_to_intervals(windows: List[Tuple[int, int]]) -> List[Interval]:
    if not windows:
        return []
    wins = sorted(windows)
    merged = [Interval(*wins[0])]
    for a, b in wins[1:]:
        last = merged[-1]
        if a <= last.t1 + 1:
            merged[-1] = Interval(last.t0, max(last.t1, b))
        else:
            merged.append(Interval(a, b))
    return merged

def intervals_to_mask(intervals: List[Interval], T: int) -> np.ndarray:
    m = np.zeros(T, dtype=bool)
    for it in intervals:
        a = max(0, it.t0); b = min(T - 1, it.t1)
        if a <= b:
            m[a:b+1] = True
    return m

def mask_to_intervals(mask: np.ndarray) -> List[Interval]:
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    # split by gaps
    splits = np.where(np.diff(idx) != 1)[0] + 1
    groups = np.split(idx, splits)
    return [Interval(int(g[0]), int(g[-1])) for g in groups if len(g) > 0]
# osc_parser/matching/post/plan.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

@dataclass
class BlockPlan:
    label: str
    type: str                # "serial" | "parallel" | "one_of" (for later)
    indices: List[int]       # indices into your flat calls list
    duration_frames: int | None  # from block(duration: ...), if you carried it; else None

def build_block_plans(calls: List[dict]) -> Dict[str, BlockPlan]:
    """
    Group by block_label, preserve first-seen order of calls within each label.
    Assumes all calls with same label share the same block_type.
    """
    by_label: Dict[str, List[int]] = {}
    type_of: Dict[str, str] = {}
    dur_of: Dict[str, int | None] = {}
    for i, c in enumerate(calls):
        lbl = c.get("block_label") or "<none>"
        by_label.setdefault(lbl, []).append(i)
        bt = c.get("block_type") or "serial"
        type_of.setdefault(lbl, bt)
        # if you propagate block duration (in seconds) → convert to frames upstream
        dur_of.setdefault(lbl, None)
    out: Dict[str, BlockPlan] = {}
    for lbl, idxs in by_label.items():
        out[lbl] = BlockPlan(label=lbl, type=type_of[lbl], indices=idxs, duration_frames=dur_of[lbl])
    return out

def block_sequence(calls: List[dict]) -> List[str]:
    """
    Return the sequence of block labels in the order they first appear
    (for top-level do-serial across blocks).
    """
    seen = set()
    seq: List[str] = []
    for c in calls:
        lbl = c.get("block_label") or "<none>"
        if lbl not in seen:
            seen.add(lbl)
            seq.append(lbl)
    return seq

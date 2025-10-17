# osc_parser/matching/match_block.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .spec import BlockQuery


def _candidate_pairs(
    feats,
    Q: BlockQuery,
    pairs: Optional[List[Tuple[str, Optional[str]]]] = None,
) -> List[Tuple[str, Optional[str]]]:
    """
    Resolve which (ego,npc) bindings to try for this segment.
    Priority:
      1) Explicit 'pairs' passed by caller
      2) Q.npc_candidates (filter to actors present in this segment)
      3) Fallback: (ego, None)  — for non-relational calls
    """
    ego = Q.ego
    actors = set(getattr(feats, "actors", []) or [])
    if pairs is not None:
        # filter to those that exist in this segment & match ego
        out = []
        for e, n in pairs:
            if e != ego:
                continue
            if n is not None and n not in actors:
                continue
            out.append((e, n))
        if out:
            return out
        # fall through if empty
    if Q.npc_candidates:
        return [(ego, n) for n in Q.npc_candidates if n in actors]
    # default: non-relational
    return [(ego, None)]


def match_block(
    feats,
    Q: BlockQuery,
    fps: int = 10,
    pairs: Optional[List[Tuple[str, Optional[str]]]] = None,
    max_results: int = 5000,
) -> List[Dict[str, Any]]:
    """
    Generic sliding-window driver for a single BlockQuery over ONE segment.

    Slides an inclusive window [t0..t1] of length up to Q.duration_frames across time.
    If cfg.allow_shorter_end is True, accepts the earliest t1 in the allowed range
    that satisfies all checks for each t0/pair.

    Returns a list of hits:
      {"ego": ego, "npc": npc, "t_start": t0, "t_end": t1}
    """
    results: List[Dict[str, Any]] = []

    # Segment length
    T = int(getattr(feats, "T", 0) or 0)
    if T <= 0:
        return results

    ego = Q.ego
    if ego not in getattr(feats, "actors", []):
        return results

    # Resolve candidate (ego,npc) pairs for this segment
    cand_pairs = _candidate_pairs(feats, Q, pairs)

    # Duration settings
    D = max(1, int(Q.duration_frames))
    allow_shorter = bool(Q.cfg.get("allow_shorter_end", True))
    min_len = 1 if allow_shorter else D

    # Optional debug
    debug = bool(Q.cfg.get("debug_match_block", False))

    # Pre-calc last valid t0
    last_t0 = max(0, T - min_len)

    for (ego_id, npc_id) in cand_pairs:
        # Scan start times
        for t0 in range(0, last_t0 + 1):
            # Determine end bounds for this start
            t1_min = min(T - 1, t0 + min_len - 1)
            t1_max = min(T - 1, t0 + D - 1)

            found_t1: Optional[int] = None

            # Try earliest acceptable end first
            for t1 in range(t1_min, t1_max + 1):
                all_ok = True
                for check in Q.checks:
                    try:
                        if not check(feats, ego_id, npc_id, t0, t1, Q.cfg):
                            all_ok = False
                            break
                    except Exception as ex:
                        if debug:
                            print(f"[match_block] check exception @({t0},{t1}) ego={ego_id} npc={npc_id}: {ex}")
                        all_ok = False
                        break

                if all_ok:
                    found_t1 = t1
                    break

            if found_t1 is not None:
                results.append({
                    "ego": ego_id,
                    "npc": npc_id,
                    "t_start": t0,
                    "t_end": found_t1,
                })
                if len(results) >= max_results:
                    return results

    return results

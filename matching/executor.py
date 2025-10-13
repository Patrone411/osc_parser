# osc_parser/matching/executor.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
from .features import TagFeatures
from .predicates import BlockQuery

def _run_checks(feats: TagFeatures, Q: BlockQuery, ego: str, npc: str, t0: int, t1: int) -> bool:
    for chk in Q.checks:
        if not chk(feats, ego, npc, t0, t1, Q.cfg):
            return False
    return True

def match_block(
    feats: TagFeatures,
    Q: BlockQuery,
    fps: int = 10,
    pairs: Optional[List[Tuple[str, str]]] = None,
    max_results: int = 5000,
) -> List[Dict[str, Any]]:
    """
    Slide a window of length Q.duration_frames across all (ego, npc) pairs,
    or over the given 'pairs'. If Q.npc_candidates is set, restrict to those.
    With Q.allow_shorter_end=True, we accept earliest t1 that satisfies checks.
    Returns: [{"ego":..., "npc":..., "t_start":t0, "t_end":t1}, ...]
    """
    T = feats.T
    dur = max(1, int(Q.duration_frames))
    results: List[Dict[str, Any]] = []

    # Determine candidate pairs
    if pairs is not None:
        cand_pairs = pairs
    else:
        if Q.npc_candidates:
            cand_pairs = [(Q.ego, npc) for npc in Q.npc_candidates if npc != Q.ego]
        else:
            cand_pairs = [(Q.ego, a) for a in feats.actors if a != Q.ego]

    # Sliding windows
    min_len = 1 if Q.allow_shorter_end else dur
    last_t0 = max(0, T - min_len)
    for ego, npc in cand_pairs:
        for t0 in range(0, last_t0 + 1):
            # earliest allowed t1 and latest (bounded by T)
            t1_min = min(T - 1, t0 + min_len - 1)
            t1_max = min(T - 1, t0 + dur - 1)

            found = None
            for t1 in range(t1_min, t1_max + 1):
                if _run_checks(feats, Q, ego, npc, t0, t1):
                    found = t1
                    break
            if found is not None:
                results.append({"ego": ego, "npc": npc, "t_start": t0, "t_end": found})
                if len(results) >= max_results:
                    return results
    return results

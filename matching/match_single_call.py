# osc_parser/matching/single_call_fast.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

# Compiler (call → BlockQuery) and generic sliding-window matcher
from .spec import build_block_query, BlockQuery
from .match_block import match_block


def match_single_call(
    feats,
    call: Dict[str, Any],
    fps: int = 10,
    cfg: Optional[Dict[str, Any]] = None,
    pairs: Optional[List[Tuple[str, Optional[str]]]] = None,
    max_results: int = 5000,
) -> List[Dict[str, Any]]:
    """
    Thin adapter:
      1) compile a single normalized call dict into a BlockQuery
      2) run the generic match_block() over one segment's TagFeatures

    Args:
        feats: TagFeatures for ONE segment
        call:  normalized call (from constraints_from_ir(...)"calls_flat"[i])
        fps:   sampling rate (frames per second)
        cfg:   optional dict of tolerances/knobs (merged with spec defaults)
        pairs: optional list of (ego, npc) candidate bindings to restrict search.
               If None, we'll use the compiler's referenced actors (if any).
        max_results: cap on matches to return for this segment.

    Returns:
        List[{"ego": str, "npc": Optional[str], "t_start": int, "t_end": int}]
    """
    Q, candidate_pairs = build_block_query(call, fps=fps, cfg=cfg or {})
    use_pairs = pairs if pairs is not None else candidate_pairs
    return match_block(feats, Q, fps=fps, pairs=use_pairs, max_results=max_results)


def match_single_call_across_segments(
    feats_by_seg: Dict[str, Any],
    call: Dict[str, Any],
    fps: int = 10,
    cfg: Optional[Dict[str, Any]] = None,
    pairs: Optional[List[Tuple[str, Optional[str]]]] = None,
    max_results_per_seg: int = 5000,
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper to run a single call across MANY segments.

    Args:
        feats_by_seg: dict {segment_id: TagFeatures}
        call:          normalized call (from constraints_from_ir)
        fps:           frames per second
        cfg:           matcher/compiler config (merged with spec defaults)
        pairs:         optional fixed candidate bindings (ego,npc) to use
        max_results_per_seg: cap per segment

    Returns:
        List of hits with 'segment' annotated.
    """
    Q, candidate_pairs = build_block_query(call, fps=fps, cfg=cfg or {})
    use_pairs = pairs if pairs is not None else candidate_pairs

    hits: List[Dict[str, Any]] = []
    for seg_id, feats in feats_by_seg.items():
        seg_hits = match_block(feats, Q, fps=fps, pairs=use_pairs, max_results=max_results_per_seg)
        for h in seg_hits:
            h["segment"] = seg_id
        hits.extend(seg_hits)
    return hits

# osc_parser/matching/results/collect.py  (or wherever your current function lives)
from __future__ import annotations
from typing import Dict, List
from .types import CallKey, PerCallSignal, ResultsStore
from .interval_ops import merge_windows_to_intervals, intervals_to_mask

# same helpers as your run()
from osc_parser.role_planning import (
    make_type_to_candidates, roles_used_by_call, role_domains_from_segment,
    prefilter_domains, build_overlap_matrix, enumerate_bindings,
)
from osc_parser.matching.match_single_call import match_for_binding

def collect_results(
    h,                      # OSCTestHarness instance
    calls: List[Dict],
    *,
    max_results_per_binding: int = 10_000,
) -> ResultsStore:
    """
    Step-2 collector: for each (call × segment × binding) run the matcher and
    store an atomic PerCallSignal (intervals + mask). This is what Step-3 uses.
    """
    store = ResultsStore()

    for ci, call in enumerate(calls):
        block_label = call.get("block_label") or ""
        call_key: CallKey = (block_label, ci)

        roles = sorted(roles_used_by_call(call))   # ego + any references
        ego_role = call.get("actor")
        others   = [r for r in roles if r != ego_role]
        require_pairs = [(ego_role, r) for r in others] if others else []

        for seg_id, feats in (h.feats_by_seg or {}).items():
            resolver = make_type_to_candidates(feats)
            domains  = role_domains_from_segment(
                h.scn_constraints, feats, roles=roles, type_to_candidates=resolver
            )
            domains = prefilter_domains(feats, domains, min_present_frames=10)

            overlap = None
            if require_pairs:
                actors = sorted({a for A in domains.values() for a in A})
                overlap = build_overlap_matrix(feats, actors, min_overlap_frames=10)

            for binding in enumerate_bindings(
                domains, distinct=True, overlap_ok=overlap,
                require_overlap_pairs=require_pairs
            ):
                hits = match_for_binding(
                    feats, call, binding,
                    fps=h.cfg.fps, max_results=max_results_per_binding,
                    cfg=h.cfg.to_query_cfg(),
                )
                if not hits:
                    continue

                wins = [(hi["t_start"], hi["t_end"]) for hi in hits]
                intervals = merge_windows_to_intervals(wins)
                T = int(getattr(feats, "T", 0) or 0)
                mask = intervals_to_mask(intervals, T) if T else None

                signal = PerCallSignal(
                    segment_id=seg_id,
                    roles=dict(binding),
                    T=T,
                    intervals=intervals,
                    mask=mask,
                    # NEW:
                    call_index=ci,
                    roles_used=tuple(roles),
                )
                store.add(call_key, signal)

    return store

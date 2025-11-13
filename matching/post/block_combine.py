# Step 3 — within-block persistence → BlockSignal
from __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Optional
from itertools import product
import numpy as np

from osc_parser.matching.results.types import PerCallSignal, BlockSignal, Interval
from osc_parser.matching.results.store import ResultStore  # your store type, provides .by_call
from osc_parser.matching.post.utils import (
    mask_from_intervals, intervals_from_mask,
    intersect_all, require_min_run, coalesce_intervals,
)


from .plan import BlockPlan

RolesKey = Tuple[Tuple[str, str], ...]  # sorted((role, actor_id), ...)

def _roles_key(d: Dict[str, str], required_roles: Optional[Iterable[str]] = None) -> RolesKey:
    if required_roles is None:
        items = sorted(d.items())
    else:
        items = sorted((r, d.get(r)) for r in required_roles)
    return tuple(items)

def _percall_to_mask(s: PerCallSignal) -> np.ndarray:
    return s.mask if s.mask is not None else mask_from_intervals(s.intervals, s.T)

def _union_masks(ms: List[np.ndarray]) -> Optional[np.ndarray]:
    if not ms:
        return None
    m = ms[0].copy()
    for k in range(1, len(ms)):
        m = np.logical_or(m, ms[k])
    return m

def _compatible_signal_with_binding(s: PerCallSignal, binding: Dict[str, str]) -> bool:
    # all roles present in the signal must match the binding
    for r, a in s.roles.items():
        if r not in binding or binding[r] != a:
            return False
    return True

def _collect_block_signals_for_seg(
    plan: BlockPlan,
    calls: List[dict],
    store: ResultStore,
    seg_id: str,
    T: int,
) -> Dict[Tuple[str, RolesKey], BlockSignal]:
    """
    Build BlockSignal per (seg, roles) for a single segment:
      - If any multi-role signals exist in the block, use them as seeds (bindings).
      - Otherwise, Cartesian product over unary role candidates.
      - For each candidate binding, require at least one compatible signal per call,
        union masks per call, AND across calls, enforce duration, emit if non-empty.
    """
    out: Dict[Tuple[str, RolesKey], BlockSignal] = {}

    # All calls in this block
    call_idxs = list(plan.indices)
    label = plan.label

    # Gather signals per call for this seg
    sigs_by_call: Dict[int, List[PerCallSignal]] = {}
    for ci in call_idxs:
        ck = (calls[ci].get("block_label") or label, ci)
        sigs = [s for s in store.by_call.get(ck, []) if s.segment_id == seg_id]
        sigs_by_call[ci] = sigs

    # Collect roles used across block
    block_roles: List[str] = []
    for ci in call_idxs:
        # prefer roles_used on signals; else fall back to roles_used_by_call you may have stuffed into call
        rs = set()
        for s in sigs_by_call[ci]:
            rs.update(s.roles_used or s.roles.keys())
        # if empty, fallback to call's declared roles_used (if you attached it) or actor
        if not rs:
            rs.add(calls[ci].get("actor"))
        for r in sorted(rs):
            if r not in block_roles:
                block_roles.append(r)

    # Seed candidate bindings
    candidate_bindings: List[Dict[str, str]] = []

    # (A) Multi-role seeds (e.g., calls that reference two roles)
    multi_role_seen = False
    seeds_set = set()
    for ci in call_idxs:
        for s in sigs_by_call[ci]:
            if len(s.roles) >= 2:
                multi_role_seen = True
                key = tuple(sorted(s.roles.items()))
                if key not in seeds_set:
                    seeds_set.add(key)
                    candidate_bindings.append(dict(s.roles))

    # (B) If no multi-role seeds, build product of unary role candidates
    if not candidate_bindings:
        # role -> set(actors) observed in unary signals across calls of this block
        role_to_actors: Dict[str, List[str]] = {}
        for r in block_roles:
            role_to_actors[r] = []
        for ci in call_idxs:
            for s in sigs_by_call[ci]:
                if len(s.roles) == 1:
                    (r, a), = s.roles.items()
                    if r in role_to_actors and a not in role_to_actors[r]:
                        role_to_actors[r].append(a)

        # if any role has no candidates, no combinations
        if all(role_to_actors[r] for r in block_roles):
            for combo in product(*[role_to_actors[r] for r in block_roles]):
                cand = {r: a for r, a in zip(block_roles, combo)}
                candidate_bindings.append(cand)

    if not candidate_bindings:
        return out

    # For each candidate binding, check every call has a compatible signal; combine masks
    need_min_run = int(plan.duration_frames or 0)

    for binding in candidate_bindings:
        per_call_masks: List[np.ndarray] = []
        ok_binding = True

        for ci in call_idxs:
            sigs = sigs_by_call[ci]
            # pick signals for this call compatible with binding; if call uses a subset of roles, that's fine
            comp_masks = []
            for s in sigs:
                if _compatible_signal_with_binding(s, binding):
                    comp_masks.append(_percall_to_mask(s))
            if not comp_masks:
                ok_binding = False
                break
            m_union = _union_masks(comp_masks)
            if m_union is None:
                ok_binding = False
                break
            per_call_masks.append(m_union)

        if not ok_binding:
            continue

        m = intersect_all(per_call_masks)
        if need_min_run:
            m = require_min_run(m, need_min_run)
        ivs = coalesce_intervals(intervals_from_mask(m))
        if not ivs:
            continue

        rk = _roles_key(binding, required_roles=binding.keys())
        out[(seg_id, rk)] = BlockSignal(
            segment_id=seg_id,
            roles=dict(binding),
            T=T,
            intervals=ivs,
            mask=m,
        )

    return out

def combine_parallel_block(
    plan: BlockPlan,
    calls: List[dict],
    store: ResultStore,
    T_by_seg: Dict[str, int],
) -> Dict[Tuple[str, RolesKey], BlockSignal]:
    out: Dict[Tuple[str, RolesKey], BlockSignal] = {}
    for seg_id, T in T_by_seg.items():
        local = _collect_block_signals_for_seg(plan, calls, store, seg_id, T)
        out.update(local)
    return out

def chain_serial_block(
    plan: BlockPlan,
    calls: List[dict],
    store: ResultStore,
    T_by_seg: Dict[str, int],
    allow_overlap: bool = True,
) -> Dict[Tuple[str, RolesKey], BlockSignal]:
    """
    Serial semantics on top of candidate role bindings:
      - enumerate candidate bindings exactly as in parallel,
      - for each call pull intervals compatible with the binding,
      - chain intervals in order (iv0 then iv1 with start >= end of iv0 [+1 if !allow_overlap]),
      - emit unioned chained intervals.
    """
    out: Dict[Tuple[str, RolesKey], BlockSignal] = {}

    def _meet(a: Interval, b: Interval) -> bool:
        return (b.t0 >= a.t1) if allow_overlap else (b.t0 >= a.t1 + 1)

    for seg_id, T in T_by_seg.items():
        # reuse the parallel enumerator to get candidate bindings and masks per call
        # but we need the raw intervals per call for chaining, so prepare sigs_by_call here too
        call_idxs = list(plan.indices)
        label = plan.label
        sigs_by_call: Dict[int, List[PerCallSignal]] = {}
        for ci in call_idxs:
            ck = (calls[ci].get("block_label") or label, ci)
            sigs_by_call[ci] = [s for s in store.by_call.get(ck, []) if s.segment_id == seg_id]

        # build candidate bindings the same way as in _collect_block_signals_for_seg
        candidate_bindings: List[Dict[str, str]] = []
        seeds_set = set()
        block_roles: List[str] = []
        for ci in call_idxs:
            rs = set()
            for s in sigs_by_call[ci]:
                rs.update(s.roles_used or s.roles.keys())
            if not rs:
                rs.add(calls[ci].get("actor"))
            for r in sorted(rs):
                if r not in block_roles:
                    block_roles.append(r)
        # multi-role seeds
        for ci in call_idxs:
            for s in sigs_by_call[ci]:
                if len(s.roles) >= 2:
                    key = tuple(sorted(s.roles.items()))
                    if key not in seeds_set:
                        seeds_set.add(key)
                        candidate_bindings.append(dict(s.roles))
        # unary product if needed
        if not candidate_bindings:
            role_to_actors: Dict[str, List[str]] = {r: [] for r in block_roles}
            for ci in call_idxs:
                for s in sigs_by_call[ci]:
                    if len(s.roles) == 1:
                        (r, a), = s.roles.items()
                        if r in role_to_actors and a not in role_to_actors[r]:
                            role_to_actors[r].append(a)
            if all(role_to_actors[r] for r in block_roles):
                for combo in product(*[role_to_actors[r] for r in block_roles]):
                    candidate_bindings.append({r: a for r, a in zip(block_roles, combo)})
        if not candidate_bindings:
            continue

        need_min_run = int(plan.duration_frames or 0)

        for binding in candidate_bindings:
            # collect intervals for each call compatible with this binding
            per_call_iv_lists: List[List[Interval]] = []
            ok_binding = True
            for ci in call_idxs:
                cur_ivs: List[Interval] = []
                for s in sigs_by_call[ci]:
                    if _compatible_signal_with_binding(s, binding):
                        ivs = s.intervals if s.mask is None else intervals_from_mask(s.mask)
                        cur_ivs.extend(ivs)
                cur_ivs.sort(key=lambda iv: (iv.t0, iv.t1))
                if not cur_ivs:
                    ok_binding = False
                    break
                per_call_iv_lists.append(cur_ivs)
            if not ok_binding:
                continue

            # chain intervals in order across calls
            chains: List[Interval] = per_call_iv_lists[0][:]
            for ivs in per_call_iv_lists[1:]:
                new_chains: List[Interval] = []
                j = 0
                for a in chains:
                    while j < len(ivs) and ivs[j].t1 < a.t0:
                        j += 1
                    k = j
                    while k < len(ivs):
                        b = ivs[k]
                        if _meet(a, b):
                            new_chains.append(Interval(a.t0, b.t1))
                        k += 1
                chains = coalesce_intervals(new_chains)
                if not chains:
                    break
            if not chains:
                continue

            m = mask_from_intervals(chains, T)
            if need_min_run:
                m = require_min_run(m, need_min_run)
            ivs_final = coalesce_intervals(intervals_from_mask(m))
            if not ivs_final:
                continue

            rk = _roles_key(binding, required_roles=binding.keys())
            out[(seg_id, rk)] = BlockSignal(
                segment_id=seg_id,
                roles=dict(binding),
                T=T,
                intervals=ivs_final,
                mask=m,
            )

    return out

# role_planning.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Generator, Tuple, Callable, Set, Union
from copy import deepcopy
import numpy as np
from osc_parser.matching.spec import build_block_query
from osc_parser.matching.match_block import match_block

__all__ = [
    "make_type_to_candidates",
    "role_domains_from_segment",
    "prefilter_domains",
    "build_overlap_matrix",
    "enumerate_bindings",
    "remap_call",
    "roles_used_by_call",   # optional export
    "match_for_binding",    # optional export
]

# --------------------------------------------------------------------------------------
# Candidate resolution
# --------------------------------------------------------------------------------------

def match_for_binding(feats, call, binding, fps, max_results, cfg):
    """
    Remap role names -> concrete actor ids using the binding, then compile and match.
    This keeps 'ego' as the actor in the call and converts any other role names
    (whatever they are in the scenario) to their bound actor ids.
    """
    # Replace role strings in the call by concrete actor ids
    call_bound = remap_call(call, binding)

    # Build query with concrete ids (avoids having role names in Q.npc_candidates)
    Q, pairs_hint = build_block_query(call_bound, fps=fps, cfg=cfg)

    # Run the matcher; pass along any candidate pair hints if present
    return match_block(
        feats=feats,
        Q=Q,
        fps=fps,
        pairs=pairs_hint,
        max_results=max_results,
    )
                       
def make_type_to_candidates(feats) -> Callable[[str], List[str]]:
    """
    Build a function mapping a declared OSC actor *type name* (e.g. "vehicle")
    -> list of actor_ids in this segment with that type and nonzero presence.

    Prefers `feats.actor_types` if available. Falls back to id prefixes.
    Only returns actors that were present in >=1 frame in this segment.
    """
    type_map: Optional[Dict[str, str]] = getattr(feats, "actor_types", None)

    # actors seen at least once in this segment
    present_ok: Set[str] = {
        aid for aid, pres in (getattr(feats, "present", {}) or {}).items()
        if getattr(pres, "sum", lambda: sum(1 for x in pres if x > 0.5))() > 0
    }

    def infer_from_id(aid: str) -> str:
        if aid.startswith("vehicle_"):    return "vehicle"
        if aid.startswith("cyclist_"):    return "cyclist"
        if aid.startswith("pedestrian_"): return "pedestrian"
        return "unknown"

    def resolver(type_name: str) -> List[str]:
        t = str(type_name).lower()
        if type_map:
            return [aid for aid, typ in type_map.items()
                    if str(typ).lower() == t and aid in present_ok]
        # fallback by naming convention
        return [aid for aid in present_ok if infer_from_id(aid) == t]

    return resolver

# --------------------------------------------------------------------------------------
# Role discovery helpers
# --------------------------------------------------------------------------------------

_REF_KEYS = ("reference", "same_as", "ahead_of", "behind", "side_of", "right_of", "left_of")

def roles_used_by_call(call) -> Set[str]:
    """
    Inspect the call and collect *role names* it references.
    Always includes call['actor']. Adds referenced roles from action args and modifiers.
    """
    roles = {call["actor"]}  # actor is a role name in the IR (e.g., 'ego_vehicle')
    aargs = call.get("action_args") or {}

    # action-level references
    ref = aargs.get("reference")
    if isinstance(ref, str):
        roles.add(ref)

    # modifier-level references (common reference keys)
    for m in call.get("modifiers") or []:
        args = m.get("args") or {}
        for key in ("same_as", "ahead_of", "behind", "side_of", "reference"):
            v = args.get(key)
            if isinstance(v, str):
                roles.add(v)

    return roles

def _declared_role_type(scn_or_constraints: Any, role_name: str) -> str:
    """
    Return the declared type for a role (e.g., 'vehicle', 'pedestrian').
    Handles either:
      - ScenarioNode-like object with .actors mapping to ActorInst (with .type)
      - dict constraints with ["actors"][role]["type"]
    """
    # Object with .actors?
    actors_obj = getattr(scn_or_constraints, "actors", None)
    if isinstance(actors_obj, dict):
        inst = actors_obj.get(role_name)
        if inst is not None:
            # ActorInst or dict-like
            if hasattr(inst, "type"):
                return str(inst.type).lower()
            if isinstance(inst, dict) and "type" in inst:
                return str(inst["type"]).lower()
        return ""

    # Dict-style constraints?
    if isinstance(scn_or_constraints, dict):
        actors = scn_or_constraints.get("actors", {}) or {}
        info = actors.get(role_name, {}) or {}
        t = info.get("type") or info.get("kind") or ""
        return str(t).lower()

    return ""

def _present_actor_ids(feats) -> List[str]:
    """All actor ids that are present at least once in this segment."""
    pres = (getattr(feats, "present", {}) or {})
    out = []
    for aid, p in pres.items():
        n_present = int(np.sum(np.asarray(p, dtype=float) > 0.5))
        if n_present > 0:
            out.append(aid)
    return out

def _all_roles_from_scn(scn_or_constraints: Any) -> List[str]:
    """Return role names from either ScenarioNode (.actors) or dict constraints ['actors']."""
    actors_obj = getattr(scn_or_constraints, "actors", None)
    if isinstance(actors_obj, dict):
        return list(actors_obj.keys())
    if isinstance(scn_or_constraints, dict):
        return list((scn_or_constraints.get("actors", {}) or {}).keys())
    return []

# --------------------------------------------------------------------------------------
# Domain construction (robust to ScenarioNode or dict)
# --------------------------------------------------------------------------------------

def role_domains_from_segment(
    scn_constraints: Union[Dict[str, Any], Any],
    feats,
    roles: Optional[List[str]] = None,
    *,
    type_to_candidates: Optional[Callable[[str], List[str]] | Dict[str, List[str]]] = None,
) -> Dict[str, List[str]]:
    """
    Build domain of concrete actor IDs per role.

    - If roles is None, uses all roles declared in the scenario (ScenarioNode or dict).
    - For each role, looks up its declared type (e.g., 'vehicle').
    - Chooses candidates using:
        • provided `type_to_candidates` (callable or dict), if given; else
        • the default resolver from `make_type_to_candidates(feats)`.
    - Falls back to "all present actors" if role type is missing or resolver returns empty.
    """
    if roles is None:
        roles = _all_roles_from_scn(scn_constraints)

    # Build resolver
    if type_to_candidates is None:
        resolver: Optional[Callable[[str], List[str]]] = make_type_to_candidates(feats)
    elif callable(type_to_candidates):
        resolver = type_to_candidates
    else:
        mapping = {str(k).lower(): list(v) for k, v in (type_to_candidates or {}).items()}
        resolver = lambda t: list(mapping.get(str(t).lower(), []))

    all_present = _present_actor_ids(feats)
    out: Dict[str, List[str]] = {}

    for r in roles:
        want_type = _declared_role_type(scn_constraints, r)
        if want_type and resolver:
            cands = list(resolver(want_type))
            out[r] = cands if cands else list(all_present)
        else:
            # no declared type → allow any present actor
            out[r] = list(all_present)

    return out

# --------------------------------------------------------------------------------------
# Domain shrinkers and overlap
# --------------------------------------------------------------------------------------

def prefilter_domains(
    feats,
    domains: Dict[str, List[str]],
    *,
    min_present_frames: int = 1,
    require_speed: bool = True,
) -> Dict[str, List[str]]:
    """Cheap 1-actor filters to shrink domains (presence length, has speed array)."""
    pres = getattr(feats, "present", {}) or {}
    spd = getattr(feats, "speed", {}) or {}

    def ok(a: str) -> bool:
        p = pres.get(a)
        if p is None:
            return False
        n_present = int(np.sum(np.asarray(p, dtype=float) > 0.5))
        if n_present < min_present_frames:
            return False
        if require_speed and (a not in spd or spd[a] is None):
            return False
        return True

    return {r: [a for a in A if ok(a)] for r, A in domains.items()}

def build_overlap_matrix(
    feats,
    actors: List[str],
    *,
    min_overlap_frames: int = 1,
) -> Dict[Tuple[str, str], bool]:
    """Precompute (actor_i, actor_j) time overlap via presence arrays."""
    pres = getattr(feats, "present", {}) or {}
    ok: Dict[Tuple[str, str], bool] = {}
    for a in actors:
        pa = np.asarray(pres.get(a, []), dtype=float) > 0.5
        for b in actors:
            if a == b:
                ok[(a, b)] = False
                continue
            pb = np.asarray(pres.get(b, []), dtype=float) > 0.5
            if pa.size and pb.size:
                overlap = int(np.sum(pa & pb)) >= int(min_overlap_frames)
                ok[(a, b)] = overlap
            else:
                ok[(a, b)] = False
    return ok

def enumerate_bindings(
    domains: Dict[str, List[str]],
    *,
    distinct: bool = True,
    overlap_ok: Optional[Dict[Tuple[str, str], bool]] = None,
    require_overlap_pairs: Optional[List[Tuple[str, str]]] = None,  # role pairs that must co-exist in time
    limit: Optional[int] = None,
) -> Generator[Dict[str, str], None, None]:
    """Backtracking with forward-checking. MRV order; prunes by distinct and pairwise overlap."""
    roles = sorted(domains, key=lambda r: len(domains[r]))
    used: set[str] = set()
    bind: Dict[str, str] = {}
    out_count = 0

    need_overlap = set(tuple(p) for p in (require_overlap_pairs or []))

    def viable_with_current(r_new: str, a_new: str) -> bool:
        # distinctness
        if distinct and a_new in used:
            return False
        # pairwise overlap constraints (only check against roles already bound)
        if overlap_ok and need_overlap:
            for (r1, r2) in need_overlap:
                if r_new == r1 and r2 in bind:
                    if not overlap_ok.get((a_new, bind[r2]), False):
                        return False
                if r_new == r2 and r1 in bind:
                    if not overlap_ok.get((bind[r1], a_new), False):
                        return False
        return True

    def dfs(i: int):
        nonlocal out_count
        if limit is not None and out_count >= limit:
            return
        if i == len(roles):
            out_count += 1
            yield dict(bind)
            return
        r = roles[i]
        for a in domains[r]:
            if not viable_with_current(r, a):
                continue
            bind[r] = a
            used.add(a)
            yield from dfs(i + 1)
            used.remove(a)
            bind.pop(r, None)

    yield from dfs(0)

# --------------------------------------------------------------------------------------
# Convenience flows (optional)
# --------------------------------------------------------------------------------------

def derive_domains_for_segment(scn, feats, call, *, min_present_frames: int = 10):
    """
    One-stop helper:
      1) infer roles the call actually needs
      2) build domains for those roles
      3) prefilter by presence
    """
    roles = list(roles_used_by_call(call))
    domains = role_domains_from_segment(scn, feats, roles=roles)
    domains = prefilter_domains(feats, domains, min_present_frames=min_present_frames)
    return domains

def enumerate_bindings_for_call(scn, feats, call, domains, *, min_overlap_frames=10):
    """
    Pick overlap gating only if we truly have multi-role interactions.
    """
    roles = list(domains.keys())
    require_pairs = []
    if "ego_vehicle" in roles and "npc" in roles:
        require_pairs = [("ego_vehicle", "npc")]

    actors = sorted({a for A in domains.values() for a in A})
    overlap = build_overlap_matrix(feats, actors, min_overlap_frames=min_overlap_frames)

    return enumerate_bindings(
        domains,
        distinct=True,
        overlap_ok=overlap if require_pairs else None,
        require_overlap_pairs=require_pairs,
    )

# --------------------------------------------------------------------------------------
# Call remapping
# --------------------------------------------------------------------------------------

def roles_in_program(blocks: List[Dict[str, Any]]) -> List[str]:
    """Union of roles referenced across blocks/calls."""
    roles = set()
    def scan_call(c):
        roles.add(c["actor"])
        aa = c.get("action_args") or {}
        for k in _REF_KEYS:
            v = aa.get(k)
            if isinstance(v, str):
                roles.add(v)
        for m in c.get("modifiers", []):
            args = m.get("args", {}) or {}
            for k in _REF_KEYS:
                v = args.get(k)
                if isinstance(v, str):
                    roles.add(v)
    def walk(b):
        for c in b.get("calls", []): scan_call(c)
        for ch in b.get("children", []): walk(ch)
    for b in blocks: walk(b)
    return sorted(roles)

def remap_call(call: Dict[str, Any], binding: Dict[str, str]) -> Dict[str, Any]:
    """Replace role strings in a call with concrete actor ids per binding."""
    c = deepcopy(call)
    if "actor" in c:
        c["actor"] = binding.get(c["actor"], c["actor"])
    aa = c.get("action_args") or {}
    for k in _REF_KEYS:
        if k in aa and isinstance(aa[k], str):
            aa[k] = binding.get(aa[k], aa[k])
    for m in c.get("modifiers", []):
        args = m.get("args", {}) or {}
        for k in _REF_KEYS:
            if k in args and isinstance(args[k], str):
                args[k] = binding.get(args[k], args[k])
    return c

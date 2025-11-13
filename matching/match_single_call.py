# osc_parser/matching/match_single_call.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import copy

from .spec import build_block_query, BlockQuery
from .match_block import match_block
from .features import TagFeatures


def _resolve_roles_in_obj(obj: Any, binding: Dict[str, str]) -> Any:
    """
    Recursively replace any string equal to a role name with its bound actor_id.
    This is safe for your flattened calls because role names (e.g., 'ego_vehicle',
    'second_car', 'npc') only appear in the actor/reference fields and not in units.
    """
    if isinstance(obj, str):
        return binding.get(obj, obj)
    if isinstance(obj, list):
        return [_resolve_roles_in_obj(x, binding) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_resolve_roles_in_obj(x, binding) for x in obj)
    if isinstance(obj, dict):
        # keep keys, resolve values only
        return {k: _resolve_roles_in_obj(v, binding) for k, v in obj.items()}
    return obj


def _call_with_binding(call: Dict[str, Any], binding: Dict[str, str]) -> Dict[str, Any]:
    """
    Produce a copy of `call` where all role strings are replaced by concrete actor ids.
    Also maps call['actor'] (which is a role) to the bound actor id.
    """
    c = copy.deepcopy(call)
    # First resolve all values
    c = _resolve_roles_in_obj(c, binding)
    # Ensure the top-level 'actor' is the concrete actor id
    actor_role = call.get("actor")
    if isinstance(actor_role, str) and actor_role in binding:
        c["actor"] = binding[actor_role]
    return c


def match_for_binding(
    feats: TagFeatures,
    call: Dict[str, Any],
    binding: Dict[str, str],
    *,
    fps: int,
    max_results: int = 2000,
    cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Run the matcher for a SINGLE flattened call on ONE segment with a specific role->actor binding.

    Returns a list of hits: {"ego","npc","t_start","t_end"} where npc is always None
    (we run unary), because all pair references are already closed over inside the checks
    by resolving roles to concrete actor ids before building the query.
    """
    # Resolve role names inside the call into concrete actor ids
    call_resolved = _call_with_binding(call, binding)

    # Build the query. Your build_block_query already:
    # - normalizes units
    # - constructs checks that capture referenced actors by id
    Q, _pairs_hint_unused = build_block_query(call_resolved, fps=fps, cfg=(cfg or {}))

    # We run UNARY: checks that need "other" already close over concrete ids.
    # Force arity=1 to avoid the binary path in match_block.
    Q.arity = 1

    # Ego id is now a concrete actor id (after _call_with_binding)
    ego_id = call_resolved.get("actor")
    if not isinstance(ego_id, str):
        return []

    # Optional: don’t clobber explicit settings that build_block_query already set
    if cfg:
        Q.cfg = {**Q.cfg, **cfg}

    # Execute: unary with explicit id list
    hits = match_block(
        feats,
        Q,
        fps=fps,
        ids=[ego_id],     # unary path
        pairs=None,       # not used
        max_results=max_results,
    )
    return hits

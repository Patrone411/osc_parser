# matching/driver.py (drop-in, minimal but working skeleton)

from typing import Dict, Any, List, Tuple, Optional
from .features import segment_ids, segment_num_lanes, segment_length, actor_ids_in_segment
from .eval_call import match_call
from .helpers import frames_from_seconds, seconds_from_physical

def _block_duration_frames(block: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[int]:
    d = block.get("duration")
    if not d:
        return None
    sec = seconds_from_physical(d)
    return frames_from_seconds(sec, cfg)

def _call_duration_frames(call: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[int]:
    d = call.get("action_args", {}).get("duration")
    if not d:
        return None
    sec = seconds_from_physical(d)
    return frames_from_seconds(sec, cfg)

# --- role binding ---

def _roles_in_scenario(norm: Dict[str, Any]) -> Dict[str, str]:
    """Return {"ego_vehicle": "vehicle", "npc": "person", ...} from norm['actors']."""
    return {name: info.get("type") for name, info in norm.get("actors", {}).items()}

def _candidate_actors_for_type(stitched: Dict[str, Any], seg_id: str, role_type: str) -> List[str]:
    """
    Simple default: return all actor ids in the segment.
    If you have type tags, filter by role_type here.
    """
    ids = actor_ids_in_segment(stitched, seg_id)
    # TODO: filter by type if your data encodes it (e.g., prefix "vehicle_", "ped_", etc.)
    return ids

def _generate_bindings(stitched: Dict[str, Any], seg_id: str, roles: Dict[str,str], limit: int = 2000):
    """
    Naive Cartesian product of role→candidates, filtered for distinct ids.
    Add heuristics or pruning if needed.
    """
    from itertools import product
    keys = list(roles.keys())
    cands = [ _candidate_actors_for_type(stitched, seg_id, roles[k]) for k in keys ]
    count = 0
    for combo in product(*cands):
        if len(set(combo)) != len(combo):   # no duplicate actor for different roles
            continue
        yield dict(zip(keys, combo))
        count += 1
        if count >= limit:
            break

# --- core recursion ---

def _match_parallel(stitched, seg_id, roles, block, cfg) -> Optional[Tuple[int,int,List[Dict[str,Any]]]]:
    dur = _block_duration_frames(block, cfg)
    T = segment_length(stitched, seg_id)
    if dur is None:
        # no explicit block duration: take max of child call durations (or cfg default)
        child_durs = [_call_duration_frames(c, cfg) for c in block.get("calls",[])]
        child_durs = [d for d in child_durs if d is not None]
        dur = max(child_durs) if child_durs else int(cfg.get("default_block_frames", 1))

    for t0 in range(0, max(0, T - dur + 1)):
        t1 = t0 + dur - 1
        # all calls must pass over [t0..t1]
        ok_calls = all(
            match_call(stitched, seg_id, roles, t0, t1, call, cfg)
            for call in block.get("calls", [])
        )
        if not ok_calls:
            continue
        # all children must also match inside [t0..t1]
        child_windows = []
        for child in block.get("children", []):
            w = _match_block_in_window(stitched, seg_id, roles, child, t0, t1, cfg)
            if w is None:
                break
            child_windows.append(w)
        else:
            # success
            return (t0, t1, child_windows)
    return None

def _match_serial(stitched, seg_id, roles, block, cfg) -> Optional[List[Tuple[int,int,Any]]]:
    """
    Chain children/calls one after the other. Each element can be a call or a nested block.
    Returns a list of child windows.
    """
    T = segment_length(stitched, seg_id)
    chain = []
    cursor = 0
    gap = int(cfg.get("serial_max_gap_frames", 0))

    # Flatten items: treat calls as small “blocks”
    items: List[Dict[str,Any]] = []
    items.extend([{"__kind__":"call", "call": c} for c in block.get("calls", [])])
    items.extend([{"__kind__":"block","block": b} for b in block.get("children", [])])

    for it in items:
        found = None
        start_from = cursor
        while start_from < T:
            if it["__kind__"] == "call":
                # choose duration for call window
                dur = _call_duration_frames(it["call"], cfg) or int(cfg.get("default_call_frames", 1))
                for t0 in range(start_from, min(T, start_from + gap + 1)):
                    t1 = min(T-1, t0 + dur - 1)
                    if match_call(stitched, seg_id, roles, t0, t1, it["call"], cfg):
                        found = (t0, t1, None)
                        break
                if found: break
            else:
                # nested block
                for t0 in range(start_from, min(T, start_from + gap + 1)):
                    # let the nested block choose its own t1
                    win = _match_block_from(stitched, seg_id, roles, it["block"], t0, cfg)
                    if win is not None:
                        found = win  # could be (t0,t1,[childs]) or a chain
                        break
                if found: break

            start_from += 1

        if found is None:
            return None
        chain.append(found)
        cursor = found[1] + 1  # next must start after this
    return chain

def _match_one_of(stitched, seg_id, roles, block, cfg):
    for child in block.get("children", []):
        w = _match_block_from(stitched, seg_id, roles, child, 0, cfg)
        if w is not None:
            return w
    return None

def _match_block_in_window(stitched, seg_id, roles, block, lo, hi, cfg):
    """
    Constrain child matching to stay inside [lo..hi].
    For serial children you can ensure their windows don’t cross.
    A simple approach: run child matcher and check it returns within [lo..hi].
    """
    res = _match_block_from(stitched, seg_id, roles, block, lo, cfg)
    if res is None: return None
    t0, t1, *_ = res
    if t0 < lo or t1 > hi:
        return None
    return res

def _match_block_from(stitched, seg_id, roles, block, start_at, cfg):
    btype = block["type"]
    if btype == "parallel":
        return _match_parallel(stitched, seg_id, roles, block, cfg)
    if btype == "serial":
        chain = _match_serial(stitched, seg_id, roles, block, cfg)
        if chain is None: return None
        # summarize chain window
        return (chain[0][0], chain[-1][1], chain)
    if btype == "one_of":
        return _match_one_of(stitched, seg_id, roles, block, cfg)
    # fallback
    return None

# --- public entry point ---

def match_scenario_recursive(stitched: Dict[str,Any], norm: Dict[str,Any], cfg: Dict[str,Any]):
    """
    Returns a list of matches:
      [{ "segment": seg_id, "roles": {role->actor_id}, "window": (t0,t1), "tree": <optional details> }, ...]
    """
    results = []
    roles_spec = _roles_in_scenario(norm)
    for seg_id in segment_ids(stitched):
        # optional map constraint: min_lanes
        want = cfg.get("min_lanes")
        if want is not None and segment_num_lanes(stitched, seg_id) != int(want):
            continue

        for roles in _generate_bindings(stitched, seg_id, roles_spec):
            for block in norm["blocks"]:
                win = _match_block_from(stitched, seg_id, roles, block, 0, cfg)
                if win is not None:
                    t0, t1, tree = win
                    results.append({
                        "segment": seg_id,
                        "roles": roles,
                        "window": (t0, t1),
                        "tree": tree,            # you can keep per-child windows here
                        "block_label": block.get("label"),
                        "block_type": block.get("type"),
                    })
    return results

# osc_parser/matching/predicates.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, List, Optional, Callable
import math
import numpy as np

from .features import TagFeatures, LATERAL_LEFT, LATERAL_RIGHT, LATERAL_SAME, LATERAL_UNKNOWN

# ------------------------ helpers: units ------------------------

def _speed_to_mps(spec: Dict[str, Any]) -> Tuple[float, float]:
    """
    spec = {"value": x, "unit": "..."} or {"range":[lo,hi], "unit":"..."}
    Returns (lo, hi) in m/s.
    """
    if "value" in spec:
        lo = hi = float(spec["value"])
    elif "range" in spec:
        lo, hi = map(float, spec["range"])
    else:
        raise ValueError("speed spec missing 'value' or 'range'")

    unit = (spec.get("unit") or "").lower()
    if unit in ("m/s", "meter_per_second", "meter per second", "meter_per_second"):
        f = 1.0
    elif unit in ("km/h", "kph", "kilometer_per_hour", "kilometre_per_hour"):
        f = 1000.0 / 3600.0
    elif unit in ("mph", "mile_per_hour"):
        f = 1609.344 / 3600.0
    else:
        # best-effort: assume already m/s
        f = 1.0
    return lo * f, hi * f

def _length_to_m(spec: Dict[str, Any]) -> Tuple[float, float]:
    """
    spec = {"value": x, "unit":"..."} or {"range":[lo,hi], "unit":"..."}
    Returns (lo, hi) in meters (scalar -> (v,v)).
    """
    if "value" in spec:
        lo = hi = float(spec["value"])
    elif "range" in spec:
        lo, hi = map(float, spec["range"])
    else:
        raise ValueError("length spec missing 'value' or 'range'")

    unit = (spec.get("unit") or "").lower()
    if unit in ("m", "meter", "metre"):
        f = 1.0
    elif unit in ("km", "kilometer", "kilometre"):
        f = 1000.0
    elif unit in ("cm", "centimeter", "centimetre"):
        f = 0.01
    elif unit in ("mm", "millimeter", "millimetre"):
        f = 0.001
    elif unit in ("ft", "feet"):
        f = 0.3048
    elif unit in ("in", "inch"):
        f = 0.0254
    elif unit in ("mi", "mile"):
        f = 1609.344
    else:
        f = 1.0
    return lo * f, hi * f

# ------------------------ BlockQuery ------------------------

CheckFn = Callable[[TagFeatures, str, str, int, int, Dict[str, Any]], bool]

@dataclass
class BlockQuery:
    ego: str
    npc_candidates: Optional[List[str]]            # None → any NPC
    duration_frames: int
    allow_shorter_end: bool = False
    cfg: Dict[str, Any] = field(default_factory=dict)
    checks: List[CheckFn] = field(default_factory=list)

# ------------------------ primitive checks ------------------------

def _present_enough(feats: TagFeatures, actor: str, t0: int, t1: int, min_cov: float) -> bool:
    p = feats.present.get(actor)
    if p is None or t1 >= len(p): return False
    win = p[t0:t1+1]
    return float(np.nanmean(win)) >= float(min_cov)

def _speed_in_range(feats: TagFeatures, actor: str, t0: int, t1: int,
                    lo: float, hi: float, coverage: float, tol: float) -> bool:
    v = feats.speed.get(actor)
    if v is None or t1 >= len(v): return False
    s = v[t0:t1+1]
    ok = (s >= (lo - tol)) & (s <= (hi + tol))
    ok[np.isnan(s)] = False
    return (ok.sum() / max(1, len(ok))) >= coverage

def _lane_at(feats: TagFeatures, actor: str, t: int) -> Optional[int]:
    arr = feats.lane_idx.get(actor)
    if arr is None or t >= len(arr): return None
    v = int(arr[t])
    return v if v > 0 else None

def _rel_lateral(feats: TagFeatures, ego: str, npc: str, t: int) -> str:
    arr = feats.lat_rel.get((ego, npc))
    if arr is None or t >= len(arr): return LATERAL_UNKNOWN
    return arr[t] or LATERAL_UNKNOWN

# ------------------------ compiled checks ------------------------

def _check_presence(feats: TagFeatures, ego: str, npc: str, t0: int, t1: int, cfg: Dict[str, Any]) -> bool:
    cov = float(cfg.get("presence_min_coverage", 0.9))
    return _present_enough(feats, ego, t0, t1, cov) and _present_enough(feats, npc, t0, t1, cov)

def _check_speed(feats: TagFeatures, ego: str, npc: str, t0: int, t1: int, cfg: Dict[str, Any]) -> bool:
    lo, hi = cfg["_speed_lo_hi_mps"]
    cov    = float(cfg.get("speed_min_coverage", 0.9))
    tol    = float(cfg.get("speed_value_tol", 0.0))
    return _speed_in_range(feats, ego, t0, t1, lo, hi, cov, tol)

def _check_lane_at_start(feats: TagFeatures, ego: str, npc: str, t0: int, t1: int, cfg: Dict[str, Any]) -> bool:
    want = int(cfg["_lane_at_start"])
    l0 = _lane_at(feats, ego, t0)
    return l0 is not None and l0 == want

def _check_ahead_or_behind(feats: TagFeatures, ego: str, npc: str, t0: int, t1: int, cfg: Dict[str, Any]) -> bool:
    when = cfg.get("_when", "start")
    t    = t0 if when == "start" else t1
    need = cfg["_longitudinal"]  # "ahead_of" or "behind"
    # Prefer relations.position if you add a decoder; for now use distance sign if present,
    # else pass if present and laterals not contradicting basic layout.
    # Minimal viable: require non-missing presence and let distance (if provided) match.
    # If you have tagged 'position' codes, plug them here.

    # Optional distance check
    if "_dist_lo_hi_m" in cfg:
        lo, hi = cfg["_dist_lo_hi_m"]
        arr = feats.rel_distance.get((ego, npc))
        if arr is None or t >= len(arr): return False
        d = arr[t]
        if math.isnan(d): return False
        if not (lo <= float(d) <= hi): return False

    # Without robust position codes, accept as “unknown-OK”.
    # You can tighten this later by decoding feats.rel_position -> {front,back,...}.
    return True

def _check_lateral_side_of(feats: TagFeatures, ego: str, npc: str, t0: int, t1: int, cfg: Dict[str, Any]) -> bool:
    when = cfg.get("_when", "start")
    t    = t0 if when == "start" else t1
    side = cfg["_side"]  # "left" | "right" | "same"
    rel  = _rel_lateral(feats, ego, npc, t)
    if side == "left":  return rel == LATERAL_LEFT
    if side == "right": return rel == LATERAL_RIGHT
    if side == "same":  return rel == LATERAL_SAME
    return False

def _check_change_speed_delta(feats: TagFeatures, ego: str, npc: str, t0: int, t1: int, cfg: Dict[str, Any]) -> bool:
    v = feats.speed.get(ego)
    if v is None or t1 >= len(v): return False
    dv = float(v[t1]) - float(v[t0])
    target = float(cfg["_dv_target_mps"])
    tol    = float(cfg.get("change_speed_tol", 0.3))
    return abs(dv - target) <= tol

# ------------------------ compiler ------------------------

def build_block_query(call_entry: Dict[str, Any], fps: int = 10, cfg: Optional[Dict[str, Any]] = None) -> Tuple[BlockQuery, Optional[List[Tuple[str, str]]]]:
    """
    Compile ONE normalized call from constraints_from_ir() into a BlockQuery.
    Returns (BlockQuery, suggested_pairs) where suggested_pairs can hint (ego, npc)
    pairings if a modifier references a concrete npc.
    Expects shape like:
      call_entry = {
        "actor": "ego_vehicle",
        "action": "drive",
        "action_args": {"duration": {"value": 12, "unit": "second"}},
        "block_duration": {"value": 15, "unit": "second"},
        "modifiers": [{"name":"speed","args":{"speed":{"range":[40,60],"unit":"kilometer_per_hour"}}}, ...]
      }
    """
    cfg = dict(cfg or {})
    ego = call_entry["actor"]

    # duration: prefer action_args.duration, else use block_duration; default 1s
    dur_s = 1.0
    act_dur = call_entry.get("action_args", {}).get("duration")
    if act_dur:
        dur_s = float(act_dur.get("value", 1.0))
    elif call_entry.get("block_duration"):
        dur_s = float(call_entry["block_duration"].get("value", 1.0))
    duration_frames = max(1, int(round(dur_s * fps)))

    allow_shorter = bool(cfg.get("allow_shorter_end", True))

    checks: List[CheckFn] = []
    npc_hints: List[Tuple[str, str]] = []

    # Always require presence coverage for both ego and referenced npc
    checks.append(_check_presence)

    for mod in call_entry.get("modifiers", []):
        name = mod.get("name")
        args = mod.get("args", {})

        # --- speed(explicit) ---
        if name == "speed" and "speed" in args:
            lo, hi = _speed_to_mps(args["speed"])
            checks.append(_check_speed)
            cfg["_speed_lo_hi_mps"] = (lo, hi)

        # --- change_speed(speed or range) → use delta V over window ---
        if name == "change_speed" and "speed" in args:
            lo, hi = _speed_to_mps(args["speed"])
            # If a range is supplied, match "any delta within" by loosening target:
            # simplest: target midpoint; adjust tolerance by half-span
            target = 0.5 * (lo + hi)
            extra_tol = 0.5 * abs(hi - lo)
            cfg.setdefault("change_speed_tol", 0.3)
            cfg["change_speed_tol"] += extra_tol
            cfg["_dv_target_mps"] = target
            checks.append(_check_change_speed_delta)

        # --- lane(lane=N) → must start in lane N ---
        if name == "lane" and "lane" in args and isinstance(args["lane"], (int, float)):
            cfg["_lane_at_start"] = int(args["lane"])
            checks.append(_check_lane_at_start)

        # --- position(distance, ahead_of/behind, at: start|end) ---
        if name == "position":
            when = args.get("at", "start")
            if "ahead_of" in args or "behind" in args:
                npc = args.get("ahead_of") or args.get("behind")
                if npc and isinstance(npc, str):
                    npc_hints.append((ego, npc))
                cfg["_when"] = when
                cfg["_longitudinal"] = "ahead_of" if "ahead_of" in args else "behind"
                if "distance" in args:
                    lo, hi = _length_to_m(args["distance"])
                    cfg["_dist_lo_hi_m"] = (lo, hi)
                checks.append(_check_ahead_or_behind)

        # --- lateral(distance?, side_of: npc, side: left|right|same, at) ---
        if name == "lateral" and "side_of" in args and "side" in args:
            npc = args["side_of"]
            if npc and isinstance(npc, str):
                npc_hints.append((ego, npc))
            cfg["_when"] = args.get("at", "start")
            cfg["_side"] = str(args["side"])
            checks.append(_check_lateral_side_of)

    # dedupe hints
    npc_hints = list({h for h in npc_hints})

    Q = BlockQuery(
        ego=ego,
        npc_candidates=None if not npc_hints else [npc for _, npc in npc_hints],
        duration_frames=duration_frames,
        allow_shorter_end=allow_shorter,
        cfg=cfg,
        checks=checks,
    )
    return Q, npc_hints or None

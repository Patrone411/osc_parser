# osc_parser/matching/single_call_fast.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# ----- small unit helpers -----
def _to_mps(val: float, unit: str) -> float:
    u = (unit or "").lower()
    if u in ("m/s", "meter_per_second"):
        return float(val)
    if u in ("kph", "km/h", "kilometer_per_hour"):
        return float(val) * (1000.0/3600.0)
    if u in ("mph", "mile_per_hour"):
        return float(val) * 0.44704
    return float(val)

def _to_m(val: float, unit: str) -> float:
    u = (unit or "").lower()
    if u in ("m", "meter", "metre"):
        return float(val)
    if u in ("km", "kilometer"):
        return float(val) * 1000.0
    if u in ("cm", "centimeter"):
        return float(val) * 0.01
    if u in ("mm", "millimeter"):
        return float(val) * 0.001
    if u in ("ft", "feet"):
        return float(val) * 0.3048
    if u in ("mi", "mile"):
        return float(val) * 1609.344
    return float(val)

def _duration_frames(action_args: Dict[str, Any], fps: int, fallback_frames: int) -> int:
    dur = action_args.get("duration")
    if isinstance(dur, dict) and "value" in dur:
        # your IR normalizes units; if unit starts with 'second', treat as seconds
        seconds = float(dur["value"])
        return max(1, int(round(seconds * fps)))
    return fallback_frames

# ----- compiled spec for a single call -----
@dataclass
class SingleCallSpec:
    actor: str
    duration_frames: int
    cfg: Dict[str, Any]
    npc: Optional[str] = None

    # lane@start
    lane_start: Optional[int] = None

    # speed window (whole window coverage)
    speed_window: Optional[Tuple[float, float]] = None  # (lo_mps, hi_mps)

    # speed at specific frame
    speed_at_start: Optional[float] = None  # m/s
    speed_at_end: Optional[float]   = None  # m/s

    # relational (longitudinal) constraints at start/end
    rel_start_need: Optional[str] = None       # "front" or "back"
    rel_start_dist: Optional[Tuple[float,float]] = None  # meters range (lo,hi) or (d,d)
    rel_end_need: Optional[str] = None
    rel_end_dist: Optional[Tuple[float,float]] = None

def spec_from_call(call: Dict[str, Any], fps: int, cfg: Dict[str, Any]) -> SingleCallSpec:
    """
    call is from constraints_from_ir(...)"calls_flat"[i]:
      { "actor": "...", "action": "...",
        "action_args": {...}, "modifiers": [ {"name":..., "args": {...}}, ... ] }
    """
    actor = call["actor"]
    action_args = call.get("action_args", {}) or {}
    modifiers   = call.get("modifiers", []) or {}

    fallback_frames = int(round(float(cfg.get("default_window_s", 5.0)) * fps))
    dur = _duration_frames(action_args, fps, fallback_frames)

    spec = SingleCallSpec(actor=actor, duration_frames=dur, cfg=dict(cfg))

    # parse modifiers
    for m in modifiers:
        name = m["name"]
        args = m.get("args", {}) or {}

        if name in ("speed", "change_speed"):
            sp = args.get("speed")
            at = args.get("at")
            if isinstance(sp, dict) and "range" in sp:
                lo, hi = sp["range"]
                unit = sp.get("unit", "meter_per_second")
                spec.speed_window = (_to_mps(float(lo), unit), _to_mps(float(hi), unit))
            elif isinstance(sp, dict) and "value" in sp:
                unit = sp.get("unit", "meter_per_second")
                v = _to_mps(float(sp["value"]), unit)
                if at == "start":
                    spec.speed_at_start = v
                elif at == "end":
                    spec.speed_at_end = v
                else:
                    spec.speed_window = (v, v)

        elif name == "lane":
            lane = args.get("lane")
            if lane is not None:
                spec.lane_start = int(lane)

        elif name in ("position", "distance"):
            at = args.get("at", "start")
            if "ahead_of" in args:
                spec.npc = spec.npc or args["ahead_of"]
                need = "front"
                dist = args.get("distance")
                if isinstance(dist, dict) and "range" in dist:
                    lo, hi = dist["range"]
                    rng = (_to_m(float(lo), "meter"), _to_m(float(hi), "meter"))
                elif isinstance(dist, dict) and "value" in dist:
                    d = _to_m(float(dist["value"]), dist.get("unit", "meter"))
                    rng = (d, d)
                else:
                    rng = None
                if at == "start":
                    spec.rel_start_need, spec.rel_start_dist = need, rng
                else:
                    spec.rel_end_need, spec.rel_end_dist = need, rng

            elif "behind" in args:
                spec.npc = spec.npc or args["behind"]
                need = "back"
                dist = args.get("distance")
                if isinstance(dist, dict) and "range" in dist:
                    lo, hi = dist["range"]
                    rng = (_to_m(float(lo), "meter"), _to_m(float(hi), "meter"))
                elif isinstance(dist, dict) and "value" in dist:
                    d = _to_m(float(dist["value"]), dist.get("unit", "meter"))
                    rng = (d, d)
                else:
                    rng = None
                if at == "start":
                    spec.rel_start_need, spec.rel_start_dist = need, rng
                else:
                    spec.rel_end_need, spec.rel_end_dist = need, rng

    return spec

# ----- helpers for masks / prefix sums -----
def _prefix_sum_bool(b: np.ndarray) -> np.ndarray:
    # b is boolean or 0/1; returns int prefix sums of length T+1
    return np.concatenate([[0], np.cumsum(b.astype(np.int32))])

def _count_in(ps: np.ndarray, a: int, b: int) -> int:
    # inclusive window [a..b]
    return int(ps[b + 1] - ps[a])

def _ok_distance(d: float, rng: Optional[Tuple[float,float]], tol: float) -> bool:
    if d is None or not np.isfinite(d):
        return False
    if rng is None:
        return True
    lo, hi = rng
    return (d >= (lo - tol)) and (d <= (hi + tol))

# ----- main matcher for one call -----
def match_single_call(feats, spec: SingleCallSpec, fps: int = 10, max_results: int = 5000) -> List[Dict[str, Any]]:
    """
    feats: TagFeatures for one segment
    spec:  compiled SingleCallSpec
    returns: [{"ego": ..., "npc": npc_or_None, "t_start": t0, "t_end": t1}, ...]
    """
    T = feats.T
    D = max(1, int(spec.duration_frames))
    allow_shorter = bool(spec.cfg.get("allow_shorter_end", False))
    min_len = 1 if allow_shorter else D

    # determine actual ego & npc ids in this segment
    if spec.actor not in feats.actors:
        # no same-name actor in this segment → no matches
        return []
    ego = spec.actor
    npc = None
    if spec.npc is not None:
        if spec.npc not in feats.actors:
            return []
        npc = spec.npc

    # ----- per-frame boolean arrays -----
    # presence
    pres_ego = feats.present.get(ego, np.zeros((T,), dtype=float)) > 0.5
    ps_pres_ego = _prefix_sum_bool(pres_ego)
    pres_npc = None
    ps_pres_npc = None
    if npc:
        pres_npc = feats.present.get(npc, np.zeros((T,), dtype=float)) > 0.5
        ps_pres_npc = _prefix_sum_bool(pres_npc)

    # lane@start mask (point check at t0)
    lane_ok_start = None
    if spec.lane_start is not None:
        l = feats.lane_idx.get(ego)
        lane_ok_start = (np.isfinite(l)) & (l.astype(np.int64) == int(spec.lane_start))

    # speed window mask (window coverage)
    speed_ok = None
    ps_speed_ok = None
    if spec.speed_window is not None:
        lo, hi = spec.speed_window
        v = feats.speed.get(ego)
        if v is None:
            speed_ok = np.zeros((T,), dtype=bool)
        else:
            speed_ok = np.isfinite(v) & (v >= (lo - float(spec.cfg.get("speed_value_tol", 0.0)))) & (v <= (hi + float(spec.cfg.get("speed_value_tol", 0.0))))
        ps_speed_ok = _prefix_sum_bool(speed_ok)

    # speed at start/end (point checks)
    def _speed_at_ok(t: int, target: float) -> bool:
        v = feats.speed.get(ego)
        if v is None or t >= len(v): return False
        val = float(v[t])
        if not np.isfinite(val): return False
        return abs(val - target) <= float(spec.cfg.get("speed_value_tol", 0.1))

    # relative start/end (point checks)
    def _front_or_back_at(t: int, need: str) -> bool:
        if not npc: return False
        # simple heuristic: dx>0 => front, else back. Adjust to your axis if needed.
        dx = feats.x.get(npc)[t] - feats.x.get(ego)[t]
        if not np.isfinite(dx): return False
        return ("front" if dx > 0 else "back") == need

    def _dist_at(t: int) -> Optional[float]:
        if not npc: return None
        d = feats.rel_distance.get((ego, npc))
        if d is None or t >= len(d): return None
        val = float(d[t])
        return val if np.isfinite(val) else None

    # thresholds
    pres_min_cov = float(spec.cfg.get("presence_min_coverage", 1.0))
    pres_allow_missing = spec.cfg.get("presence_allow_missing")  # int or None
    speed_min_cov = float(spec.cfg.get("speed_min_coverage", 1.0))
    dist_tol = float(spec.cfg.get("distance_tol", 2.0))

    # sliding windows
    results: List[Dict[str, Any]] = []
    last_t0 = max(0, T - min_len)
    for t0 in range(0, last_t0 + 1):
        # lane at start
        if lane_ok_start is not None and not lane_ok_start[t0]:
            continue

        # presence: ego at least min coverage in any candidate window
        # We'll check exact coverage per (t0,t1) via prefix sums below.

        # relational at start
        if spec.rel_start_need is not None:
            if not _front_or_back_at(t0, spec.rel_start_need):
                continue
            if spec.rel_start_dist is not None:
                d = _dist_at(t0)
                if not _ok_distance(d, spec.rel_start_dist, dist_tol):
                    continue

        # scan t1
        t1_min = min(T - 1, t0 + min_len - 1)
        t1_max = min(T - 1, t0 + D - 1)

        found = None
        for t1 in range(t1_min, t1_max + 1):
            # presence coverage checks
            win_len = (t1 - t0 + 1)

            # ego presence
            ego_valid = _count_in(ps_pres_ego, t0, t1)
            if isinstance(pres_allow_missing, int):
                if (win_len - ego_valid) > pres_allow_missing:
                    continue
            else:
                if (ego_valid / win_len) < pres_min_cov:
                    continue

            # npc presence (if any)
            if npc:
                npc_valid = _count_in(ps_pres_npc, t0, t1)
                if isinstance(pres_allow_missing, int):
                    if (win_len - npc_valid) > pres_allow_missing:
                        continue
                else:
                    if (npc_valid / win_len) < pres_min_cov:
                        continue

            # speed window coverage
            if ps_speed_ok is not None:
                ok_count = _count_in(ps_speed_ok, t0, t1)
                if (ok_count / win_len) < speed_min_cov:
                    continue

            # speed at start/end (point)
            if spec.speed_at_start is not None:
                if not _speed_at_ok(t0, spec.speed_at_start):
                    continue
            if spec.speed_at_end is not None:
                if not _speed_at_ok(t1, spec.speed_at_end):
                    continue

            # relational at end (point)
            if spec.rel_end_need is not None:
                if not _front_or_back_at(t1, spec.rel_end_need):
                    continue
                if spec.rel_end_dist is not None:
                    d = _dist_at(t1)
                    if not _ok_distance(d, spec.rel_end_dist, dist_tol):
                        continue

            found = t1
            break

        if found is not None:
            results.append({"ego": ego, "npc": npc, "t_start": t0, "t_end": found})
            if len(results) >= max_results:
                return results

    return results
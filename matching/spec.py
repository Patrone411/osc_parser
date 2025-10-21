# osc_parser/matching/spec.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import math
import numpy as np

# ======================================================================================
# BlockQuery schema
# ======================================================================================
@dataclass
class BlockQuery:
    """
    What the matcher needs for a single-block/window query.
    """
    ego: str
    npc_candidates: List[str]          # if empty, driver may try all others
    duration_frames: int               # window length in frames (inclusive end)
    checks: List[Callable[..., bool]]  # fn(feats, ego, npc, t0, t1, cfg) -> bool
    cfg: Dict[str, Any] = field(default_factory=dict)


# ======================================================================================
# Units, normalization, small helpers
# ======================================================================================
_SPEED_UNITS = {
    "m/s": 1.0,
    "meter_per_second": 1.0,
    "meters_per_second": 1.0,
    "kilometer_per_hour": 1.0 / 3.6,
    "kilometers_per_hour": 1.0 / 3.6,
    "kph": 1.0 / 3.6,
    "km/h": 1.0 / 3.6,
    "mph": 0.44704,
}

_ANGLE_UNITS = {
    "rad": 1.0, "radian": 1.0, "radians": 1.0,
    "deg": math.pi / 180.0, "degree": math.pi / 180.0, "degrees": math.pi / 180.0,
}

_DISTANCE_UNITS = {
    "m": 1.0, "meter": 1.0, "meters": 1.0,
    "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
    "cm": 0.01, "millimeter": 0.001, "mm": 0.001, "feet": 0.3048, "ft": 0.3048,
}

_TIME_UNITS = {
    "s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
    "ms": 1e-3, "millisecond": 1e-3, "milliseconds": 1e-3,
    "min": 60.0, "minute": 60.0, "minutes": 60.0,
    "h": 3600.0, "hr": 3600.0, "hour": 3600.0, "hours": 3600.0,
}

_ACCEL_UNITS = {
    "m/s^2": 1.0, "mps2": 1.0,
    "km/h/s": 1000.0/3600.0, "kph/s": 1000.0/3600.0,
}

_JERK_UNITS = {
    "m/s^3": 1.0, "mps3": 1.0,
}

_DEFAULT_CFG = {

    # presence & coverage
    "presence_min_coverage": 0.9,
    "presence_allow_missing": 0,  # number of frames allowed missing (None to use coverage ratio instead)
    "speed_min_coverage": 0.9,    # also used for distance coverage unless you add a dedicated key

    # tolerances
    "speed_value_tol": 0.10,      # m/s
    "distance_tol": 2.0,          # m
    "change_speed_tol": 0.30,     # m/s

    # lateral / relation
    "relation_snap_radius": 1,    # (placeholder)
    "lateral_allow_missing": True,

    # duration handling
    "allow_shorter_end": True,
    "default_window_s": 5.0,

    # during semantics (window checks)
    "during_mode": "all",
    "during_max_false": 0,         # only used if during_mode == "all"
    "st_reach_tol_s": 1.0,
    "st_reach_tol_t": 0.5,

    # assign_orientation tolerance (radians)
    "yaw_reach_tol": 0.05,  # ≈ 2.9°
    "accel_value_tol": 0.2,  # m/s^2
    "stationary_max_false": 0,      # allow this many violations inside the window (0 = strict)
    "keep_speed_tol": 0.20,   # m/s deviation allowed from sampled speed
    "jerk_value_tol": 0.2,     # m/s^3 tolerance when checking rate_peak
    "accel_min_coverage": 0.9,     # coverage when during_mode == "coverage"

    # --- lane following knobs ---
    "lane_follow_mode": "all",
    "lane_follow_allow_false": 0,
    "lane_min_coverage": 0.95,

    # --- time headway knobs ---
    "time_headway_tol": 0.30,     # seconds; numeric tolerance for matching the target or sampled headway
    "headway_min_coverage": 0.85, # coverage threshold for keep_time_headway when during_mode == "coverage"
    "min_speed_for_headway": 0.30,# m/s; below this we consider headway undefined
}

def _check_change_lane_action(
    feats,
    ego: str,
    npc: Optional[str],          # unused here; kept for unified signature
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
    target_lane: Optional[int],
    num_of_lanes: Optional[int],
    side: Optional[str],
    reference: Optional[str],
) -> bool:
    """
    OSC 8.8.3.3 change_lane (recognizer, simplified):
      • If target lane is given → require lane_idx(ego,t1) == target and lane_idx(ego,t1) != lane_idx(ego,t0).
      • Else compute target from (num_of_lanes, side, reference):
            target = lane_idx(reference,t1)  (side == 'same'/'same_as')
            target = lane_idx(reference,t1) ± num_of_lanes  (left/right; default num_of_lanes = 1)
      • Presence and finiteness checks included.
      • We do NOT enforce lateral offset / shape here (needs lane geometry); can be added later.
    """
    import numpy as np

    le = _get(feats.lane_idx, ego)
    if le.size == 0 or t0 >= le.size or t1 >= le.size:
        return False
    if not (np.isfinite(le[t0]) and np.isfinite(le[t1])):
        return False
    start_lane = int(np.rint(le[t0]))
    end_lane   = int(np.rint(le[t1]))

    # Path 1: explicit target lane
    if target_lane is not None:
        tgt = int(target_lane)
        return (end_lane == tgt) and (end_lane != start_lane)

    # Path 2: compute from (num_of_lanes, side, reference)
    ref_name = reference or ego
    lr = _get(feats.lane_idx, ref_name)
    if lr.size == 0 or t1 >= lr.size or not np.isfinite(lr[t1]):
        return False
    ref_lane_end = int(np.rint(lr[t1]))

    side_l = (side or "").lower()
    if side_l in ("same", "same_as"):
        tgt = ref_lane_end
    elif side_l in ("left", "right"):
        n = int(num_of_lanes) if num_of_lanes is not None else 1
        sgn = _lane_sign_from_side(side_l)  # left=-1, right=+1 (consistent with modifier check)
        tgt = ref_lane_end + sgn * abs(n)
    else:
        # Not enough info to determine a target
        return False

    return (end_lane == tgt) and (end_lane != start_lane)

def _norm_physical(d: Dict[str, Any], kind: str) -> Dict[str, Any]:
    """
    Normalize {"value":..,"unit":..} / {"range":[lo,hi],"unit":..} to SI.
    kind ∈ {"speed","angle","distance","acceleration","jerk"}.
    """
    if not d:
        return {}
    if "value" in d:
        v = float(d["value"]); u = str(d.get("unit", "")).lower()
        if kind == "speed":
            return {"value": v * _SPEED_UNITS.get(u, 1.0), "unit": "m/s"}
        if kind == "angle":
            return {"value": v * _ANGLE_UNITS.get(u, 1.0), "unit": "rad"}
        if kind == "distance":
            return {"value": v * _DISTANCE_UNITS.get(u, 1.0), "unit": "m"}
        if kind == "acceleration":
            return {"value": v * _ACCEL_UNITS.get(u, 1.0), "unit": "m/s^2"}
        if kind == "jerk":
            return {"value": v * _JERK_UNITS.get(u, 1.0), "unit": "m/s^3"}
    if "range" in d:
        lo, hi = d["range"]; u = str(d.get("unit", "")).lower()
        if kind == "speed":
            s = _SPEED_UNITS.get(u, 1.0); return {"range": [float(lo)*s, float(hi)*s], "unit": "m/s"}
        if kind == "angle":
            s = _ANGLE_UNITS.get(u, 1.0); return {"range": [float(lo)*s, float(hi)*s], "unit": "rad"}
        if kind == "distance":
            s = _DISTANCE_UNITS.get(u, 1.0); return {"range": [float(lo)*s, float(hi)*s], "unit": "m"}
        if kind == "acceleration":
            s = _ACCEL_UNITS.get(u, 1.0); return {"range": [float(lo)*s, float(hi)*s], "unit": "m/s^2"}
        if kind == "jerk":
            s = _JERK_UNITS.get(u, 1.0); return {"range": [float(lo)*s, float(hi)*s], "unit": "m/s^3"}
    return d

def _anchor_index(t0: int, t1: int, at: Optional[str]) -> int:
    return t1 if (str(at).lower() == "end") else t0

def _val_or_range_to_bounds(spec: Dict[str, Any], tol: float = 0.0) -> Tuple[float, float]:
    if not spec:
        return (-np.inf, np.inf)
    if "value" in spec:
        v = float(spec["value"])
        return v - tol, v + tol
    if "range" in spec:
        lo, hi = spec["range"]
        return float(lo), float(hi)
    return (-np.inf, np.inf)

def _coverage_ratio(x: np.ndarray, lo: float, hi: float) -> float:
    m = np.isfinite(x)
    if not np.any(m): return 0.0
    ok = (x >= lo) & (x <= hi) & m
    return float(np.sum(ok) / np.sum(m))

def _presence_coverage(pres: np.ndarray) -> float:
    if pres is None or len(pres) == 0: return 0.0
    m = np.isfinite(pres)
    if not np.any(m): return 0.0
    return float(np.mean(pres[m] > 0.5))

def _all_during_in_range(x: np.ndarray, lo: float, hi: float, pres: Optional[np.ndarray], max_false: int) -> bool:
    x = np.asarray(x, dtype=float)
    m = np.isfinite(x)
    if pres is not None:
        m &= (np.asarray(pres) > 0.5)
    if not np.any(m):
        return False
    ok = (x >= lo) & (x <= hi)
    violations = int(np.sum(m & ~ok))
    return violations <= int(max_false)

def _get(arr_map: Dict[str, np.ndarray], key: str) -> np.ndarray:
    a = arr_map.get(key)
    return np.asarray(a) if a is not None else np.array([], dtype=float)

def _lane_sign_from_side(side: Optional[str]) -> int:
    s = (side or "").lower()
    return -1 if s == "left" else (1 if s == "right" else 0)

def _to_seconds_unit(u: Optional[str]) -> float:
    if not u:
        return 1.0
    return _TIME_UNITS.get(str(u).lower(), 1.0)

def _time_val_or_range_to_bounds(spec: Dict[str, Any], tol: float) -> Tuple[float, float]:
    """
    Normalize a 'time' Physical/Range dict into numeric bounds in **seconds**.
    Adds ±tol when a scalar value is provided (like _val_or_range_to_bounds).
    Accepts:
      {"value": 4.1, "unit": "s"}  → (4.1 - tol, 4.1 + tol)
      {"range": [4.0, 5.0], "unit": "s"} → (4.0, 5.0)
    Unit is optional; defaults to seconds.
    """
    if "value" in spec:
        k = _to_seconds_unit(spec.get("unit"))
        v = float(spec["value"]) * k
        return (v - float(tol), v + float(tol))
    if "range" in spec:
        k = _to_seconds_unit(spec.get("unit"))
        lo, hi = spec["range"]
        return (float(lo) * k, float(hi) * k)
    # Fallback: treat as 0±tol
    return (-tol, +tol)

def _signed_longitudinal_gap(feats, ego: str, npc: str, t: int) -> Optional[float]:
    """
    Prefer Frenet 's' to compute signed longitudinal gap:
      ds = s_ego - s_npc (meters). Returns None if any is invalid at t.
    """
    s_e = getattr(feats, "s", {}).get(ego)
    s_n = getattr(feats, "s", {}).get(npc)
    if s_e is None or s_n is None:
        return None
    if t >= len(s_e) or t >= len(s_n):
        return None
    se, sn = float(s_e[t]), float(s_n[t])
    if not (np.isfinite(se) and np.isfinite(sn)):
        return None
    return se - sn

def _ego_longitudinal_speed(feats, ego: str, t: int) -> Optional[float]:
    """
    Prefer Frenet s_dot (longitudinal speed). Fallback to scalar speed if needed.
    Returns signed s_dot if available, else +|speed| (non-negative).
    """
    sdot = getattr(feats, "s_dot", {}).get(ego)
    if sdot is not None and t < len(sdot):
        val = float(sdot[t])
        return val if np.isfinite(val) else None
    v = _get(feats.speed, ego)
    if t >= v.size:
        return None
    val = float(v[t])
    if not np.isfinite(val):
        return None
    return abs(val)  # no sign info in scalar speed

def _headway_time_mag(feats, ego: str, npc: str, t: int, min_speed: float) -> Optional[float]:
    """
    Compute |time headway| ≈ |Δs| / max(|v_long|, eps).
    Uses Frenet s & s_dot; falls back to scalar speed if s_dot missing.
    Returns None if Δs or speed invalid / too small.
    """
    ds = _signed_longitudinal_gap(feats, ego, npc, t)
    if ds is None:
        return None
    v_long = _ego_longitudinal_speed(feats, ego, t)
    if v_long is None:
        return None
    v_abs = abs(v_long)
    if not np.isfinite(v_abs) or v_abs < float(min_speed):
        return None
    return abs(ds) / v_abs


# ======================================================================================
# Checks (each returns bool for a given window [t0..t1], inclusive)
# ======================================================================================
def _check_presence(feats, ego, npc, t0, t1, cfg) -> bool:
    win = slice(t0, t1 + 1)
    pres_e = _get(feats.present, ego)[win]
    cov_e = _presence_coverage(pres_e)
    need = float(cfg.get("presence_min_coverage", _DEFAULT_CFG["presence_min_coverage"]))
    if cov_e < need:
        # allow integer missing if configured
        allow = cfg.get("presence_allow_missing", None)
        if isinstance(allow, int):
            if len(pres_e) - int(np.sum(pres_e > 0.5)) > allow:
                return False
        else:
            return False
    if npc is not None:
        pres_n = _get(feats.present, npc)[win]
        cov_n = _presence_coverage(pres_n)
        if cov_n < need:
            allow = cfg.get("presence_allow_missing", None)
            if isinstance(allow, int):
                if len(pres_n) - int(np.sum(pres_n > 0.5)) > allow:
                    return False
            else:
                return False
    return True

def _check_speed(feats, ego, npc, t0, t1, cfg, arg_speed: Dict[str, Any], at: Optional[str]) -> bool:
    spec = _norm_physical(arg_speed, "speed")
    tol = float(cfg.get("speed_value_tol", _DEFAULT_CFG["speed_value_tol"]))
    lo, hi = _val_or_range_to_bounds(spec, tol=tol)
    v = _get(feats.speed, ego)
    if at:
        ti = _anchor_index(t0, t1, at)
        if ti >= v.size or not np.isfinite(v[ti]): return False
        return (v[ti] >= lo) and (v[ti] <= hi)
    # during window
    sl = slice(t0, t1 + 1)
    
    mode = str(cfg.get("during_mode", _DEFAULT_CFG["during_mode"])).lower()
    if mode == "coverage":
        cov = _coverage_ratio(v[sl], lo, hi)
        need = float(cfg.get("speed_min_coverage", _DEFAULT_CFG["speed_min_coverage"]))
        return cov >= need
    else:
        max_false = int(cfg.get("during_max_false", _DEFAULT_CFG["during_max_false"]))
        pres = _get(feats.present, ego)[sl]
        return _all_during_in_range(v[sl], lo, hi, pres=pres, max_false=max_false)

def _check_position(feats, ego, npc, t0, t1, cfg, where: str, dist_arg: Optional[Dict[str, Any]], at: str) -> bool:
    if npc is None: return False
    ti = _anchor_index(t0, t1, at)
    pos = feats.rel_position.get((ego, npc))
    if pos is None or ti >= len(pos): return False
    need = "front" if where == "ahead_of" else "back"
    if str(pos[ti]) != need:
        return False
    if dist_arg:
        spec = _norm_physical(dist_arg, "distance")
        lo, hi = _val_or_range_to_bounds(spec, tol=float(cfg.get("distance_tol", _DEFAULT_CFG["distance_tol"])))
        d = feats.rel_distance.get((ego, npc))
        if d is None or ti >= len(d) or not np.isfinite(d[ti]): return False
        return (d[ti] >= lo) and (d[ti] <= hi)
    return True

def _check_lateral(feats, ego, npc, t0, t1, cfg, side: str, dist_arg: Optional[Dict[str, Any]], at: str) -> bool:
    if npc is None: return False
    ti = _anchor_index(t0, t1, at)
    lat = feats.lat_rel.get((ego, npc))
    if lat is None or ti >= len(lat): return False
    if str(lat[ti]).lower() != str(side).lower():
        return False
    if dist_arg:
        # try Frenet t gap if available
        t_e = _get(feats.t, ego)
        t_n = _get(feats.t, npc)
        if t_e.size == 0 or t_n.size == 0 or ti >= t_e.size or ti >= t_n.size:
            return bool(cfg.get("lateral_allow_missing", True))
        if not (np.isfinite(t_e[ti]) and np.isfinite(t_n[ti])):
            return bool(cfg.get("lateral_allow_missing", True))
        gap = abs(float(t_e[ti] - t_n[ti]))
        spec = _norm_physical(dist_arg, "distance")
        lo, hi = _val_or_range_to_bounds(spec, tol=float(cfg.get("distance_tol", _DEFAULT_CFG["distance_tol"])))
        return (gap >= lo) and (gap <= hi)
    return True

def _check_lane_number(feats, ego, npc, t0, t1, cfg, lane: int, at: str) -> bool:
    ti = _anchor_index(t0, t1, at)
    ls = _get(feats.lane_idx, ego)
    if ti >= ls.size or not np.isfinite(ls[ti]): return False
    return int(round(ls[ti])) == int(lane)

def _check_lane_same_as(feats, ego, npc, t0, t1, cfg, at: str) -> bool:
    if npc is None: return False
    ti = _anchor_index(t0, t1, at)
    le = _get(feats.lane_idx, ego)
    ln = _get(feats.lane_idx, npc)
    if ti >= le.size or ti >= ln.size: return False
    if not (np.isfinite(le[ti]) and np.isfinite(ln[ti])): return False
    return int(round(le[ti])) == int(round(ln[ti]))

def _check_lane_side_of(feats, ego, npc, t0, t1, cfg, lane: Optional[int], side: str, at: str) -> bool:
    if not _check_lateral(feats, ego, npc, t0, t1, cfg, side=side, dist_arg=None, at=at):
        return False
    if lane is None:
        return True
    return _check_lane_number(feats, ego, npc, t0, t1, cfg, lane=lane, at=at)

def _check_change_lane(feats, ego, npc, t0, t1, cfg, delta_lane: Dict[str, Any], side: Optional[str]) -> bool:
    le = _get(feats.lane_idx, ego)
    if le.size == 0 or t0 >= le.size or t1 >= le.size: return False
    if not (np.isfinite(le[t0]) and np.isfinite(le[t1])): return False
    delta = float(le[t1] - le[t0])
    # force near-integer net change
    if abs(delta - round(delta)) > 0.25:
        return False
    if "value" in (delta_lane or {}):
        k = float(delta_lane["value"])
        sgn = _lane_sign_from_side(side)
        if sgn != 0:
            k = sgn * abs(k)
        return abs(delta - k) <= 0.3
    if "range" in (delta_lane or {}):
        lo, hi = delta_lane["range"]
        sgn = _lane_sign_from_side(side)
        if sgn < 0:
            lo, hi = -float(hi), -float(lo)
        elif sgn > 0:
            lo, hi = float(lo), float(hi)
        else:
            delta = abs(delta)
        return (delta >= float(lo)) and (delta <= float(hi))
    return False

def _check_change_speed(feats, ego, npc, t0, t1, cfg, delta_speed: Dict[str, Any]) -> bool:
    v = _get(feats.speed, ego)
    if v.size == 0 or t0 >= v.size or t1 >= v.size: return False
    dv = float(v[t1] - v[t0])
    spec = _norm_physical(delta_speed, "speed")
    lo, hi = _val_or_range_to_bounds(spec, tol=float(cfg.get("change_speed_tol", _DEFAULT_CFG["change_speed_tol"])))
    return (dv >= lo) and (dv <= hi)

def _check_acceleration(feats, ego, npc, t0, t1, cfg, accel_arg: Dict[str, Any], at: Optional[str]) -> bool:
    a = _get(feats.accel, ego)
    if a.size == 0: return False
    spec = _norm_physical(accel_arg or {}, "acceleration") 
    if at:
        ti = _anchor_index(t0, t1, at)
        if ti >= a.size or not np.isfinite(a[ti]): return False
        lo, hi = _val_or_range_to_bounds(spec, tol=0.0)
        return (a[ti] >= lo) and (a[ti] <= hi)
    aa = a[t0:t1+1]
    if not np.any(np.isfinite(aa)): return False
    mean_a = float(np.nanmean(aa))
    lo, hi = _val_or_range_to_bounds(spec, tol=0.0)
    return (mean_a >= lo) and (mean_a <= hi)

def _check_yaw(feats, ego, npc, t0, t1, cfg, angle_arg: Dict[str, Any], at: str) -> bool:
    yaw = _get(feats.yaw, ego)
    spec = _norm_physical(angle_arg, "angle")
    ti = _anchor_index(t0, t1, at)
    if ti >= yaw.size or not np.isfinite(yaw[ti]): return False
    lo, hi = _val_or_range_to_bounds(spec, tol=0.0)
    return (yaw[ti] >= lo) and (yaw[ti] <= hi)

def _check_yaw_delta(feats, ego, npc, t0, t1, cfg, angle_arg: Dict[str, Any], at: str) -> bool:
    ydel = _get(feats.yaw_delta, ego)
    spec = _norm_physical(angle_arg, "angle")
    ti = _anchor_index(t0, t1, at)
    if ti >= ydel.size or not np.isfinite(ydel[ti]): return False
    lo, hi = _val_or_range_to_bounds(spec, tol=0.0)
    return (ydel[ti] >= lo) and (ydel[ti] <= hi)

def _check_distance_traveled(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
    dist_arg: Dict[str, Any],
) -> bool:
    """
    Movement modifier distance:
      - Enforce that the distance traveled by 'ego' in [t0..t1] matches target.
      - Prefer Frenet s: d = |s[t1] - s[t0]|.
      - Optional fallback to XY arc length (accumulate segment lengths).
    """
    # 1) interpret target bounds (meters)
    spec = _norm_physical(dist_arg or {}, "distance")
    lo, hi = _val_or_range_to_bounds(spec, tol=float(cfg.get("distance_tol", 2.0)))

    # 2) prefer Frenet s difference
    s = getattr(feats, "s", {}).get(ego)
    pres = _get(feats.present, ego)
    if s is not None and t0 < len(s) and t1 < len(s):
        if pres.size and (t0 < pres.size and t1 < pres.size):
            if not (pres[t0] > 0.5 and pres[t1] > 0.5):
                # fall back to XY or fail
                pass
        s0, s1 = float(s[t0]), float(s[t1])
        if np.isfinite(s0) and np.isfinite(s1):
            d = abs(s1 - s0)
            return (d >= lo) and (d <= hi)

    # 3) fallback: accumulate XY segment lengths where present & finite
    x = _get(feats.x, ego); y = _get(feats.y, ego)
    if x.size and y.size and (t1 < x.size) and (t1 < y.size):
        sl = slice(t0, t1 + 1)
        xi = np.asarray(x[sl], dtype=float)
        yi = np.asarray(y[sl], dtype=float)
        pres = _get(feats.present, ego)[sl] > 0.5
        finite = np.isfinite(xi) & np.isfinite(yi)
        idx = np.where(pres & finite)[0]
        if idx.size >= 2:
            dsum = 0.0
            for k in range(idx.size - 1):
                i, j = idx[k], idx[k + 1]
                dx = xi[j] - xi[i]
                dy = yi[j] - yi[i]
                dsum += float(np.hypot(dx, dy))
            return (dsum >= lo) and (dsum <= hi)

    return False

"""def _check_distance(feats, ego, npc, t0, t1, cfg, dist_arg: Dict[str, Any], at: Optional[str]) -> bool:
    if npc is None: return False
    d = feats.rel_distance.get((ego, npc))
    if d is None: return False
    spec = _norm_physical(dist_arg, "distance")
    lo, hi = _val_or_range_to_bounds(spec, tol=float(cfg.get("distance_tol", _DEFAULT_CFG["distance_tol"])))
    if at:
        ti = _anchor_index(t0, t1, at)
        if ti >= len(d) or not np.isfinite(d[ti]): return False
        return (d[ti] >= lo) and (d[ti] <= hi)
    sl = slice(t0, t1 + 1)
    mode = str(cfg.get("during_mode", _DEFAULT_CFG["during_mode"])).lower()
    if mode == "coverage":
        cov = _coverage_ratio(np.asarray(d[sl], dtype=float), lo, hi)
        need = float(cfg.get("speed_min_coverage", _DEFAULT_CFG["speed_min_coverage"]))
        return cov >= need
    else:
        max_false = int(cfg.get("during_max_false", _DEFAULT_CFG["during_max_false"]))
        pres = _get(feats.present, ego)[sl]  # or AND ego&npc if you prefer
        return _all_during_in_range(np.asarray(d[sl], dtype=float), lo, hi, pres=pres, max_false=max_false)"""

def _check_speed_same_as(feats, ego, npc, t0, t1, cfg, at: Optional[str]) -> bool:
    if npc is None: return False
    v_e = _get(feats.speed, ego)
    v_n = _get(feats.speed, npc)
    tol = float(cfg.get("speed_value_tol", _DEFAULT_CFG["speed_value_tol"]))
    if at:
        ti = _anchor_index(t0, t1, at)
        if ti >= v_e.size or ti >= v_n.size: return False
        if not (np.isfinite(v_e[ti]) and np.isfinite(v_n[ti])): return False
        return abs(float(v_e[ti] - v_n[ti])) <= tol
    ve = v_e[t0:t1+1]; vn = v_n[t0:t1+1]
    m = np.isfinite(ve) & np.isfinite(vn)
    if not np.any(m): return False
    diff = abs(float(np.nanmean(ve[m]) - np.nanmean(vn[m])))
    return diff <= tol

def _check_change_space_gap(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
    target_arg: Dict[str, Any],
    direction: str,
) -> bool:
    """
    OSC 8.8.3.4 change_space_gap:
      - direction in {ahead, behind} → use Δs (longitudinal)
      - direction in {left, right}   → use Δt (lateral)
      - inside/outside requires map.driving_rule → not implemented here
    Target is measured along the chosen axis, not Euclidean.

    Ends when the target gap is achieved at window end (t1).
    """
    if npc is None:
        return False

    dir_l = str(direction or "").lower()
    # Normalize target as a distance in meters
    spec = _norm_physical(target_arg or {}, "distance")
    if "value" in spec:
        tgt = float(spec["value"])
        lo, hi = tgt, tgt
    elif "range" in spec:
        lo, hi = map(float, spec["range"])
    else:
        return False

    tol = float(cfg.get("space_gap_tol", cfg.get("distance_tol", 2.0)))

    ti = t1  # action semantics: "at end"
    # Safeguard presence/size
    def _fin(v): return (v is not None) and np.isfinite(v)

    if dir_l in ("ahead", "behind"):
        # Prefer categorical front/back from rel_position
        pos = feats.rel_position.get((ego, npc))
        if pos is None or ti >= len(pos):
            return False

        need = "front" if dir_l == "ahead" else "back"
        if str(pos[ti]) != need:
            return False

        # Use Frenet s if available
        s_e = feats.s.get(ego); s_n = feats.s.get(npc)
        if s_e is None or s_n is None or ti >= len(s_e) or ti >= len(s_n):
            return False
        se = float(s_e[ti]); sn = float(s_n[ti])
        if not (_fin(se) and _fin(sn)):
            return False

        # Δs with sign: ego ahead ⇒ se - sn > 0
        ds = se - sn
        val = ds if dir_l == "ahead" else -ds  # always compare positive target
        return (val >= (lo - tol)) and (val <= (hi + tol))

    if dir_l in ("left", "right"):
        # Prefer categorical left/right from lat_rel for sign
        lat = feats.lat_rel.get((ego, npc))
        if lat is None or ti >= len(lat):
            return False
        need = dir_l
        if str(lat[ti]).lower() != need:
            return False

        # Use Frenet t if available
        t_e = feats.t.get(ego); t_n = feats.t.get(npc)
        if t_e is None or t_n is None or ti >= len(t_e) or ti >= len(t_n):
            # optional: allow missing if configured
            return bool(cfg.get("lateral_allow_missing", True))
        te = float(t_e[ti]); tn = float(t_n[ti])
        if not (_fin(te) and _fin(tn)):
            return bool(cfg.get("lateral_allow_missing", True))

        dt = te - tn  # left-of means positive per Frenet convention
        val = abs(dt)  # magnitude must match target; side enforced via lat_rel
        return (val >= (lo - tol)) and (val <= (hi + tol))

    # inside/outside would need driving rules (not implemented)
    if bool(cfg.get("debug_match_block", False)):
        print(f"[change_space_gap] direction '{direction}' not supported (need inside/outside logic)")
    return False

def _check_keep_space_gap(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
    direction: str,
) -> bool:
    """
    OSC 8.8.3.5 keep_space_gap:
      - Sample the space gap at action start (t0) along the requested direction.
      - Enforce that gap (magnitude and sign/orientation) 'during' the window [t0..t1].
      - direction ∈ {"longitudinal", "lateral"}  (inside/outside needs driving rules → not implemented)

    Uses Frenet coordinates if available:
      longitudinal → Δs = s_ego - s_npc
      lateral      → Δt = t_ego - t_npc  (left positive)

    We enforce:
      • |Δaxis| ≈ |Δaxis(t0)| within ±space_gap_tol
      • sign(Δaxis) consistent with sign at t0 (unless magnitude near 0)
      • (optional) categorical consistency via rel_position/lat_rel when present
    """
    if npc is None:
        return False

    dir_l = (direction or "").lower()
    sl = slice(t0, t1 + 1)

    tol = float(cfg.get("space_gap_tol", cfg.get("distance_tol", 2.0)))
    during_mode = str(cfg.get("during_mode", _DEFAULT_CFG["during_mode"])).lower()
    during_max_false = int(cfg.get("during_max_false", 0))
    cov_need = float(cfg.get("speed_min_coverage", 0.9))  # reuse coverage threshold

    def _ok_sign(a: float, b: float, eps: float) -> bool:
        # allow sign flip only if one of them is near zero (≤ eps)
        if not (np.isfinite(a) and np.isfinite(b)):
            return False
        if abs(a) <= eps or abs(b) <= eps:
            return True
        return (a >= 0) == (b >= 0)

    if dir_l in ("longitudinal", "long", "s"):
        s_e = feats.s.get(ego)
        s_n = feats.s.get(npc)
        if s_e is None or s_n is None:
            return False
        if t0 >= len(s_e) or t0 >= len(s_n):
            return False
        if not (np.isfinite(s_e[t0]) and np.isfinite(s_n[t0])):
            return False

        gap0 = float(s_e[t0] - s_n[t0])
        # series during window
        se = np.asarray(s_e[sl], dtype=float)
        sn = np.asarray(s_n[sl], dtype=float)
        m = np.isfinite(se) & np.isfinite(sn)
        if not np.any(m):
            return False
        gaps = se - sn

        mag_ok = np.zeros_like(m, dtype=bool)
        sign_ok = np.zeros_like(m, dtype=bool)
        mag_ok[m] = np.abs(np.abs(gaps[m]) - abs(gap0)) <= tol
        sign_ok[m] = np.array([_ok_sign(g, gap0, tol) for g in gaps[m]])

        # optional categorical check
        pos = feats.rel_position.get((ego, npc))
        if pos is not None:
            pos_arr = np.asarray(pos[sl], dtype=object)
            need = "front" if gap0 >= 0 else "back"
            pos_m = (pos_arr == need)
            # treat 'unknown' as missing; only enforce where known
            known = (pos_arr == "front") | (pos_arr == "back")
            cat_ok = ~known | pos_m
            ok = mag_ok & sign_ok & cat_ok
        else:
            ok = mag_ok & sign_ok

    elif dir_l in ("lateral", "lat", "t"):
        t_e = feats.t.get(ego)
        t_n = feats.t.get(npc)
        if t_e is None or t_n is None:
            # allow missing if configured (mirrors lateral distance behavior)
            return bool(cfg.get("lateral_allow_missing", True))
        if t0 >= len(t_e) or t0 >= len(t_n):
            return False
        if not (np.isfinite(t_e[t0]) and np.isfinite(t_n[t0])):
            # if we can't sample target at t0, fail (semantics: sample on invoke)
            return False

        gap0 = float(t_e[t0] - t_n[t0])   # left positive
        te = np.asarray(t_e[sl], dtype=float)
        tn = np.asarray(t_n[sl], dtype=float)
        m = np.isfinite(te) & np.isfinite(tn)
        if not np.any(m):
            return bool(cfg.get("lateral_allow_missing", True))
        gaps = te - tn

        mag_ok = np.zeros_like(m, dtype=bool)
        sign_ok = np.zeros_like(m, dtype=bool)
        mag_ok[m] = np.abs(np.abs(gaps[m]) - abs(gap0)) <= tol
        sign_ok[m] = np.array([_ok_sign(g, gap0, tol) for g in gaps[m]])

        # optional categorical check: enforce left/right where known
        lat = feats.lat_rel.get((ego, npc))
        if lat is not None:
            lat_arr = np.asarray(lat[sl], dtype=object)
            need = "left" if gap0 >= 0 else "right"
            lat_m = (lat_arr == need)
            known = (lat_arr == "left") | (lat_arr == "right")
            cat_ok = ~known | lat_m
            ok = mag_ok & sign_ok & cat_ok
        else:
            ok = mag_ok & sign_ok
    else:
        # inside/outside not supported without driving rules
        return False

    # apply "during" semantics
    if during_mode == "coverage":
        cov = float(np.sum(ok) / np.sum(m)) if np.any(m) else 0.0
        return cov >= cov_need
    else:
        violations = int(np.sum(m & ~ok))
        return violations <= during_max_false
    
def _parse_unit_scale(d: Dict[str, Any]) -> float:
    u = str(d.get("unit", "m")).lower()
    return _DISTANCE_UNITS.get(u, 1.0)

def _check_assign_position_xy(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
    x_target: float,
    y_target: float,
) -> bool:
    """
    assign_position to absolute XY (map frame). Succeeds when the end-anchored
    distance to (x*,y*) is within position_reach_tol.
    """
    x = _get(feats.x, ego)
    y = _get(feats.y, ego)
    if t1 >= x.size or t1 >= y.size:
        return False
    if not (np.isfinite(x[t1]) and np.isfinite(y[t1])):
        return False
    tol = float(cfg.get("position_reach_tol", 1.0))
    dx = float(x[t1] - x_target)
    dy = float(y[t1] - y_target)
    return (dx*dx + dy*dy) <= (tol * tol)

def _check_assign_position_st(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
    s_target: float,
    t_target: float,
) -> bool:
    """
    assign_position to Frenet (s,t). Succeeds when |s-s*| <= st_reach_tol_s and
    |t-t*| <= st_reach_tol_t at window end.
    """
    s = feats.s.get(ego)
    t = feats.t.get(ego)
    if s is None or t is None:
        return False
    if t1 >= len(s) or t1 >= len(t):
        return False
    if not (np.isfinite(s[t1]) and np.isfinite(t[t1])):
        return False
    tol_s = float(cfg.get("st_reach_tol_s", 1.0))
    tol_t = float(cfg.get("st_reach_tol_t", 0.5))
    return (abs(float(s[t1] - s_target)) <= tol_s) and (abs(float(t[t1] - t_target)) <= tol_t)

def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def _angdiff(a: float, b: float) -> float:
    return _wrap_pi(a - b)

def _check_assign_orientation_yaw(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
    yaw_arg: Dict[str, Any],
) -> bool:
    """
    assign_orientation: evaluate only yaw; action ends when yaw reaches target.
    End-anchored check: |wrap(yaw[t1] - yaw_target)| <= yaw_reach_tol
    """
    if isinstance(yaw_arg, (int, float)):
        spec = {"value": float(yaw_arg), "unit": "rad"}
    elif isinstance(yaw_arg, dict):
        spec = _norm_physical(yaw_arg, "angle")
    else:
        return False
    yaw_target = float(spec["value"])
    tol = float(cfg.get("yaw_reach_tol", 0.05))
    yaw = _get(feats.yaw, ego)
    if t1 >= yaw.size or not np.isfinite(yaw[t1]):
        return False
    return abs(_angdiff(float(yaw[t1]), yaw_target)) <= tol
    
def _check_assign_speed(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
    speed_arg: Dict[str, Any],
) -> bool:
    """
    assign_speed: end-anchored speed check.
    Succeeds when speed[ego][t1] matches the target (value or range).
    """
    if not isinstance(speed_arg, dict):
        return False
    spec = _norm_physical(speed_arg, "speed")
    tol = float(cfg.get("speed_value_tol", 0.10))
    lo, hi = _val_or_range_to_bounds(spec, tol=tol)

    v = _get(feats.speed, ego)
    if t1 >= v.size or not np.isfinite(v[t1]):
        return False
    val = float(v[t1])
    return (val >= lo) and (val <= hi)


def _check_assign_acceleration(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
    accel_arg: Dict[str, Any],
) -> bool:
    """
    assign_acceleration: end-anchored scalar acceleration check.
    Succeeds when accel[ego][t1] matches the target (value or range).
    """
    if not isinstance(accel_arg, dict):
        return False
    spec = _norm_physical(accel_arg, "acceleration")
    tol = float(cfg.get("accel_value_tol", 0.2))
    lo, hi = _val_or_range_to_bounds(spec, tol=tol)

    a = _get(feats.accel, ego)
    if t1 >= a.size or not np.isfinite(a[t1]):
        return False
    val = float(a[t1])
    return (val >= lo) and (val <= hi)

def _check_remain_stationary(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
) -> bool:
    """
    OSC 8.8.2.10 remain_stationary:
      - Throughout [t0..t1], translational speed must be ~0 in all directions.
      - We enforce:
         • scalar speed <= stationary_speed_tol
         • and, where available, |s_dot| <= stationary_axis_tol and |t_dot| <= stationary_axis_tol
      - Uses strict “during” semantics with up to 'stationary_max_false' allowed violations.
    """
    import numpy as np

    sl = slice(t0, t1 + 1)
    tol_v  = float(cfg.get("stationary_speed_tol", 0.15))
    tol_ax = float(cfg.get("stationary_axis_tol", 0.15))
    max_false = int(cfg.get("stationary_max_false", 0))

    # presence mask
    pres = feats.present.get(ego)
    if pres is None:
        return False
    m_pres = np.asarray(pres[sl], dtype=float) > 0.5
    if not np.any(m_pres):
        return False

    # scalar speed
    v = _get(feats.speed, ego)  # assumes helper exists; returns 1D np.array
    v_win = np.asarray(v[sl], dtype=float)
    v_ok = (~np.isfinite(v_win)) | (np.abs(v_win) <= tol_v)

    # s_dot / t_dot (optional; only enforce where finite)
    sdot_ok = None
    tdot_ok = None

    sdot = getattr(feats, "s_dot", {}).get(ego) if hasattr(feats, "s_dot") else None
    if sdot is not None:
        s_win = np.asarray(sdot[sl], dtype=float)
        sdot_ok = (~np.isfinite(s_win)) | (np.abs(s_win) <= tol_ax)

    tdot = getattr(feats, "t_dot", {}).get(ego) if hasattr(feats, "t_dot") else None
    if tdot is not None:
        t_win = np.asarray(tdot[sl], dtype=float)
        tdot_ok = (~np.isfinite(t_win)) | (np.abs(t_win) <= tol_ax)

    ok = v_ok
    if sdot_ok is not None:
        ok = ok & sdot_ok
    if tdot_ok is not None:
        ok = ok & tdot_ok

    # Only evaluate on present frames
    eval_mask = m_pres
    violations = int(np.sum(eval_mask & ~ok))
    return violations <= max_false

def _check_keep_speed(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
) -> bool:
    """
    OSC 8.8.2.13 keep_speed:
      - Sample scalar speed at action start (t0) → v0
      - Require |v(t) - v0| <= keep_speed_tol during [t0..t1]
      - Apply 'during' semantics (coverage or universal) and only evaluate
        frames where ego is present and speed is finite.
    """
    import numpy as np

    v = _get(feats.speed, ego)
    if t0 >= v.size or not np.isfinite(v[t0]):
        return False
    v0 = float(v[t0])

    sl = slice(t0, t1 + 1)
    v_win = np.asarray(v[sl], dtype=float)

    pres = feats.present.get(ego)
    if pres is None:
        return False
    pres_win = np.asarray(pres[sl], dtype=float) > 0.5

    finite = np.isfinite(v_win)
    eval_mask = pres_win & finite
    if not np.any(eval_mask):
        return False

    tol = float(cfg.get("keep_speed_tol", cfg.get("speed_value_tol", 0.1)))
    ok_series = np.zeros_like(eval_mask, dtype=bool)
    ok_series[eval_mask] = np.abs(v_win[eval_mask] - v0) <= tol

    during_mode = str(cfg.get("during_mode", _DEFAULT_CFG["during_mode"])).lower()
    if during_mode == "coverage":
        need = float(cfg.get("speed_min_coverage", 0.9))
        cov = float(np.sum(ok_series) / np.sum(eval_mask))
        return cov >= need
    else:
        max_false = int(cfg.get("during_max_false", 0))
        violations = int(np.sum(eval_mask & ~ok_series))
        return violations <= max_false

def _check_change_acceleration(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
    target_accel_arg: Dict[str, Any],
    rate_profile: Optional[str] = None,
    rate_peak_arg: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    OSC 8.8.2.14 change_acceleration
      - Always require end-anchored accel target at t1.
      - If 'rate_peak' (jerk) provided, require that within [t0..t1] the max |jerk| >= target.
      - 'rate_profile' is parsed but not shaped here (macro support could be added later).

    Jerk is estimated from accel by 1st difference: j[t] ≈ (a[t] - a[t-1]) * fps.
    """
    import numpy as np

    # 1) End-anchored acceleration target
    spec = _norm_physical(target_accel_arg, "acceleration")
    a_tol = float(cfg.get("accel_value_tol", 0.2))
    lo, hi = _val_or_range_to_bounds(spec, tol=a_tol)

    a = _get(feats.accel, ego)
    if t1 >= a.size or not np.isfinite(a[t1]):
        return False
    a_end = float(a[t1])
    if not (lo <= a_end <= hi):
        return False

    # 2) Optional jerk peak requirement
    if rate_peak_arg is not None:
        # interpret jerk units (assume m/s^3 if unit omitted)
        # accept {"value": x, "unit": "m/s^3"} or {"range":[lo,hi], "unit":...}
        j_tol = float(cfg.get("jerk_value_tol", 0.2))
        # Minimal normalization: reuse _norm_physical; falls back to raw if unit unknown
        j_spec = _norm_physical(rate_peak_arg, "jerk")  # may just pass through numbers
        # Convert to [lo,hi] where lo is the minimum required ABS jerk peak
        if "value" in j_spec:
            j_req_lo = float(j_spec["value"]) - j_tol
            j_req_hi = float("inf")  # peak ≥ target
        elif "range" in j_spec:
            j_req_lo = float(min(j_spec["range"])) - j_tol
            j_req_hi = float("inf")  # interpret as at least the lower bound
        else:
            return False

        fps = float(cfg.get("fps", 10.0))
        sl = slice(max(0, t0), t1 + 1)

        a_win = np.asarray(a[sl], dtype=float)
        pres = feats.present.get(ego)
        if pres is None:
            return False
        pres_win = np.asarray(pres[sl], dtype=float) > 0.5

        # finite jerk samples exist where both a[t] and a[t-1] are finite
        if a_win.size < 2:
            return False
        a_prev = a_win[:-1]
        a_curr = a_win[1:]
        pres_prev = pres_win[:-1]
        pres_curr = pres_win[1:]
        valid = np.isfinite(a_prev) & np.isfinite(a_curr) & pres_prev & pres_curr
        if not np.any(valid):
            return False

        jerk = (a_curr - a_prev) * fps
        peak_abs = float(np.nanmax(np.abs(jerk[valid])))
        if not (peak_abs >= j_req_lo):
            return False

    return True

def _check_keep_acceleration(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
) -> bool:
    """
    OSC 8.8.2.15 keep_acceleration:
      - Sample scalar longitudinal acceleration at start (t0) -> a0
      - Require |a(t) - a0| <= keep_accel_tol during [t0..t1]
      - Apply 'during' semantics (coverage vs universal) on frames where ego is present & accel is finite.
    """
    import numpy as np

    a = _get(feats.accel, ego)
    if t0 >= a.size or not np.isfinite(a[t0]):
        return False
    a0 = float(a[t0])

    sl = slice(t0, t1 + 1)
    a_win = np.asarray(a[sl], dtype=float)

    pres = feats.present.get(ego)
    if pres is None:
        return False
    pres_win = np.asarray(pres[sl], dtype=float) > 0.5

    finite = np.isfinite(a_win)
    eval_mask = pres_win & finite
    if not np.any(eval_mask):
        return False

    tol = float(cfg.get("keep_accel_tol", cfg.get("accel_value_tol", 0.2)))
    ok_series = np.zeros_like(eval_mask, dtype=bool)
    ok_series[eval_mask] = np.abs(a_win[eval_mask] - a0) <= tol

    during_mode = str(cfg.get("during_mode", _DEFAULT_CFG["during_mode"])).lower()

    if during_mode == "coverage":
        need = float(cfg.get("accel_min_coverage", cfg.get("speed_min_coverage", 0.9)))
        cov = float(np.sum(ok_series) / np.sum(eval_mask))
        return cov >= need
    else:
        max_false = int(cfg.get("during_max_false", 0))
        violations = int(np.sum(eval_mask & ~ok_series))
        return violations <= max_false

def _check_follow_lane(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
    target_lane: Optional[int],
) -> bool:
    """
    OSC 8.8.3.2 follow_lane (simplified):
      • Actor remains in the SAME lane from start to end.
        - If target_lane is provided: lane == target_lane for the whole window.
        - Else: lane == lane(t0) for the whole window.

    Uses:
      feats.lane_idx[ego] (lane equality over [t0..t1])
      feats.present[ego]  (evaluation mask)
    """
    import numpy as np

    sl = slice(t0, t1 + 1)

    # presence mask for evaluation
    pres = feats.present.get(ego)
    if pres is None:
        return False
    pres_win = np.asarray(pres[sl], dtype=float) > 0.5
    if not np.any(pres_win):
        return False

    # lane series
    lane_series = _get(feats.lane_idx, ego)
    lane_win = np.asarray(lane_series[sl], dtype=float)
    lane_known = np.isfinite(lane_win)
    eval_lane = pres_win & lane_known
    if not np.any(eval_lane):
        return False

    # required lane id
    if target_lane is not None:
        lane_req = int(target_lane)
    else:
        l0 = float(lane_series[t0]) if t0 < lane_series.size else np.nan
        if not np.isfinite(l0):
            return False
        lane_req = int(np.rint(l0))

    lane_eq = np.rint(lane_win[eval_lane]).astype(int) == lane_req

    # gating semantics
    lane_mode = str(cfg.get("lane_follow_mode", "all")).lower()
    if lane_mode == "coverage":
        need = float(cfg.get("lane_min_coverage", 0.95))
        cov = float(np.sum(lane_eq) / np.sum(eval_lane))
        return cov >= need
    else:
        allow = int(cfg.get("lane_follow_allow_false", 0))
        violations = int(np.sum(~lane_eq))
        return violations <= allow

def _check_change_time_headway(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
    target_time_arg: Dict[str, Any],
    direction: str,
) -> bool:
    """
    OSC 8.8.3.6 change_time_headway:
      - At END (t1), enforce:
          • categorical orientation vs reference (ahead/back)
          • |time_headway(ego,ref)| within target bounds (seconds)
      - time_headway ≈ |Δs| / |v_long(ego)|
    """
    if npc is None:
        return False
    dir_l = (direction or "").lower()
    if dir_l not in ("ahead", "behind"):
        return False

    # categorical orientation at end (prefer rel_position)
    pos = feats.rel_position.get((ego, npc))
    if pos is None or t1 >= len(pos):
        return False
    need = "front" if dir_l == "ahead" else "back"
    if str(pos[t1]) != need:
        return False

    # target bounds (seconds)
    spec = _norm_physical(target_time_arg, "time") if isinstance(target_time_arg, dict) else {"value": target_time_arg}
    tol = float(cfg.get("time_headway_tol", 0.30))
    lo, hi = _time_val_or_range_to_bounds(spec, tol=tol)

    # headway magnitude at end
    min_speed = float(cfg.get("min_speed_for_headway", 0.30))
    h_mag = _headway_time_mag(feats, ego, npc, t1, min_speed=min_speed)
    if h_mag is None:
        return False
    return (h_mag >= lo) and (h_mag <= hi)

def _check_keep_time_headway(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
) -> bool:
    """
    OSC 8.8.3.7 keep_time_headway:
      - Sample target headway at start (t0): h0 (signed via rel_position if available).
        We actually enforce the **magnitude** with categorical orientation 'during' the window.
      - During [t0..t1], require:
          • orientation (front/back) matches sign of h0 wherever known
          • |headway_mag(t) - |h0|| <= tol on evaluable frames (presence & speed >= min)
      - Apply 'during' semantics via coverage (headway_min_coverage) or 'all' with allowed violations.
    """
    if npc is None:
        return False

    # Determine sign via rel_position at t0; fall back to sign(Δs)
    sign0 = None
    pos0 = feats.rel_position.get((ego, npc))
    if pos0 is not None and t0 < len(pos0):
        if str(pos0[t0]) == "front":
            sign0 = +1.0
        elif str(pos0[t0]) == "back":
            sign0 = -1.0

    ds0 = _signed_longitudinal_gap(feats, ego, npc, t0)
    if ds0 is None and sign0 is None:
        return False
    if sign0 is None and ds0 is not None:
        sign0 = 1.0 if ds0 >= 0.0 else -1.0

    min_speed = float(cfg.get("min_speed_for_headway", 0.30))
    h0_mag = _headway_time_mag(feats, ego, npc, t0, min_speed=min_speed)
    if h0_mag is None:
        return False

    # window series
    sl = slice(t0, t1 + 1)
    pres = feats.present.get(ego)
    if pres is None:
        return False
    pres_win = np.asarray(pres[sl], dtype=float) > 0.5

    # orientation categorical over window
    pos_series = feats.rel_position.get((ego, npc))
    orient_ok = None
    if pos_series is not None:
        pos_win = np.asarray(pos_series[sl], dtype=object)
        need_str = "front" if sign0 >= 0 else "back"
        known = (pos_win == "front") | (pos_win == "back")
        orient_ok = (~known) | (pos_win == need_str)

    # numeric headway magnitude over window (no feats.T dependency)
    idxs = list(range(t0, t1 + 1))
    eval_mask = np.zeros(len(idxs), dtype=bool)
    h_arr = np.full(len(idxs), np.nan, dtype=float)
    for k, ti in enumerate(idxs):
        if not pres_win[k]:
            continue
        h = _headway_time_mag(feats, ego, npc, ti, min_speed=min_speed)
        if h is None:
            continue
        h_arr[k] = h
        eval_mask[k] = True
    if not np.any(eval_mask):
        return False

    # compare magnitudes within tolerance
    tol = float(cfg.get("time_headway_tol", 0.30))
    mag_ok = np.zeros_like(eval_mask, dtype=bool)
    mag_ok[eval_mask] = np.abs(h_arr[eval_mask] - h0_mag) <= tol

    ok = mag_ok
    if orient_ok is not None:
        ok = ok & orient_ok

    during_mode = str(cfg.get("during_mode", _DEFAULT_CFG["during_mode"])).lower()
    if during_mode == "coverage":
        need = float(cfg.get("headway_min_coverage", 0.85))
        cov = float(np.sum(ok & eval_mask) / np.sum(eval_mask))
        return cov >= need
    else:
        max_false = int(cfg.get("during_max_false", 0))
        violations = int(np.sum(eval_mask & ~ok))
        return violations <= max_false

def _check_keep_lane(
    feats,
    ego: str,
    npc: Optional[str],
    t0: int,
    t1: int,
    cfg: Dict[str, Any],
) -> bool:
    """
    Movement modifier keep_lane():
      • Actor remains in the SAME lane from start to end,
        using the lane at t0 as the implicit target.
      • Implemented by delegating to _check_follow_lane with target_lane=None.
    """
    return _check_follow_lane(feats, ego, npc, t0, t1, cfg, target_lane=None)
    

# ======================================================================================
# Compiler: build_block_query(call, fps, cfg) → (BlockQuery, candidate_pairs_or_None)
# ======================================================================================
def build_block_query(call: Dict[str, Any], fps: int, cfg: Optional[Dict[str, Any]] = None) -> Tuple[BlockQuery, Optional[List[Tuple[str, Optional[str]]]]]:
    """
    Translate one normalized call (from constraints_from_ir) into a BlockQuery.
    """
    cfg = {**_DEFAULT_CFG, **(cfg or {})}
    cfg.setdefault("fps", float(fps))
    ego = call.get("actor")
    if not ego:
        raise ValueError("build_block_query: missing 'actor' in call")

    # --- duration → frames ---
    dur = (call.get("action_args") or {}).get("duration") or {}
    if "value" in dur:
        val = float(dur["value"]); unit = str(dur.get("unit", "second")).lower()
        if unit in ("frame", "frames"):
            duration_frames = max(1, int(round(val)))
        else:
            duration_frames = max(1, int(round(val * float(fps))))
    else:
        duration_frames = max(1, int(round(float(cfg.get("default_window_s", 5.0)) * float(fps))))

    checks: List[Callable[..., bool]] = []
    referenced: List[str] = []

    def _ref(actor_name: Optional[str]):
        if actor_name and actor_name != ego and actor_name not in referenced:
            referenced.append(actor_name)

    # always gate by presence coverage
    checks.append(lambda F, E, N, t0, t1, C: _check_presence(F, E, N, t0, t1, C))

    # --- ACTION ARGUMENTS (not modifiers) ---
    action_name = str(call.get("action", "")).lower()
    aargs = (call.get("action_args") or {})

    # Helpers already defined above:
    #  - _ref(actor_name)
    #  - _parse_unit_scale()
    #  - _check_* functions
    if action_name == "change_lane":
        # Two signatures (OSC 2.0):
        #   1) num_of_lanes:uint, side:lane_change_side, reference:physical_object [, offset, rate_profile, rate_peak]
        #   2) target: lane [, offset, rate_profile, rate_peak]
        a_num  = aargs.get("num_of_lanes") or aargs.get("num_lanes") or aargs.get("count")
        a_side = aargs.get("side")
        a_ref  = aargs.get("reference") or ego

        target_lane = aargs.get("target")
        if isinstance(target_lane, dict) and "lane" in target_lane:
            # validator/IR may provide {"lane": <int>} for resolved targets
            target_lane = target_lane.get("lane")

        # record referenced actor (if any)
        if a_ref and a_ref != ego:
            referenced.append(a_ref)

        # (Offset / rate_profile / rate_peak intentionally ignored in this recognizer; see docstring)
        checks.append(
            (lambda target_lane=target_lane, a_num=a_num, a_side=a_side, a_ref=a_ref:
                (lambda F, E, N, t0, t1, C:
                    _check_change_lane_action(F, E, N, t0, t1, C, target_lane, a_num, a_side, a_ref)))
        )
    
    elif action_name == "assign_position":
        pos_arg  = aargs.get("position")
        rp_arg   = aargs.get("route_point")
        odr_arg  = aargs.get("odr_point")

        provided = [x for x in (pos_arg, rp_arg, odr_arg) if x is not None]
        if len(provided) != 1:
            raise ValueError("assign_position requires exactly one of {position, route_point, odr_point}")

        target = provided[0]
        if not isinstance(target, dict):
            raise ValueError("assign_position target must be a dict containing either (x,y) or (s,t) or an ODR dict")

        # XY -> absolute map coordinates
        if ("x" in target) and ("y" in target):
            scale = _parse_unit_scale(target)
            xt = float(target["x"]) * scale
            yt = float(target["y"]) * scale
            checks.append(
                (lambda xt=xt, yt=yt:
                    (lambda F, E, N, t0, t1, C: _check_assign_position_xy(F, E, N, t0, t1, C, xt, yt)))
            )

        # ST -> Frenet coordinates
        elif ("s" in target) and ("t" in target):
            scale = _parse_unit_scale(target)
            st = float(target["s"]) * scale
            tt = float(target["t"]) * scale
            checks.append(
                (lambda st=st, tt=tt:
                    (lambda F, E, N, t0, t1, C: _check_assign_position_st(F, E, N, t0, t1, C, st, tt)))
            )

        # ODR dict (road/lane/…): if numeric s/t present, treat as ST for now
        elif {"road", "lane"} <= {k.lower() for k in target.keys()}:
            s_val = target.get("s"); t_val = target.get("t")
            if (s_val is not None) and (t_val is not None):
                st = float(s_val); tt = float(t_val)
                checks.append(
                    (lambda st=st, tt=tt:
                        (lambda F, E, N, t0, t1, C: _check_assign_position_st(F, E, N, t0, t1, C, st, tt)))
                )
            # else: leave for future ODR resolution (no check added)
        else:
            raise ValueError("assign_position target must include (x,y) or (s,t), or be an ODR dict with road/lane/(s,t)")
        

    elif action_name == "change_space_gap":
        target = aargs.get("target")
        direction = aargs.get("direction")
        reference = aargs.get("reference")
        if not (target and direction and reference):
            raise ValueError("change_space_gap requires 'target', 'direction', and 'reference'")
        _ref(reference)
        checks.append(
            (lambda target=target, direction=direction, reference=reference:
                (lambda F, E, N, t0, t1, C:
                    _check_change_space_gap(F, E, reference, t0, t1, C, target, direction)))
        )

    elif action_name == "keep_space_gap":
        direction = aargs.get("direction")
        reference = aargs.get("reference")
        if not reference:
            raise ValueError("keep_space_gap requires 'reference'")
        _ref(reference)
        checks.append(
            (lambda direction=direction, reference=reference:
                (lambda F, E, N, t0, t1, C:
                    _check_keep_space_gap(F, E, reference, t0, t1, C, direction)))
        )

    elif action_name == "change_time_headway":
        target = aargs.get("target")
        direction = aargs.get("direction")
        reference = aargs.get("reference")
        if not (target and direction and reference):
            raise ValueError("change_time_headway requires 'target', 'direction', and 'reference'")
        _ref(reference)
        checks.append(
            (lambda target=target, direction=direction, reference=reference:
                (lambda F, E, N, t0, t1, C:
                    _check_change_time_headway(F, E, reference, t0, t1, C, target, direction)))
        )

    elif action_name == "keep_time_headway":
        reference = aargs.get("reference")
        if not reference:
            raise ValueError("keep_time_headway requires 'reference'")
        _ref(reference)
        checks.append(
            (lambda reference=reference:
                (lambda F, E, N, t0, t1, C:
                    _check_keep_time_headway(F, E, reference, t0, t1, C)))
        )

    elif action_name == "follow_lane":
        target_lane = aargs.get("target")
        if isinstance(target_lane, dict) and "lane" in target_lane:
            target_lane = target_lane.get("lane")
        if target_lane is not None:
            target_lane = int(target_lane)
        checks.append(
            (lambda target_lane=target_lane:
                (lambda F, E, N, t0, t1, C:
                    _check_follow_lane(F, E, N, t0, t1, C, target_lane)))
        )

    elif action_name == "remain_stationary":
        checks.append((lambda: (lambda F, E, N, t0, t1, C: _check_remain_stationary(F, E, N, t0, t1, C)))())

    elif action_name == "keep_speed":
        checks.append((lambda: (lambda F, E, N, t0, t1, C: _check_keep_speed(F, E, N, t0, t1, C)))())

    elif action_name == "change_acceleration":
        target = aargs.get("target") or aargs.get("acceleration")
        rp     = aargs.get("rate_profile")
        peak   = aargs.get("rate_peak")
        if target is None:
            raise ValueError("change_acceleration requires 'target' acceleration")
        checks.append(
            (lambda target=target, rp=rp, peak=peak:
                (lambda F, E, N, t0, t1, C:
                    _check_change_acceleration(F, E, N, t0, t1, C, target, rp, peak)))
        )

    elif action_name == "keep_acceleration":
        checks.append((lambda: (lambda F, E, N, t0, t1, C: _check_keep_acceleration(F, E, N, t0, t1, C)))())

    # --- assign_speed ---
    elif action_name == "assign_speed":
        # accept new 'target' and legacy 'speed'
        speed_arg = aargs.get("target") or aargs.get("speed")
        if speed_arg is None:
            raise ValueError("assign_speed requires 'target' (or legacy 'speed')")
        checks.append(
            (lambda speed_arg=speed_arg:
                (lambda F, E, N, t0, t1, C:
                    _check_assign_speed(F, E, N, t0, t1, C, speed_arg)))
        )

    elif action_name == "assign_acceleration":
        accel_arg = aargs.get("target") or aargs.get("acceleration") or aargs.get("accel")
        if accel_arg is None:
            raise ValueError("assign_acceleration requires 'target' (or legacy 'acceleration')")
        checks.append(
            (lambda accel_arg=accel_arg:
                (lambda F, E, N, t0, t1, C:
                    _check_assign_acceleration(F, E, N, t0, t1, C, accel_arg)))
        )

    elif action_name == "assign_orientation":
        # Expect orientation_3d dict; allow plain yaw angle as a shorthand
        tgt = aargs.get("target") or aargs.get("orientation") or aargs.get("orientation_3d")
        yaw_arg = None
        if isinstance(tgt, dict):
            # canonical orientation_3d
            yaw_arg = tgt.get("yaw") or tgt.get("angle")  # allow {'angle': ...} fallback
        else:
            # allow a direct angle Physical as shorthand
            yaw_arg = tgt
        if yaw_arg is None:
            raise ValueError("assign_orientation requires 'target' with a yaw angle")
        checks.append(
            (lambda yaw_arg=yaw_arg:
                (lambda F, E, N, t0, t1, C:
                    _check_assign_orientation_yaw(F, E, N, t0, t1, C, yaw_arg)))
        )
    
    # --- translate modifiers ---
    for m in (call.get("modifiers") or []):
        name = str(m.get("name", "")).lower()
        args = m.get("args", {}) or {}

        if name == "speed":
            if "same_as" in args:
                other = args.get("same_as"); _ref(other)
                at = args.get("at")
                checks.append(lambda other=other, at=at:
                              (lambda F, E, N, t0, t1, C: _check_speed_same_as(F, E, other, t0, t1, C, at)))
            else:
                at = args.get("at")
                sp = args.get("speed") or {}
                checks.append(lambda sp=sp, at=at:
                              (lambda F, E, N, t0, t1, C: _check_speed(F, E, N, t0, t1, C, sp, at)))

        elif name == "position":
            at = args.get("at", "start")
            if "ahead_of" in args:
                other = args.get("ahead_of"); _ref(other)
                dist = args.get("distance")
                checks.append(lambda other=other, dist=dist, at=at:
                              (lambda F, E, N, t0, t1, C: _check_position(F, E, other, t0, t1, C, "ahead_of", dist, at)))
            elif "behind" in args:
                other = args.get("behind"); _ref(other)
                dist = args.get("distance")
                checks.append(lambda other=other, dist=dist, at=at:
                              (lambda F, E, N, t0, t1, C: _check_position(F, E, other, t0, t1, C, "behind", dist, at)))

        elif name == "lateral":
            other = args.get("side_of"); _ref(other)
            side = args.get("side")
            at = args.get("at", "start")
            dist = args.get("distance")
            checks.append(lambda other=other, side=side, dist=dist, at=at:
                          (lambda F, E, N, t0, t1, C: _check_lateral(F, E, other, t0, t1, C, side, dist, at)))

        elif name == "lane":
            at = args.get("at", "start")
            # NOTE: accept optional 'from' to be spec-friendly; not used in this checker
            _from = args.get("from")  # parsed, currently unused

            if "same_as" in args:
                other = args.get("same_as"); _ref(other)
                checks.append(lambda other=other, at=at:
                              (lambda F, E, N, t0, t1, C: _check_lane_same_as(F, E, other, t0, t1, C, at)))
                
            elif "side_of" in args and "side" in args:
                other = args.get("side_of"); _ref(other)
                side = args.get("side")
                lane = args.get("lane")  # optional
                checks.append(lambda other=other, side=side, lane=lane, at=at:
                              (lambda F, E, N, t0, t1, C: _check_lane_side_of(F, E, other, t0, t1, C, lane, side, at)))
            elif "lane" in args:
                lane = int(args.get("lane"))
                checks.append(lambda lane=lane, at=at:
                              (lambda F, E, N, t0, t1, C: _check_lane_number(F, E, N, t0, t1, C, lane, at)))

        elif name == "change_lane":
            # Accept lane delta as:
            #   • lane: {value: ±k} or plain number
            #   • side: left|right  (or alias: from)
            #   If lane delta missing but side/from provided → default to 1
            dl = args.get("lane")
            if isinstance(dl, (int, float)):
                delta_lane = {"value": float(dl)}
            else:
                delta_lane = dl or None

            side = args.get("side") or args.get("from")

            if delta_lane is None and side is not None:
                delta_lane = {"value": 1}  # default one-lane shift to the given side

            # If still nothing to check, skip adding a check
            if delta_lane is not None:
                checks.append(
                    (lambda d=delta_lane, side=side:
                        (lambda F, E, N, t0, t1, C:
                            _check_change_lane(F, E, N, t0, t1, C, d, side)))
                )

        elif name == "change_speed":
            dv = args.get("speed") or {}
            checks.append(lambda dv=dv:
                          (lambda F, E, N, t0, t1, C: _check_change_speed(F, E, N, t0, t1, C, dv)))

        elif name == "acceleration":
            at = args.get("at")
            acc = args.get("accel") or {}
            checks.append(lambda acc=acc, at=at:
                          (lambda F, E, N, t0, t1, C: _check_acceleration(F, E, N, t0, t1, C, acc, at)))

        elif name == "yaw":
            at = args.get("at", "start")
            ang = args.get("angle") or {}
            checks.append(lambda ang=ang, at=at:
                          (lambda F, E, N, t0, t1, C: _check_yaw(F, E, N, t0, t1, C, ang, at)))

        elif name == "yaw_delta":
            at = args.get("at", "start")
            ang = args.get("angle") or {}
            checks.append(lambda ang=ang, at=at:
                          (lambda F, E, N, t0, t1, C: _check_yaw_delta(F, E, N, t0, t1, C, ang, at)))

        elif name == "distance":
            # Spec: distance traveled over the window, NOT spacing to another actor
            dist = args.get("distance") or {}
            checks.append(
                (lambda dist=dist:
                    (lambda F, E, N, t0, t1, C:
                        _check_distance_traveled(F, E, N, t0, t1, C, dist)))
            )
    
        # --- keep_speed ---
        elif name == "keep_speed":
            # Enforce that the sampled speed at t0 is maintained during [t0..t1]
            checks.append(
                (lambda: (lambda F, E, N, t0, t1, C: _check_keep_speed(F, E, N, t0, t1, C)))()
            )

        elif name == "keep_lane":
            # Keep the current lane over the whole window; target is lane(t0)
            checks.append(
                (lambda: (lambda F, E, N, t0, t1, C: _check_keep_lane(F, E, N, t0, t1, C)))()
            )

    Q = BlockQuery(
        ego=ego,
        npc_candidates=referenced[:],
        duration_frames=int(duration_frames),
        checks=checks,
        cfg=cfg
    )

    pairs = None
    if referenced:
        pairs = [(ego, r) for r in referenced]

    # Debug
    print(f"[build_block_query] ego={ego}  duration_frames={duration_frames}  fps={fps}")
    if referenced:
        print(f"[build_block_query] referenced NPCs: {referenced}")
    print(f"[build_block_query] compiled checks: {len(checks)}")

    return Q, pairs

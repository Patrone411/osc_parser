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
    "during_mode": "coverage",     # "coverage" (fraction) or "all" (strict ∀t)
    "during_max_false": 0,         # only used if during_mode == "all"
}

def _norm_physical(d: Dict[str, Any], kind: str) -> Dict[str, Any]:
    """
    Normalize {"value":..,"unit":..} / {"range":[lo,hi],"unit":..} to SI.
    kind ∈ {"speed","angle","distance"}.
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
    if "range" in d:
        lo, hi = d["range"]; u = str(d.get("unit", "")).lower()
        if kind == "speed":
            s = _SPEED_UNITS.get(u, 1.0); return {"range": [float(lo)*s, float(hi)*s], "unit": "m/s"}
        if kind == "angle":
            s = _ANGLE_UNITS.get(u, 1.0); return {"range": [float(lo)*s, float(hi)*s], "unit": "rad"}
        if kind == "distance":
            s = _DISTANCE_UNITS.get(u, 1.0); return {"range": [float(lo)*s, float(hi)*s], "unit": "m"}
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
    spec = accel_arg or {}
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

def _check_distance(feats, ego, npc, t0, t1, cfg, dist_arg: Dict[str, Any], at: Optional[str]) -> bool:
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
        return _all_during_in_range(np.asarray(d[sl], dtype=float), lo, hi, pres=pres, max_false=max_false)

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
    during_mode = str(cfg.get("during_mode", "coverage")).lower()
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
    
# ======================================================================================
# Compiler: build_block_query(call, fps, cfg) → (BlockQuery, candidate_pairs_or_None)
# ======================================================================================
def build_block_query(call: Dict[str, Any], fps: int, cfg: Optional[Dict[str, Any]] = None) -> Tuple[BlockQuery, Optional[List[Tuple[str, Optional[str]]]]]:
    """
    Translate one normalized call (from constraints_from_ir) into a BlockQuery.
    """
    cfg = {**_DEFAULT_CFG, **(cfg or {})}

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
            delta_lane = args.get("lane") or {}
            side = args.get("side")
            checks.append(lambda d=delta_lane, side=side:
                          (lambda F, E, N, t0, t1, C: _check_change_lane(F, E, N, t0, t1, C, d, side)))

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
            at = args.get("at")
            other = args.get("to"); _ref(other)
            dist = args.get("distance") or {}
            checks.append(lambda other=other, dist=dist, at=at:
                          (lambda F, E, N, t0, t1, C: _check_distance(F, E, other, t0, t1, C, dist, at)))

        if name == "change_space_gap":
            target = args.get("target")
            direction = args.get("direction")
            reference = args.get("reference")
            if reference:
                # mark NPC candidate
                def _ref(actor_name: Optional[str]):
                    if actor_name and actor_name != ego and actor_name not in referenced:
                        referenced.append(actor_name)
                _ref(reference)
            else:
                # Without a reference we can't evaluate this action
                raise ValueError("change_space_gap requires 'reference' actor")

            # Add a single end-anchored check
            checks.append(
                (lambda target=target, direction=direction, reference=reference:
                    (lambda F, E, N, t0, t1, C:
                        _check_change_space_gap(F, E, reference, t0, t1, C, target, direction)))
            )
            
    if name == "keep_space_gap":
        direction = args.get("direction")
        reference = args.get("reference")
        if not reference:
            raise ValueError("keep_space_gap requires 'reference' actor")
        # register referenced NPC candidate
        def _ref(actor_name: Optional[str]):
            if actor_name and actor_name != ego and actor_name not in referenced:
                referenced.append(actor_name)
        _ref(reference)

        # Enforce the sampled gap 'during' [t0..t1]
        checks.append(
            (lambda direction=direction, reference=reference:
                (lambda F, E, N, t0, t1, C:
                    _check_keep_space_gap(F, E, reference, t0, t1, C, direction)))
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

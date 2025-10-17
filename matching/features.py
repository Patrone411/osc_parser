# osc_parser/matching/features.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
import json
import math
import numpy as np

# Lateral labels
LATERAL_LEFT   = "left"
LATERAL_RIGHT  = "right"
LATERAL_SAME   = "same"
LATERAL_UNKNOWN= "unknown"

REL_POS_ALLOWED = {"front", "back", "unknown"}

# Finite-difference settings
DEFAULT_FPS = 10  # Hz
DEFAULT_DT  = 1.0 / DEFAULT_FPS

# --- helpers -----------------------------------------------------------------

def _presence_from_s(s_list, T: int) -> Optional[np.ndarray]:
    """
    Presence = 1 when s[t] is not null (or not the literal "null"), else 0.
    Pads/crops to length T. Returns None if s_list is missing/empty.
    """
    if s_list is None:
        return None
    arr = np.asarray(s_list, dtype=object)
    if arr.size == 0:
        return None
    arr = _pad_or_crop(arr, T, fill=None)
    pres = np.ones((T,), dtype=float)
    # treat None or the string "null" as absent
    mask = (arr == None) | (arr == "null")  # noqa: E711
    pres[mask] = 0.0
    return pres

def _to_np(a, dtype=float):
    if a is None: return None
    try:
        arr = np.asarray(a, dtype=dtype)
        # empty -> None
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None

def _pad_or_crop(arr: Optional[np.ndarray], T: int, fill=np.nan) -> Optional[np.ndarray]:
    if arr is None: return None
    if arr.ndim != 1: arr = arr.ravel()
    n = arr.shape[0]
    if n == T:
        return arr
    if n > T:
        return arr[:T].copy()
    out = np.full((T,), fill, dtype=arr.dtype)
    out[:n] = arr
    return out

def _finite01(x):
    if x is None: return None
    y = np.ones_like(x, dtype=float)
    y[~np.isfinite(x)] = 0.0
    return y

def _accel_from_speed(v: Optional[np.ndarray], dt: float) -> np.ndarray:
    """
    Compute acceleration from speed using finite differences.
    - Central difference for interior: a[t] = (v[t+1]-v[t-1])/(2*dt)
    - Forward/backward difference at edges
    - If neighbors are not finite, result is NaN at that index
    """
    if v is None:
        return np.array([], dtype=float)
    v = np.asarray(v, dtype=float)
    T = v.shape[0]
    a = np.full((T,), np.nan, dtype=float)
    if T == 0:
        return a
    # interior central diff
    if T >= 3:
        v_prev = np.roll(v, 1)
        v_next = np.roll(v, -1)
        ok_c = np.isfinite(v_prev) & np.isfinite(v_next)
        ok_c[0] = False
        ok_c[-1] = False
        a[ok_c] = (v_next[ok_c] - v_prev[ok_c]) / (2.0 * dt)
    # forward at 0
    if T >= 2 and np.isfinite(v[0]) and np.isfinite(v[1]):
        a[0] = (v[1] - v[0]) / dt
    # backward at T-1
    if T >= 2 and np.isfinite(v[-1]) and np.isfinite(v[-2]):
        a[-1] = (v[-1] - v[-2]) / dt
    return a

def segment_ids(stitched: Dict[str, Any]) -> List[str]:
    """
    Return all segment ids. Your data stores them under 'road_segments'.
    """
    return list(stitched.get("road_segments", {}).keys())

def segment_num_lanes(stitched: Dict[str, Any], seg_id: str) -> Optional[int]:
    """
    num_lanes is stored per segment in 'road_segments'[seg_id]['num_lanes'].
    """
    seg = stitched.get("road_segments", {}).get(seg_id) or {}
    val = seg.get("num_lanes")
    try:
        return int(val) if val is not None else None
    except Exception:
        return None

def actor_ids_in_segment(stitched: Dict[str, Any], seg_id: str) -> List[str]:
    """
    Actor ids present in a given segment come from 'actor_activities_per_segment'[seg_id].
    """
    return list(stitched.get("actor_activities_per_segment", {}).get(seg_id, {}).keys())

    
#TODO check if thjis is actually doing something that makes sense
def segment_length(stitched: Dict[str, Any], seg_id: str) -> int:
    """
    Determine the number of frames for a segment.
    Prefer per-segment actor arrays (e.g., 's' or 'osc_lane_id').
    Fallback to global long_v series if needed.
    """
    acts = stitched.get("actor_activities_per_segment", {}).get(seg_id, {})
    max_len = 0
    for _actor, payload in acts.items():
        if not isinstance(payload, dict):
            continue
        for key in ("s", "osc_lane_id"):
            arr = payload.get(key)
            if isinstance(arr, list):
                max_len = max(max_len, len(arr))

    if max_len > 0:
        return max_len

    # Fallback: use any global long_v length
    long_v = (
        stitched.get("general_actor_activities", {})
                .get("actor_activities", {})
                .get("long_v", {})
    )
    for _actor, arr in long_v.items():
        if isinstance(arr, list):
            max_len = max(max_len, len(arr))

    return max_len  # 0 if truly nothing found
    
@dataclass
class TagFeatures:
    """Per-segment time series bundle used by the matcher."""
    segment_id: str
    num_lanes: int
    length_m: float
    actors: List[str]
    T: int

    # per-actor series (length T)
    speed: Dict[str, np.ndarray] = field(default_factory=dict)       # m/s
    yaw:   Dict[str, np.ndarray] = field(default_factory=dict)       # rad (if available; else NaN)
    x:     Dict[str, np.ndarray] = field(default_factory=dict)       # meters (map frame)
    y:     Dict[str, np.ndarray] = field(default_factory=dict)
    lane_idx: Dict[str, np.ndarray] = field(default_factory=dict)    # 1..N (0/NaN unknown)
    present:  Dict[str, np.ndarray] = field(default_factory=dict)    # 0/1 per frame
    accel:    Dict[str, np.ndarray] = field(default_factory=dict)    # m/s^2 (finite diff of speed)

    # NEW: lane-frame longitudinal/lateral kinematics (CARLA road ref frame)
    s:        Dict[str, np.ndarray] = field(default_factory=dict)    # m (longitudinal along lane)
    t:        Dict[str, np.ndarray] = field(default_factory=dict)    # m (lateral from centerline)
    s_dot:    Dict[str, np.ndarray] = field(default_factory=dict)    # m/s
    t_dot:    Dict[str, np.ndarray] = field(default_factory=dict)    # m/s
    s_ddot:   Dict[str, np.ndarray] = field(default_factory=dict)    # m/s^2
    t_ddot:   Dict[str, np.ndarray] = field(default_factory=dict)    # m/s^2
    yaw_delta:Dict[str, np.ndarray] = field(default_factory=dict)    # rad (heading minus lane tangent)

    # per-pair series (length T)
    rel_position: Dict[Tuple[str,str], np.ndarray] = field(default_factory=dict)   # "front"/"back"/"unknown"
    lat_rel: Dict[Tuple[str,str], np.ndarray] = field(default_factory=dict)   # "left"/"right"/"same"/"unknown"
    rel_distance: Dict[Tuple[str,str], np.ndarray] = field(default_factory=dict)  # meters

    # --------- factory ---------
    @staticmethod
    def load_json(path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return json.load(f)

    @classmethod
    def from_tag_json(cls, data: Dict[str, Any], seg_id: str) -> "TagFeatures":
        # lanes per segment
        seg_meta = (data.get("road_segments") or {}).get(seg_id, {})
        num_lanes = int(seg_meta.get("num_lanes", 0))
        num_segments = int(seg_meta.get("num_segments", 0)) if seg_meta.get("num_segments") is not None else 0
        length_m = float(num_segments) * 5.0

        # lane IDs per actor/segment + lane-frame kinematics (s/t, dot, ddot, yaw_delta)
        seg_acts = (data.get("segment_actor_data") or {}).get(seg_id, {}) or {}

        # global kinematics (all_payloads) – organize per actor id
        payloads = (data.get("general_actor_activities") or {}).get("all_payloads") or {}

        # actors listed for the segment (keys in seg_acts)
        actors = sorted(list(seg_acts.keys()))
        # choose T from the longest osc_lane_id array present
        T = 0
        for a in actors:
            lane = _to_np(seg_acts.get(a, {}).get("osc_lane_id"))
            if lane is not None:
                T = max(T, lane.shape[0])
        if T <= 0:
            # fallback: if no lanes, try x length of first actor found in payloads
            for a in payloads.keys():
                arr = _to_np(payloads.get(a, {}).get("x"))
                if arr is not None and arr.size > 0:
                    T = arr.shape[0]
                    break
        if T <= 0:
            raise ValueError(f"Cannot determine segment length for {seg_id}")

        # build per-actor series
        speed: Dict[str, np.ndarray] = {}
        yaw:   Dict[str, np.ndarray] = {}
        x:     Dict[str, np.ndarray] = {}
        y:     Dict[str, np.ndarray] = {}
        lane_idx: Dict[str, np.ndarray] = {}
        present:  Dict[str, np.ndarray] = {}
        accel:    Dict[str, np.ndarray] = {}

        # NEW: lane frame series containers
        s:        Dict[str, np.ndarray] = {}
        t:        Dict[str, np.ndarray] = {}
        s_dot:    Dict[str, np.ndarray] = {}
        t_dot:    Dict[str, np.ndarray] = {}
        s_ddot:   Dict[str, np.ndarray] = {}
        t_ddot:   Dict[str, np.ndarray] = {}
        yaw_delta:Dict[str, np.ndarray] = {}

        def _payload_for(actor: str) -> Dict[str, Any]:
            # payloads could be keyed by actor id or nested differently;
            # try direct first, then search shallowly.
            if actor in payloads:
                return payloads[actor] or {}
            # fallback: some dumps put arrays directly (flat dict of arrays)
            if isinstance(payloads, dict) and all(isinstance(v, list) for v in payloads.values()):
                return payloads
            return {}

        for a in actors:
            a_seg = seg_acts.get(a, {}) or {}

            # lanes
            l = _to_np(a_seg.get("osc_lane_id"), dtype=float)
            l = _pad_or_crop(l, T, fill=np.nan)
            if l is None: l = np.full((T,), np.nan)
            lane_idx[a] = l

            pay = _payload_for(a)

            # speed (long_v) → m/s (assume already m/s; if it's kph, convert here)
            v = _to_np(pay.get("long_v"), dtype=float)
            v = _pad_or_crop(v, T)
            speed[a] = v if v is not None else np.full((T,), np.nan)

            # yaw/x/y
            yy = _pad_or_crop(_to_np(pay.get("yaw"), dtype=float), T)
            xx = _pad_or_crop(_to_np(pay.get("x"), dtype=float), T)
            yy2= _pad_or_crop(_to_np(pay.get("y"), dtype=float), T)
            yaw[a] = yy if yy is not None else np.full((T,), np.nan)
            x[a]   = xx if xx is not None else np.full((T,), np.nan)
            y[a]   = yy2 if yy2 is not None else np.full((T,), np.nan)

            # presence: 1 if x & y are finite, else 0 (prefer 's' field if provided)
            s_list = a_seg.get("s")
            pres = _presence_from_s(s_list, T)
            if pres is None:
                pres = np.where(np.isfinite(x[a]) & np.isfinite(y[a]), 1.0, 0.0)
            present[a] = pres

            # acceleration from speed via finite differences
            accel[a] = _accel_from_speed(speed[a], DEFAULT_DT)

            # --- NEW: lane-frame signals from per-segment actor block ---
            s[a]         = _pad_or_crop(_to_np(a_seg.get("s"),       dtype=float), T)
            t[a]         = _pad_or_crop(_to_np(a_seg.get("t"),       dtype=float), T)
            s_dot[a]     = _pad_or_crop(_to_np(a_seg.get("s_dot"),   dtype=float), T)
            t_dot[a]     = _pad_or_crop(_to_np(a_seg.get("t_dot"),   dtype=float), T)
            s_ddot[a]    = _pad_or_crop(_to_np(a_seg.get("s_ddot"),  dtype=float), T)
            t_ddot[a]    = _pad_or_crop(_to_np(a_seg.get("t_ddot"),  dtype=float), T)
            yaw_delta[a] = _pad_or_crop(_to_np(a_seg.get("yaw_delta"), dtype=float), T)

            # fill Nones with NaNs so downstream code can rely on ndarray
            if s[a]      is None: s[a]      = np.full((T,), np.nan)
            if t[a]      is None: t[a]      = np.full((T,), np.nan)
            if s_dot[a]  is None: s_dot[a]  = np.full((T,), np.nan)
            if t_dot[a]  is None: t_dot[a]  = np.full((T,), np.nan)
            if s_ddot[a] is None: s_ddot[a] = np.full((T,), np.nan)
            if t_ddot[a] is None: t_ddot[a] = np.full((T,), np.nan)
            if yaw_delta[a] is None: yaw_delta[a] = np.full((T,), np.nan)

        # per-pair: lateral (from lanes)
        lat_rel: Dict[Tuple[str,str], np.ndarray] = {}

        for i in range(len(actors)):
            for j in range(len(actors)):
                if i == j: continue
                e = actors[i]; n = actors[j]
                # lateral from lane indices
                l_e = lane_idx[e]
                l_n = lane_idx[n]
                l_n = np.asarray(l_n, dtype=float)
                l_e = np.asarray(l_e, dtype=float)
                lbl = np.full((T,), LATERAL_UNKNOWN, dtype=object)
                known = np.isfinite(l_e) & np.isfinite(l_n)
                gt = np.zeros_like(known, dtype=bool)
                lt = np.zeros_like(known, dtype=bool)
                np.greater(l_n, l_e, where=known, out=gt)  # only compare where values are finite
                np.less(l_n,  l_e, where=known, out=lt)
                eq = (l_n == l_e) & known
                lbl[gt] = LATERAL_RIGHT
                lbl[lt] = LATERAL_LEFT
                lbl[eq] = LATERAL_SAME
                lat_rel[(e, n)] = lbl

        # --------- per-pair rel_position + rel_distance from inter_actor_activities ---------
        def _inter_map_for_segment(root: Dict[str, Any], seg: str) -> Dict[str, Any]:
            # Try common layouts; pick the one your exporter actually uses.
            if "inter_actor_activities_per_segment" in root:
                return root["inter_actor_activities_per_segment"].get(seg, {}) or {}
            seg_block = (root.get("segments") or {}).get(seg, {}) or {}
            if "inter_actor_activities" in seg_block:
                return seg_block["inter_actor_activities"] or {}
            return root.get("inter_actor_activities", {}) or {}

        inter_map = _inter_map_for_segment(data, seg_id)
        rel_position: Dict[Tuple[str,str], np.ndarray] = {}
        rel_distance: Dict[Tuple[str,str], np.ndarray] = {}

        for e in actors:
            for n in actors:
                if e == n: 
                    continue
                pair_block = (inter_map.get(e) or {}).get(n) or {}

                # --- position labels (mandatory) ---
                pos_seq = pair_block.get("position")
                if pos_seq is None:
                    raise KeyError(
                        f"[TagFeatures] Missing inter_actor_activities[{e}][{n}]['position'] for segment {seg_id}"
                    )
                pos_arr = np.asarray(pos_seq, dtype=object)
                pos_arr = _pad_or_crop(pos_arr, T, fill="unknown")
                mask_bad = ~np.isin(pos_arr, list(REL_POS_ALLOWED))
                if np.any(mask_bad):
                    pos_arr = pos_arr.copy()
                    pos_arr[mask_bad] = "unknown"
                rel_position[(e, n)] = pos_arr

                # --- euclidean distance (mandatory) ---
                dist_seq = pair_block.get("eucl_distance")
                if dist_seq is None:
                    dist_seq = pair_block.get("distance")
                    if dist_seq is None:
                        raise KeyError(
                            f"[TagFeatures] Missing inter_actor_activities[{e}][{n}]['eucl_distance'] / ['distance'] for segment {seg_id}"
                        )
                dist_arr = _to_np(dist_seq, dtype=float)
                dist_arr = _pad_or_crop(dist_arr, T, fill=np.nan)
                if dist_arr is None:
                    raise ValueError(
                        f"[TagFeatures] eucl_distance empty for ({e},{n}) in segment {seg_id}"
                    )
                dist_arr = dist_arr.copy()
                dist_arr[dist_arr < 0] = 0.0
                rel_distance[(e, n)] = dist_arr

        return cls(
            segment_id=seg_id,
            num_lanes=num_lanes,
            length_m=length_m,
            actors=actors,
            T=T,
            speed=speed,
            yaw=yaw,
            x=x,
            y=y,
            lane_idx=lane_idx,
            present=present,
            accel=accel,
            # lane-frame series
            s=s, t=t, s_dot=s_dot, t_dot=t_dot, s_ddot=s_ddot, t_ddot=t_ddot, yaw_delta=yaw_delta,
            rel_position=rel_position,
            lat_rel=lat_rel,
            rel_distance=rel_distance,
        )

    # convenience: build features for all segments (optionally filtered by min lanes)
    @classmethod
    def load_all_segments(cls, data: Dict[str, Any], min_lanes: Optional[int] = None) -> Dict[str, "TagFeatures"]:
        feats_by_seg: Dict[str, TagFeatures] = {}
        road = data.get("road_segments") or {}
        for seg_id, meta in road.items():
            nlanes = int(meta.get("num_lanes", 0))
            if min_lanes is not None and nlanes < int(min_lanes):
                continue
            feats_by_seg[seg_id] = cls.from_tag_json(data, seg_id)
        return feats_by_seg

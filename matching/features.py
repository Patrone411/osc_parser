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

# --- add this helper near the top of features.py ---
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

@dataclass
class TagFeatures:
    """Per-segment time series bundle used by the matcher."""
    segment_id: str
    num_lanes: int
    actors: List[str]
    T: int

    # per-actor series (length T)
    speed: Dict[str, np.ndarray] = field(default_factory=dict)       # m/s
    yaw:   Dict[str, np.ndarray] = field(default_factory=dict)       # rad (if available; else NaN)
    x:     Dict[str, np.ndarray] = field(default_factory=dict)       # meters (map frame)
    y:     Dict[str, np.ndarray] = field(default_factory=dict)
    lane_idx: Dict[str, np.ndarray] = field(default_factory=dict)    # 1..N (0/NaN unknown)
    present:  Dict[str, np.ndarray] = field(default_factory=dict)    # 0/1 per frame

    # per-pair series (length T)
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

        # lane IDs per actor/segment
        seg_acts = (data.get("actor_activities_per_segment") or {}).get(seg_id, {}) or {}

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
            # lanes
            l = _to_np(seg_acts.get(a, {}).get("osc_lane_id"), dtype=float)
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

            # presence: 1 if x & y are finite, else 0
            s_list = seg_acts.get(a, {}).get("s")
            pres = _presence_from_s(s_list, T)
            if pres is None:
                pres = np.where(np.isfinite(x[a]) & np.isfinite(y[a]), 1.0, 0.0)
            present[a] = pres

        # per-pair: lateral + distance
        lat_rel: Dict[Tuple[str,str], np.ndarray] = {}
        rel_distance: Dict[Tuple[str,str], np.ndarray] = {}

        for i in range(len(actors)):
            for j in range(len(actors)):
                if i == j: continue
                e = actors[i]; n = actors[j]
                # lateral from lane indices
                l_e = lane_idx[e]
                l_n = lane_idx[n]
                lbl = np.full((T,), LATERAL_UNKNOWN, dtype=object)
                known = np.isfinite(l_e) & np.isfinite(l_n)
                gt = (l_n > l_e) & known
                lt = (l_n < l_e) & known
                eq = (l_n == l_e) & known
                lbl[gt] = LATERAL_RIGHT
                lbl[lt] = LATERAL_LEFT
                lbl[eq] = LATERAL_SAME
                lat_rel[(e, n)] = lbl

                # distance from (x,y)
                dx = x[e] - x[n]
                dy = y[e] - y[n]
                dist = np.sqrt(dx*dx + dy*dy)
                rel_distance[(e, n)] = dist

        return cls(
            segment_id=seg_id,
            num_lanes=num_lanes,
            actors=actors,
            T=T,
            speed=speed,
            yaw=yaw,
            x=x,
            y=y,
            lane_idx=lane_idx,
            present=present,
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

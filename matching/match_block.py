# osc_parser/matching/match_block.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .spec import BlockQuery

def _slice(a, t0, t1):
    """Inclusive slice [t0..t1]."""
    if a is None:
        return None
    return a[t0: t1 + 1]

def _speed_series_mps(feats, aid: str):
    """Return a speed series in m/s for actor `aid` if available."""
    lv = getattr(feats, "long_v", {}) or {}
    if aid in lv and lv[aid] is not None:
        return lv[aid]
    sp = getattr(feats, "speed", {}) or {}
    return sp.get(aid)

def _candidate_pairs(
    feats,
    Q: BlockQuery,
    pairs: Optional[List[Tuple[str, Optional[str]]]] = None,
) -> List[Tuple[str, Optional[str]]]:
    """
    Resolve which (ego,npc) bindings to try for this segment.
    Priority:
      1) Explicit 'pairs' passed by caller
      2) Q.npc_candidates (filter to actors present in this segment)
      3) Fallback: (ego, None) — for non-relational calls
    """
    ego = Q.ego
    actors = set(getattr(feats, "actors", []) or [])
    if pairs is not None:
        out = []
        for e, n in pairs:
            if e != ego:
                continue
            if n is not None and n not in actors:
                continue
            out.append((e, n))
        if out:
            return out
        # fall through if empty

    if Q.npc_candidates:
        return [(ego, n) for n in Q.npc_candidates if n in actors]

    # default: non-relational (single-actor path will pass npc=None to checks)
    return [(ego, None)]

def _candidate_ids(
    feats,
    Q: BlockQuery,
    ids: Optional[List[str]] = None,
) -> List[str]:
    """
    Resolve which single-actor ids to try.
    Priority:
      1) Explicit 'ids' from caller
      2) The query's ego (if present in this segment)
      3) Fallback: all actors in the segment
    """
    actors = list(getattr(feats, "actors", []) or [])
    if ids:
        return [a for a in ids if (not actors or a in actors)]

    ego = getattr(Q, "ego", None)
    if ego and ((not actors) or (ego in actors)):
        return [ego]

    return actors

def _determine_arity(Q: BlockQuery, ids, pairs) -> int:
    """
    Decide whether to run unary (1) or binary (2) matching.
    Rules:
      - If Q exposes .arity, honor it.
      - Else if caller provided ids -> unary.
      - Else if caller provided pairs -> if any pair has npc is not None -> binary; else unary.
      - Else if Q has npc_candidates -> binary; otherwise unary.
    """
    arity = getattr(Q, "arity", None)
    if arity in (1, 2):
        return int(arity)

    if ids is not None:
        return 1

    if pairs is not None:
        has_real_npc = any(n is not None for _, n in pairs)
        return 2 if has_real_npc else 1

    return 2 if getattr(Q, "npc_candidates", None) else 1

def match_block(
    feats,
    Q: BlockQuery,
    fps: int = 10,
    *,
    ids: Optional[List[str]] = None,                         # unary
    pairs: Optional[List[Tuple[str, Optional[str]]]] = None, # binary or (ego, None)
    max_results: int = 5000,
) -> List[Dict[str, Any]]:
    """
    Sliding-window driver over ONE segment with fast extend-to-failure for action scope.

    Returns: list of {"ego","npc","t_start","t_end"} (inclusive).
    """
    results: List[Dict[str, Any]] = []

    # Segment length
    T = int(getattr(feats, "T", 0) or 0)
    if T <= 0:
        return results

    # If speed missing but long_v (m/s) exists, synthesize speed in m/s.
    if getattr(feats, "speed", None) is None and getattr(feats, "long_v", None) is not None:
        try:
            feats.speed = {aid: np.asarray(v, dtype=float) for aid, v in feats.long_v.items()}
            setattr(feats, "speed_unit", "meters_per_second")
        except Exception:
            pass

    # Sanity: ego presence (when specified)
    ego_decl = getattr(Q, "ego", None)
    seg_actors = list(getattr(feats, "actors", []) or [])
    if ego_decl is not None and seg_actors and ego_decl not in seg_actors:
        return results

    # Decide arity
    arity = _determine_arity(Q, ids, pairs)

    # Duration & windowing
    D = max(1, int(getattr(Q, "duration_frames", 1)))
    cfg = getattr(Q, "cfg", {}) or {}

    duration_scope = str(cfg.get("duration_scope", "block")).lower()  # "block" | "action"
    allow_shorter  = bool(cfg.get("allow_shorter_end", False))
    coalesce_hits  = bool(cfg.get("coalesce_hits", duration_scope == "action"))

    # effective minimum length in frames (inclusive [t0..t1])
    min_len = 1 if (duration_scope == "block" and allow_shorter) else D

    debug = bool(cfg.get("debug_match_block", False))
    debug_checks = bool(cfg.get("debug_checks", False))

    # last start index we will consider
    last_t0 = max(0, (T - min_len) if duration_scope == "block" else (T - D))

    def _t1_bounds(t0: int) -> Tuple[int, int]:
        """Return (t1_min, t1_max) inclusive bounds for this t0."""
        if duration_scope == "action":
            # At least D frames; allow extension to segment end
            t1_min = min(T - 1, t0 + D - 1)
            t1_max = T - 1
        else:
            # Exactly D unless allow_shorter_end True at tail
            t1_min = min(T - 1, t0 + min_len - 1)
            t1_max = min(T - 1, t0 + D - 1)
        return t1_min, t1_max

    def _run_checks(ego_id, npc_id, t0, t1) -> bool:
        # Early debug probe
        if debug:
            # presence + speed quick peek
            pres_map = getattr(feats, "present", {}) or {}
            pres = pres_map.get(ego_id, None)
            pwin = _slice(pres, t0, t1)
            pres_cov = None
            if pwin is not None and len(pwin):
                pres_cov = float(np.sum(np.asarray(pwin, dtype=float) > 0.5)) / float(len(pwin))
            sp_series = _speed_series_mps(feats, ego_id)
            sp_min = sp_max = None
            if sp_series is not None:
                w = _slice(sp_series, t0, t1)
                if w is not None and len(w):
                    sp_min = float(np.min(w)); sp_max = float(np.max(w))
            if npc_id is None:
                print(f"[win] ego={ego_id} t=[{t0},{t1}] pres_cov={pres_cov} "
                      f"speed_min={sp_min} speed_max={sp_max}")
            else:
                # npc quick stats
                pres_n = (getattr(feats, "present", {}) or {}).get(npc_id, None)
                pwin_n = _slice(pres_n, t0, t1)
                cov_n = None
                if pwin_n is not None and len(pwin_n):
                    cov_n = float(np.sum(np.asarray(pwin_n, dtype=float) > 0.5)) / float(len(pwin_n))
                print(f"[win] ego={ego_id}, npc={npc_id} t=[{t0},{t1}] pres_cov_e={pres_cov} pres_cov_n={cov_n} "
                      f"ego_speed_min={sp_min} ego_speed_max={sp_max}")

        for i, check in enumerate(Q.checks):
            try:
                ok = check(feats, ego_id, npc_id, t0, t1, cfg)
            except Exception as ex:
                if debug:
                    print(f"[match_block] check exception @({t0},{t1}) ego={ego_id} npc={npc_id}: {ex}")
                ok = False
            if debug_checks:
                label = getattr(check, "_label", f"check#{i}")
                print(f"   -> {label}: {ok}")
            if not ok:
                return False
        return True

    # ---------- UNARY ----------
    if arity == 1:
        cand_ids = _candidate_ids(feats, Q, ids)

        t0 = 0
        while t0 <= last_t0:
            progressed = False
            for ego_id in cand_ids:
                if seg_actors and ego_id not in seg_actors:
                    continue

                t1_min, t1_max = _t1_bounds(t0)
                if t1_min > t1_max:
                    continue

                # First test the minimal required window
                if not _run_checks(ego_id, None, t0, t1_min):
                    # try next candidate id at same t0
                    continue

                # Greedily extend forward until failure (fast)
                t1 = t1_min
                while t1 < t1_max and _run_checks(ego_id, None, t0, t1 + 1):
                    t1 += 1

                results.append({"ego": ego_id, "npc": None, "t_start": t0, "t_end": t1})
                if len(results) >= max_results:
                    return results

                # Coalesce contiguous run by jumping past it
                t0 = t1 + 1 if coalesce_hits else (t0 + 1)
                progressed = True
                break  # don’t try other ids at this same t0

            if not progressed:
                t0 += 1

        return results

    # ---------- BINARY ----------
    cand_pairs = _candidate_pairs(feats, Q, pairs)

    t0 = 0
    while t0 <= last_t0:
        progressed = False
        for (ego_id, npc_id) in cand_pairs:
            t1_min, t1_max = _t1_bounds(t0)
            if t1_min > t1_max:
                continue

            if not _run_checks(ego_id, npc_id, t0, t1_min):
                continue

            t1 = t1_min
            while t1 < t1_max and _run_checks(ego_id, npc_id, t0, t1 + 1):
                t1 += 1

            results.append({"ego": ego_id, "npc": npc_id, "t_start": t0, "t_end": t1})
            if len(results) >= max_results:
                return results

            t0 = t1 + 1 if coalesce_hits else (t0 + 1)
            progressed = True
            break  # don’t try other pairs at same t0

        if not progressed:
            t0 += 1

    return results

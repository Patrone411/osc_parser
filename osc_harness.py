# osc_parser/osc_harness.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Iterable, Tuple
import os
import sys
import traceback
from collections.abc import Iterable as _Iterable

# --- Matching / roles / queries ------------------------------------------------
from osc_parser.matching.features import TagFeatures
from osc_parser.matching.match_single_call import match_for_binding
from osc_parser.matching.spec import build_block_query, BlockQuery
from osc_parser.matching.results.store import ResultStore
from osc_parser.matching.results.types import CallKey
from osc_parser.matching.results.collect import collect_results as _collect_results
# Role/binding utilities (your project should already provide these)
from osc_parser.role_planning import (
    make_type_to_candidates,
    roles_used_by_call,
    role_domains_from_segment,
    prefilter_domains,
    build_overlap_matrix,
    enumerate_bindings,
)

# --- Feature provider base types ----------------------------------------------
# Providers should implement .load() -> FeatureLoadResult
# See osc_parser/feature_providers/feature_provider.py
from osc_parser.feature_providers.feature_provider import FeatureLoadResult, SegmentMeta


# =============================================================================
# Harness configuration
# =============================================================================

@dataclass
class HarnessConfig:
    fps: int = 10
    # matching window semantics
    duration_scope: str = "action"       # "action" or "block"
    allow_shorter_end: bool = False
    coalesce_hits: Optional[bool] = None # None => use matcher default, True/False to override
    # data filtering
    min_lanes: int = 2
    exact_lanes: Optional[int] = None   # if set, overrides min_lanes logic
    debug_segments: bool = False        # print lane filtering summary
    # debug flags
    debug_domains: bool = False
    debug_overlap: bool = False
    debug_compile: bool = False
    debug_match: bool = False            # passed through to match_block via cfg
    debug_checks: bool = False           # passed through to match_block via cfg

    def to_query_cfg(self) -> dict:
        out = {
            "fps": float(self.fps),
            "duration_scope": self.duration_scope,
            "allow_shorter_end": self.allow_shorter_end,
        }

        # helper to add only non-None values
        def set_if(k, v):
            if v is not None:
                out[k] = v

        # existing knobs
        set_if("presence_min_coverage", getattr(self, "presence_min_cov", None))
        set_if("speed_min_coverage",    getattr(self, "speed_min_cov", None))
        set_if("speed_value_tol",       getattr(self, "eps_speed_mps", None))
        set_if("distance_tol",          getattr(self, "eps_distance_m", None))
        set_if("yaw_reach_tol",         getattr(self, "eps_yaw_rad", None))
        set_if("lane_id_convention",    getattr(self, "lane_id_convention", None))

        # gap-fillers
        set_if("anchor_slop_frames",        getattr(self, "anchor_slop_frames", None))
        set_if("lane_eq_tol",               getattr(self, "lane_eq_tol", None))
        set_if("lane_unknown_ok",           getattr(self, "lane_unknown_ok", None))
        set_if("lane_dwell_pre",            getattr(self, "lane_dwell_pre", None))
        set_if("lane_dwell_post",           getattr(self, "lane_dwell_post", None))
        set_if("change_lane_left_is_dec",   getattr(self, "change_lane_left_is_dec", None))
        set_if("vel_smooth_frames",         getattr(self, "vel_smooth_frames", None))
        set_if("min_run_frames",            getattr(self, "min_run_frames", None))
        set_if("dilate_frames",             getattr(self, "dilate_frames", None))
        set_if("erode_frames",              getattr(self, "erode_frames", None))

        if self.coalesce_hits is not None:
            out["coalesce_hits"] = bool(self.coalesce_hits)
        if self.debug_match:
            out["debug_match_block"] = True
        if self.debug_checks:
            out["debug_checks"] = True
        return out


# Optional: simple carrier for segment-level metadata
@dataclass
class SegmentMeta:
    scene_id: Optional[str] = None
    folder: Optional[str] = None
    source_uri: Optional[str] = None


# =============================================================================
# Frontend (OSC → flat calls) import shim
# =============================================================================

def _flatten_osc_file_to_calls(osc_path: str, entry_names: Iterable[str]) -> List[Dict[str, Any]]:
    """
    Import shim to use your existing OSC frontend.
    Adjust the try/except imports to match your codebase.
    The function should return a list of normalized 'flat calls' dicts.
    """
    # Try common locations in your repo; keep whichever applies in your project.
    frontend = None
    for mod, fun in [
        ("osc_parser.osc_frontend", "flatten_osc_file_to_calls"),
        ("osc_parser.frontend", "flatten_osc_file_to_calls"),
        ("osc_parser.dsl.frontend", "flatten_osc_file_to_calls"),
    ]:
        try:
            frontend = getattr(__import__(mod, fromlist=[fun]), fun)
            break
        except Exception:
            continue

    if frontend is None:
        raise ImportError(
            "Could not import a function to flatten OSC to calls. "
            "Provide flatten_osc_file_to_calls(path, entry_names)->List[Dict]."
        )
    return list(frontend(osc_path, entry_names))


# =============================================================================
# Harness
# =============================================================================

@dataclass
class OSCTestHarness:
    # Inputs
    osc_path: str
    entry_names: Iterable[str]
    cfg: HarnessConfig

    scn_constraints: Optional[Dict[str, Any]] = None

    # Optional: choose exactly one of these loading paths
    json_path: Optional[str] = None
    feature_provider: Optional[object] = None  # or: Any

    # Loaded artifacts
    feats_by_seg: Dict[str, TagFeatures] = field(default_factory=dict)
    seg_meta_by_id: Dict[str, SegmentMeta] = field(default_factory=dict)
    flattened_calls: List[Dict[str, Any]] = field(default_factory=list)

    # ---------------- Public API ----------------
    def load(self, *, load_calls: bool = True) -> None:
        """Load features and (optionally) flatten OSC to calls."""
        self._load_features()
        if load_calls and not self.flattened_calls:
            self._load_calls()

    def _iter_matches_for_call(self, call, *, max_results_per_call: int):
        """
        Yields (seg_id, binding, seg_hits) for a single call using the *same*
        domains/prefilter/overlap/bindings/matcher path as run().
        """
        roles = sorted(roles_used_by_call(call))
        ego_role = call.get("actor")
        others   = [r for r in roles if r != ego_role]
        require_pairs = [(ego_role, r) for r in others] if others else []

        for seg_id, feats in (self.feats_by_seg or {}).items():
            # --- domains (identical to run) ---
            resolver = make_type_to_candidates(feats)
            domains = role_domains_from_segment(
                self.scn_constraints if hasattr(self, "scn_constraints") else None,
                feats,
                roles=roles,
                type_to_candidates=resolver,
            )
            domains = prefilter_domains(feats, domains, min_present_frames=10)

            # Empty domain? Nothing to do for this segment
            if any(len(cands) == 0 for cands in domains.values()):
                continue

            # --- overlap + bindings (identical to run) ---
            overlap = None
            if require_pairs:
                actors = sorted({a for A in domains.values() for a in A})
                overlap = build_overlap_matrix(feats, actors, min_overlap_frames=10)

            bindings = list(
                enumerate_bindings(
                    domains,
                    distinct=True,
                    overlap_ok=overlap,
                    require_overlap_pairs=require_pairs,
                )
            )
            if not bindings:
                continue

            # --- matcher (identical to run) ---
            for binding in bindings:
                seg_hits = []
                try:
                    seg_hits = match_for_binding(
                        feats, call, binding,
                        fps=self.cfg.fps,
                        max_results=max_results_per_call,
                        cfg=self.cfg.to_query_cfg(),
                    )
                except Exception as ex:
                    print(f"[collect] ERROR seg={seg_id} binding={binding}: {ex}", file=sys.stderr)
                    continue

                yield seg_id, binding, seg_hits

    def run(self, max_results_per_call: int = 2000) -> List[Dict[str, Any]]:
        hits_all: List[Dict[str, Any]] = []
        if not self.flattened_calls or not self.feats_by_seg:
            return hits_all

        for ci, call in enumerate(self.flattened_calls):
            for seg_id, binding, seg_hits in self._iter_matches_for_call(call, max_results_per_call=max_results_per_call):
                # Attach provenance (unchanged)
                meta = self.seg_meta_by_id.get(seg_id)
                for hit in seg_hits:
                    hit["segment"] = seg_id
                    hit["roles"]   = dict(binding)
                    if meta:
                        if getattr(meta, "scene_id", None)  is not None: hit["scene_id"]  = meta.scene_id
                        if getattr(meta, "folder",   None)  is not None: hit["folder"]    = meta.folder
                        if getattr(meta, "source_uri", None) is not None: hit["source_uri"] = meta.source_uri
                hits_all.extend(seg_hits)
        return hits_all

    # ---------------- Loading helpers ----------------

    def _load_features(self) -> None:
        """Load features either via a provider or from a JSON path."""
        if self.feature_provider is not None:
            feats, meta = self._load_features_via_provider(self.feature_provider, self.cfg.min_lanes)
            # ENFORCE FILTER HERE
            feats = self._apply_lane_filter(feats)
            self.feats_by_seg = feats or {}
            self.seg_meta_by_id = meta or {}
            if not self.feats_by_seg:
                print("[harness] Provider returned no features after lane filtering.", file=sys.stderr)
            return

        if self.json_path:
            data = TagFeatures.load_json(self.json_path)
            feats_by_seg = TagFeatures.load_all_segments(data, min_lanes=None)  # let harness decide
            # ENFORCE FILTER HERE
            feats_by_seg = self._apply_lane_filter(feats_by_seg)
            self.feats_by_seg = feats_by_seg or {}
            self.seg_meta_by_id = {}
            return

        raise ValueError("OSCTestHarness: no feature source configured (provider or json_path required).")

    def load_features_only(self) -> None:
        self._load_features()

    def set_calls(self, calls: List[Dict[str, Any]]) -> None:
        self.flattened_calls = list(calls)  # shallow copy for safety
        
    def _load_features_via_provider(self, provider, min_lanes: int | None):
        """
        Ask the provider for features and merge everything into flat dicts:
        feats_by_seg:    {seg_id: TagFeatures}
        seg_meta_by_id:  {seg_id: SegmentMeta}
        Accepts either a single FeatureLoadResult or an Iterable of them.
        """
        feats_by_seg: dict[str, TagFeatures] = {}
        seg_meta_by_id: dict[str, SegmentMeta] = {}

        def _ingest(res: FeatureLoadResult):
            # Merge per-segment TagFeatures
            for seg_id, feats in (res.feats_by_seg or {}).items():
                feats_by_seg[seg_id] = feats
            # Merge metadata (later entries can overwrite earlier)
            for seg_id, meta in (res.seg_meta_by_id or {}).items():
                seg_meta_by_id[seg_id] = meta

        payload = provider.load()

        if isinstance(payload, FeatureLoadResult):
            _ingest(payload)
        elif isinstance(payload, _Iterable):
            for item in payload:
                if not isinstance(item, FeatureLoadResult):
                    raise TypeError(f"Provider yielded unexpected item: {type(item)}")
                _ingest(item)
        else:
            raise TypeError(f"Provider returned unexpected type: {type(payload)}")

        return feats_by_seg, seg_meta_by_id

    def _load_calls(self) -> None:
        """Flatten OSC file into a list of normalized calls."""
        self.flattened_calls = _flatten_osc_file_to_calls(self.osc_path, self.entry_names)

    # ---------------- Debug helper ----------------

    def _compile_query(self, call: Dict[str, Any]) -> BlockQuery:
        """
        Build a BlockQuery for debug visibility.
        Note: matching goes through match_for_binding() which will also build Q,
        so this is only for prints/inspection.
        """
        Q, _pairs_hint = build_block_query(call, fps=self.cfg.fps, cfg=self.cfg.to_query_cfg())
        
        print(
            f"[Q] seg=? actor={Q.ego} arity={Q.arity} D={Q.duration_frames} "
            f"scope={Q.cfg.get('duration_scope')} allow_shorter_end={Q.cfg.get('allow_shorter_end')}",
            flush=True
        )
        print("    referenced NPC candidates:", getattr(Q, "npc_candidates", []), flush=True)
        print("    checks:", [getattr(c, "_label", f"check#{i}") for i, c in enumerate(Q.checks)], flush=True)
        # Ensure harness-level defaults don’t clobber explicit settings:
        Q.cfg.setdefault("duration_scope", self.cfg.duration_scope)
        Q.cfg.setdefault("allow_shorter_end", self.cfg.allow_shorter_end)
        if self.cfg.coalesce_hits is not None:
            Q.cfg.setdefault("coalesce_hits", self.cfg.coalesce_hits)
        return Q

    def _apply_lane_filter(self, feats_by_seg: Dict[str, TagFeatures]) -> Dict[str, TagFeatures]:
        if feats_by_seg is None:
            return {}

        exact = getattr(self.cfg, "exact_lanes", None)
        min_req = getattr(self.cfg, "min_lanes", None)

        def _ok(f: TagFeatures) -> bool:
            n = getattr(f, "num_lanes", None)
            if n is None:
                return False
            if exact is not None:
                return int(n) == int(exact)
            if min_req is not None:
                return int(n) >= int(min_req)
            return True

        before = len(feats_by_seg)
        kept = {sid: f for sid, f in feats_by_seg.items() if _ok(f)}
        dropped = {sid: f for sid, f in feats_by_seg.items() if sid not in kept}

        if getattr(self.cfg, "debug_segments", False):
            print(f"[harness] lane filter: kept {len(kept)}/{before} segments "
                f"(exact={exact} min_lanes={min_req})")
            if dropped:
                print("[harness] dropped segments (lane count):")
                for sid, f in dropped.items():
                    print(f"   - {sid}: {getattr(f, 'num_lanes', None)}")

            # histogram of lanes among kept segments
            hist: Dict[int, int] = {}
            for f in kept.values():
                n = int(getattr(f, "num_lanes", 0) or 0)
                hist[n] = hist.get(n, 0) + 1
            if hist:
                print(f"[harness] kept lane histogram: {dict(sorted(hist.items()))}")

        return kept
    
    def collect_results(self, calls, **kw):
        return _collect_results(self, calls, **kw)
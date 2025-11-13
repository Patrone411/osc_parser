# osc_parser/matching/engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from osc_parser.osc_harness import OSCTestHarness, HarnessConfig
from osc_parser.matching.results.types import BlockSignal, ResultsStore
from osc_parser.matching.post.block_combine import combine_parallel_block, chain_serial_block

@dataclass
class MatchBatchResult:
    source_uri: str
    atomic: ResultsStore
    block_hits: Dict[str, Dict[Tuple[str, Tuple[Tuple[str, str], ...]], BlockSignal]]

class MatchEngine:
    """
    Wraps OSCTestHarness to:
      - set calls and scenario constraints
      - collect atomic signals per call/binding
      - combine inside-block signals (parallel/serial)
    """
    def __init__(self, cfg: HarnessConfig, scn_constraints: dict, calls: List[dict]):
        self.cfg = cfg
        self.calls = calls
        self.h = OSCTestHarness(
            osc_path="",
            entry_names={"top"},
            cfg=cfg,
            feature_provider=None,
        )
        # step 2: restrict role domains with scenario typing
        self.h.scn_constraints = scn_constraints
        self.h.set_calls(calls)

    def process_loaded_features(self) -> MatchBatchResult:
        # assumes h.feats_by_seg and h.seg_meta_by_id are set externally per pickle
        store = self.h.collect_results(self.calls)
        T_by_seg = {seg: feats.T for seg, feats in (self.h.feats_by_seg or {}).items()}

        block_hits: Dict[str, Dict[Tuple[str, Tuple[Tuple[str, str], ...]], BlockSignal]] = {}
        for label, plan in getattr(self.h, "block_plans", {}).items():
            # (h.block_plans) is optional; callers can pass their own plans if preferred
            pass

        # Usually you’ll pass explicit plans computed from the OSC program.
        # Provide a helper instead:
        return MatchBatchResult(source_uri="<unknown>", atomic=store, block_hits={})

    def process_loaded_features_with_plans(
        self,
        plans: Dict[str, object],
        source_uri: str = "<unknown>",
    ) -> MatchBatchResult:
        store = self.h.collect_results(self.calls)
        T_by_seg = {seg: feats.T for seg, feats in (self.h.feats_by_seg or {}).items()}
        block_hits: Dict[str, Dict[Tuple[str, Tuple[Tuple[str, str], ...]], BlockSignal]] = {}

        for label, plan in plans.items():
            if plan.type == "parallel":
                block_hits[label] = combine_parallel_block(plan, self.calls, store, T_by_seg)
            elif plan.type == "serial":
                block_hits[label] = chain_serial_block(plan, self.calls, store, T_by_seg, allow_overlap=True)
            else:
                block_hits[label] = {}

        return MatchBatchResult(source_uri=source_uri, atomic=store, block_hits=block_hits)

    # helpers to swap features per pickle (matches your current usage)
    def set_features(self, feats_by_seg: Dict, seg_meta_by_id: Dict) -> None:
        self.h.feats_by_seg = dict(feats_by_seg)
        self.h.seg_meta_by_id = dict(seg_meta_by_id)

    def clear_features(self) -> None:
        self.h.feats_by_seg.clear()
        self.h.seg_meta_by_id.clear()

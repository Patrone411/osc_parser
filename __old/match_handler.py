"""
Matching module for constraints derived from OSC IR.

Provides:
- ConstraintMatcher: loads tag feature JSON and matches normalized calls

Usage
-----
from .osc_matching import ConstraintMatcher

matcher = ConstraintMatcher(
    data_path="./osc_parser/data/4680e1fb10c57daa_tags.json",
    fps=10,
)

# If you have constraints from the parser (constraints_by_scenario):
hits_first = matcher.match_first_call(constraints_by_scenario, scenario="top")
print(len(hits_first))

# Or match all calls:
all_hits = matcher.match_all_calls(constraints_by_scenario, scenario="top")
for idx, hits in all_hits.items():
    print(idx, len(hits))
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional
import json

from osc_parser.matching.features import TagFeatures
from osc_parser.matching.match_single_call import match_single_call_across_segments

__all__ = ["ConstraintMatcher", "DEFAULT_MATCH_CFG"]


DEFAULT_MATCH_CFG: Dict[str, Any] = {
    "presence_min_coverage": 0.9,
    "speed_min_coverage": 0.9,
    "speed_value_tol": 0.1,   # m/s
    "distance_tol": 2.0,      # m
    "change_speed_tol": 0.3,  # m/s
    "lateral_allow_missing": True,
    "allow_shorter_end": True,
    "during_mode": "coverage",  # or "all" for universal "during"
    "during_max_false": 0,       # only used when during_mode == "all"
    "space_gap_tol": 0.2,
}


class ConstraintMatcher:
    """Runs matching against tag feature data for calls derived from constraints.

    Parameters
    ----------
    data_path : str
        Path to JSON file with tag features.
    fps : int
        Frame rate to pass into matching.
    min_lanes : int
        Minimum number of lanes required when loading segments.
    cfg : Optional[Mapping[str, Any]]
        Matching configuration; merges over defaults.
    """

    def __init__(
        self,
        *,
        data_path: str,
        fps: int = 10,
        min_lanes: int = 2,
        cfg: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.data_path = data_path
        self.fps = fps
        self.min_lanes = min_lanes
        with open(self.data_path, "r") as f:
            data = json.load(f)
        self.feats_by_seg = TagFeatures.load_all_segments(data, min_lanes=self.min_lanes)
        self.cfg: Dict[str, Any] = {**DEFAULT_MATCH_CFG, **(dict(cfg) if cfg else {})}

    def match_call(
        self,
        call: Mapping[str, Any],
        *,
        max_results_per_seg: int = 5000,
    ) -> List[Any]:
        """Match a single normalized call across all segments."""
        hits = match_single_call_across_segments(
            self.feats_by_seg,
            call,
            fps=self.fps,
            cfg=self.cfg,
            max_results_per_seg=max_results_per_seg,
        )
        return list(hits)

    def match_first_call(
        self,
        constraints_by_scenario: Mapping[str, Any],
        *,
        scenario: str = "top",
        max_results_per_seg: int = 5000,
    ) -> List[Any]:
        """Convenience: grab the first call of a scenario and match it."""
        if scenario not in constraints_by_scenario:
            raise KeyError(
                f"Scenario '{scenario}' not found. Available: {list(constraints_by_scenario.keys())}"
            )
        norm = constraints_by_scenario[scenario]
        calls = norm.get("calls_flat", [])
        if not calls:
            raise ValueError(f"No calls_flat found for scenario '{scenario}'.")
        return self.match_call(calls[0], max_results_per_seg=max_results_per_seg)

    def match_all_calls(
        self,
        constraints_by_scenario: Mapping[str, Any],
        *,
        scenario: str = "top",
        max_results_per_seg: int = 5000,
    ) -> Dict[int, List[Any]]:
        """Match every call in a scenario; returns {index: hits}."""
        if scenario not in constraints_by_scenario:
            raise KeyError(
                f"Scenario '{scenario}' not found. Available: {list(constraints_by_scenario.keys())}"
            )
        norm = constraints_by_scenario[scenario]
        calls = norm.get("calls_flat", [])
        results: Dict[int, List[Any]] = {}
        for i, call in enumerate(calls):
            results[i] = self.match_call(call, max_results_per_seg=max_results_per_seg)
        return results


if __name__ == "__main__":
    # Minimal smoke test to verify imports; adjust paths to your project.
    import json
    from pathlib import Path

    sample_data = Path("./osc_parser/data/4680e1fb10c57daa_tags.json")
    if sample_data.exists():
        cm = ConstraintMatcher(data_path=str(sample_data))
        # A real test requires constraints; this file just verifies construction.
        print("ConstraintMatcher ready; provide constraints to run matches.")
    else:
        print("Sample data JSON not found; adjust path for testing.")

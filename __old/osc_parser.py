"""
Parsing module for OpenSCENARIO (.osc) files.

Provides:
- ParseResult: dataclass capturing lowered IR, constraints, validator, and validation status
- OSCParser: class that compiles an OSC file to IR, validates semantics, and returns constraints

Usage
-----
from .osc_parsing import OSCParser

parser = OSCParser(
    osc_path="osc_parser/osc/relative.osc",
    entry_names={"top"},
)
result = parser.run()
print("Valid:", result.valid)
print("Scenarios:", list(result.scenarios.keys()))
print("Constraints:", list(result.constraints_by_scenario.keys()))

# Access constraints for a specific scenario
constraints_top = result.constraints_for("top")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple
import logging
import json

# Project imports (relative to package containing MiniOSC2ScenarioConfig)
from . import MiniOSC2ScenarioConfig, ConfigInit
from .pytree.ir_lowering import IRLowering
from .pytree.print_tree import print_ir  # noqa: F401 (useful for debugging)

from osc_parser.matching.constraints import constraints_from_ir

from osc_parser.srunner.osc2.semantics.registry import SemanticsRegistry
from osc_parser.srunner.osc2.semantics.validator import SemanticValidator, infer_type
from osc_parser.srunner.osc2.semantics.ir_adapter import validate_from_ir
from osc_parser.srunner.osc2.ast_manager.post_checks import (  # noqa: F401
    check_namespace_collisions,
    global_scope_from_ast_tree,
)

__all__ = ["OSCParser", "ParseResult"]

logger = logging.getLogger(__name__)

def block_outline(block):
    """Keep only type + name (label) recursively."""
    return {
        "type": block.get("type"),
        "name": block.get("label"),          # rename 'label'→'name' for clarity
        "children": [block_outline(c) for c in block.get("children", [])],
    }

def scenario_outline(constraints_by_scenario, scenario: str):
    """Outline for a given scenario → list of top-level block outlines."""
    sc = constraints_by_scenario[scenario]
    return [block_outline(b) for b in sc.get("blocks", [])]

@dataclass
class ParseResult:
    """Return object for OSCParser.run()."""

    scenarios: Mapping[str, Any]
    constraints_by_scenario: Mapping[str, Any]
    validator: SemanticValidator
    valid: bool
    errors: List[str]

    def constraints_for(self, scenario: str) -> Mapping[str, Any]:
        if scenario not in self.constraints_by_scenario:
            raise KeyError(
                f"Scenario '{scenario}' not found in constraints. Available: {list(self.constraints_by_scenario.keys())}"
            )
        return self.constraints_by_scenario[scenario]


class OSCParser:
    """Parses an OSC file, lowers to IR, and validates semantics.

    Parameters
    ----------
    osc_path : str
        Path to the .osc file.
    registry_path : str
        Path to the OSC semantics registry JSON.
    entry_names : Optional[Set[str]]
        Which entry scenarios to lower; set() means all. Default {"top"}.
    type_hook : Callable
        Function used by the validator for type inference.
    debug_types : bool
        Passes debug flag to the validator.
    """

    def __init__(
        self,
        osc_path: str,
        *,
        registry_path: str = "osc_parser/srunner/osc2/semantics/osc_semantics_registry.json",
        entry_names: Optional[Set[str]] = frozenset({"top"}),
        type_hook: Callable = infer_type,
        debug_types: bool = True,
    ) -> None:
        self.osc_path = osc_path
        self.registry_path = registry_path
        self.entry_names = set(entry_names) if entry_names is not None else set()
        self.type_hook = type_hook
        self.debug_types = debug_types

    def _build_config(self) -> Tuple[MiniOSC2ScenarioConfig, ConfigInit]:
        config = MiniOSC2ScenarioConfig(self.osc_path)
        pass1 = ConfigInit(config)
        pass1.visit(config.ast_tree)  # registers symbols/units/vars/actors
        return config, pass1

    def _lower_to_ir(self, config: MiniOSC2ScenarioConfig, pass1: ConfigInit):
        lowered = IRLowering(config, actor_registry=pass1.actor_registry, entry_names=self.entry_names).lower(config.ast_tree)
        # IRLowering.lower() returns List[ScenarioNode]; normalize to a mapping
        if isinstance(lowered, dict):
            return lowered
        return {scn.name: scn for scn in lowered}

    def _make_validator(self) -> Tuple[SemanticsRegistry, SemanticValidator]:
        sem_registry = SemanticsRegistry.from_file(self.registry_path)
        validator = SemanticValidator(sem_registry, type_of_expr=self.type_hook, debug_types=self.debug_types)
        return sem_registry, validator

    def run(self) -> ParseResult:
        config, pass1 = self._build_config()
        scenarios_any = self._lower_to_ir(config, pass1)   # could be dict or list
        _, validator = self._make_validator()

        # --- normalize shapes ---
        if isinstance(scenarios_any, dict):
            scenarios_list = list(scenarios_any.values())   # for validators/builders
            scenarios_map  = scenarios_any                  # for ParseResult.scripts
        else:
            scenarios_list = scenarios_any
            scenarios_map  = {scn.name: scn for scn in scenarios_any}

        errors = []
        valid = True
        try:
            validate_from_ir(scenarios_list, validator)
        except Exception as exc:
            valid = False
            errors.append(f"Semantic validation failed: {exc}")

        # IMPORTANT: pass the LIST here, not the dict
        constraints = constraints_from_ir(scenarios_list)

        return ParseResult(
            scenarios=scenarios_map,
            constraints_by_scenario=constraints,
            validator=validator,
            valid=valid,
            errors=errors,
    )


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    parser = OSCParser(
        osc_path="osc_parser/osc/get_ahead.osc",
        #osc_path="osc_parser/osc/relative.osc",
        #entry_names={"top"},
    )
    res = parser.run()
    print("Semantic validation completed.")
    print("Valid:", res.valid)
    if res.errors:
        print("Errors:")
        for e in res.errors:
            print(" -", e)
    print("Constraints by scenario:", list(res.constraints_by_scenario.keys()))
    for item in res.constraints_by_scenario["top"]["calls_flat"]:
        print(item)
    #get scenario layout / nested structure
    cs = res.constraints_by_scenario
    outline = scenario_outline(cs, "top")
    print(json.dumps(outline, indent=2))

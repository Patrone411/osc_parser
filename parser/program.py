# osc_parser/parser/program.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from osc_parser import MiniOSC2ScenarioConfig, ConfigInit
from osc_parser.pytree.ir_lowering import IRLowering
from osc_parser.srunner.osc2.semantics.registry import SemanticsRegistry
from osc_parser.srunner.osc2.semantics.validator import SemanticValidator, infer_type
from osc_parser.srunner.osc2.semantics.ir_adapter import validate_from_ir, get_min_lanes
from osc_parser.matching.constraints import constraints_from_ir
from osc_parser.matching.post.plan import build_block_plans

@dataclass
class OSCProgram:
    osc_path: str
    entry_names: Set[str] = None
    registry_path: str = "osc_parser/srunner/osc2/semantics/osc_semantics_registry.json"
    debug_types: bool = True

    # outputs
    scenarios_list: List = None          # List[ScenarioNode]
    constraints_by_scenario: Dict = None
    calls: List[dict] = None
    min_lanes: int = 0
    plans: Dict[str, object] = None      # label -> BlockPlan
    validator: Optional[SemanticValidator] = None

    def compile(self) -> "OSCProgram":
        entries = self.entry_names or {"top"}

        config = MiniOSC2ScenarioConfig(self.osc_path)
        pass1 = ConfigInit(config)
        pass1.visit(config.ast_tree)

        lower = IRLowering(config, actor_registry=pass1.actor_registry, entry_names=set(entries))
        scn_map = lower.lower(config.ast_tree)
        self.scenarios_list = list(scn_map.values())

        sem_registry = SemanticsRegistry.from_file(self.registry_path)
        self.validator = SemanticValidator(sem_registry, type_of_expr=infer_type, debug_types=self.debug_types)
        validate_from_ir(self.scenarios_list, self.validator)

        self.constraints_by_scenario = constraints_from_ir(self.scenarios_list)
        entry = next(iter(self.constraints_by_scenario))
        self.calls = self.constraints_by_scenario[entry]["calls_flat"]

        self.min_lanes = get_min_lanes(self.scenarios_list, scenario_name=entry, default=0)
        self.plans = build_block_plans(self.calls)
        return self

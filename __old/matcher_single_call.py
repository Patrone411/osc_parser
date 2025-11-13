from osc_parser.matching.features import TagFeatures
from osc_parser.matching.constraints import constraints_from_ir
from osc_parser.matching.match_single_call import match_single_call
from osc_parser.matching.match_single_call import match_single_call_across_segments

from . import MiniOSC2ScenarioConfig, ConfigInit, print_pytree, pytree_to_actor_constraints
from .pytree.ir_lowering import IRLowering
from .pytree.print_tree import print_ir
import json
from osc_parser.srunner.osc2.semantics.registry import SemanticsRegistry
from osc_parser.srunner.osc2.semantics.validator import SemanticValidator, infer_type
from osc_parser.srunner.osc2.semantics.ir_adapter import validate_from_ir
from osc_parser.srunner.osc2.ast_manager.post_checks import check_namespace_collisions, global_scope_from_ast_tree


PREFIX = "osc_parser/osc/"
osc_file = "test_actions.osc"
osc_file = "relative.osc"


config = MiniOSC2ScenarioConfig(PREFIX + osc_file)
# PASS 1: symbols/units/vars/actor-registry/instances
pass1 = ConfigInit(config)
pass1.visit(config.ast_tree)

# PASS 2: lower to IR (choose entry scenarios, e.g., {"top"} or set() for all)
lower = IRLowering(config, actor_registry=pass1.actor_registry, entry_names={"top"})
scenarios = lower.lower(config.ast_tree)

# Load registry + validator (optionally pass a better type hook)
REGISTRY_PATH = "osc_parser/srunner/osc2/semantics/osc_semantics_registry.json"
sem_registry = SemanticsRegistry.from_file(REGISTRY_PATH)

#validator    = SemanticValidator(sem_registry)  # or with a type hook
validator = SemanticValidator(sem_registry, type_of_expr=infer_type, debug_types=True)
# Validate semantics using the IR
validate_from_ir(scenarios, validator)
print("Semantic validation completed.")

constraints_by_scenario = constraints_from_ir(scenarios)
print(constraints_by_scenario)
# Example filter usage:
top = constraints_by_scenario["top"]

# 3) pick a normalized call from constraints_from_ir(...)
norm_all = constraints_from_ir(scenarios)
norm = norm_all["top"]

call = norm["calls_flat"][0]
print("call: ", call)

cfg = {
    "presence_min_coverage": 0.9,
    "speed_min_coverage": 0.9,
    "speed_value_tol": 0.1,         # m/s
    "distance_tol": 2.0,            # m
    "change_speed_tol": 0.3,        # m/s
    "lateral_allow_missing": True,
    "allow_shorter_end": True,
    "during_mode": "coverage",      # or "all" for universal “during”
    "during_max_false": 0,          # only used when during_mode == "all"
    "space_gap_tol": 0.2,
}

PATH = "./osc_parser/data/4680e1fb10c57daa_tags.json"  # adjust if needed

with open(PATH, "r") as f:
        data = json.load(f)

feats_by_seg = TagFeatures.load_all_segments(data, min_lanes=2)
# 5) run across segments
hits = match_single_call_across_segments(
    feats_by_seg,
    call,
    fps=10,
    cfg=cfg,
    max_results_per_seg=5000,
)

print(f"Total hits: {len(hits)}")
for h in hits[:10]:
    print(h)
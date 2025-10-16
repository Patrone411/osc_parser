from osc_parser.matching.features import TagFeatures
from osc_parser.constraints import constraints_from_ir
from osc_parser.matching.match_single_call import spec_from_call, match_single_call

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

# 3) compile to a spec
fps = 10
cfg = {
    "default_window_s": 5.0,
    "allow_shorter_end": True,
    "presence_min_coverage": 0.9,
    "speed_min_coverage": 0.9,
    "speed_value_tol": 0.1,   # m/s
    "distance_tol": 2.0,      # meters
}
spec = spec_from_call(call, fps=fps, cfg=cfg)
print (spec)

# 1) load your stitched/tag json
data = TagFeatures.load_json("./osc_parser/data/tag_result.json")

# 2) filter segments by map constraint (e.g., min lanes = 2)
feats_by_seg = TagFeatures.load_all_segments(data, min_lanes=2)
"""for seg_id, feats in feats_by_seg.items():
    print(seg_id)
    print(feats)"""

hits = []
for seg_id, feats in feats_by_seg.items():
    seg_hits = match_single_call(feats, spec, fps=fps)
    for h in seg_hits:
        h["segment"] = seg_id
    hits.extend(seg_hits)

print(f"found {len(hits)} matches")
print(hits[:10])


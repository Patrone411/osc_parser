import json

from osc_parser.constraints import constraints_from_ir
from osc_parser.matching.adapters import extract_serial_of_parallel
from osc_parser.matching.multi_call import match_parallel_group

from . import MiniOSC2ScenarioConfig, ConfigInit, print_pytree, pytree_to_actor_constraints
from .pytree.ir_lowering import IRLowering

from osc_parser.srunner.osc2.semantics.registry import SemanticsRegistry
from osc_parser.srunner.osc2.semantics.validator import SemanticValidator, infer_type
from osc_parser.srunner.osc2.semantics.ir_adapter import validate_from_ir

from osc_parser.matching.features import TagFeatures

def match_serial_across_segments(feats_by_seg, program, fps=10, cfg=None, **kw):
    out = []
    for seg_id, feats in feats_by_seg.items():
        seqs = match_serial_program(feats, program, fps=fps, cfg=cfg or {}, **kw)
        for s in seqs:
            s["segment"] = seg_id
        out.extend(seqs)
    return out

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


constraints = constraints_from_ir(scenarios)
scn = constraints["top"]

parallel_block = next(
    b for b in scn["blocks"][0]["children"] if b["type"] == "parallel" and b["label"] == "get_ahead"
)



DATA_PATH = "./osc_parser/data/4680e1fb10c57daa_tags.json"  # adjust if needed

with open(DATA_PATH, "r") as f:
        data = json.load(f)

feats_by_seg = TagFeatures.load_all_segments(data, min_lanes=2)

# 3) Run the parallel matcher on one segment
hits = match_parallel_group(feats, parallel_block, fps=10, cfg={})

for h in hits:
    print(h)

DATA_PATH = "./osc_parser/data/4680e1fb10c57daa_tags.json"  # adjust if needed

with open(DATA_PATH, "r") as f:
        data = json.load(f)

feats_by_seg = TagFeatures.load_all_segments(data, min_lanes=2)
for seg_id, feats in feats_by_seg.items():
    hits = match_parallel_group(feats, parallel_block, fps=10, cfg={})
    for h in hits:
        print(h)


import json

from osc_parser.constraints import constraints_from_ir
from osc_parser.matching.adapters import extract_serial_of_parallel
from osc_parser.matching.multi_call import match_serial_program

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



# 1) Build constraints from your parsed IR (list of ScenarioNode)
scn_name = scenarios[0].name
constraints = constraints_from_ir(scenarios)
scn_dict = constraints[scn_name]

# 2) Extract the ordered list of parallel groups from the serial block
program = extract_serial_of_parallel(scn_dict, serial_label=None)  # or pass a label

DATA_PATH = "./osc_parser/data/4680e1fb10c57daa_tags.json"  # adjust if needed

with open(DATA_PATH, "r") as f:
        data = json.load(f)

feats_by_seg = TagFeatures.load_all_segments(data, min_lanes=2)

sequences = match_serial_across_segments(
    feats_by_seg, program, fps=10, cfg={}, contiguous=True
)


#single segment feat
"""# 3) Run the matcher (one segment)
fps = 10
cfg = {}  # optional tolerances
sequences = match_serial_program(
    feats,                 # TagFeatures for this segment
    program,               # [{"type":"parallel","duration":{...},"calls":[...]} ...]
    fps=fps,
    cfg=cfg,
    contiguous=True,       # require back-to-back groups; set False to allow gaps
    # min_gap=0, max_gap=None,
)"""

# 4) Inspect results
for i, seq in enumerate(sequences):
    print(f"Sequence #{i}:")
    for j, grp in enumerate(seq["groups"]):
        print(f"  Group {j}: t=[{grp['t_start']},{grp['t_end']}], ego={grp['ego']}, npc={grp['npc']}")

from . import MiniOSC2ScenarioConfig, ConfigInit, print_pytree, pytree_to_actor_constraints
from .pytree.ir_lowering import IRLowering
from .pytree.print_tree import print_ir

PREFIX = "osc_parser/osc/"
osc_file = "test.osc"



config = MiniOSC2ScenarioConfig(PREFIX + osc_file)

# PASS 1: symbols/units/vars/actor-registry/instances
pass1 = ConfigInit(config)
pass1.visit(config.ast_tree)



def _name_of(x):
    # Works for Identifier nodes or plain strings
    return getattr(x, "name", x) if not isinstance(x, str) else x

def check_namespace_collisions(ast_root, pass1):
    from osc_parser.srunner.osc2.ast_manager import ast_node

    # Buckets we care about (seed with what Pass1 already collected)
    buckets = {
        "physical_type": set(getattr(pass1.father_ins, "physical_dict", {}).keys()),
        "unit":          set(getattr(pass1.father_ins, "unit_dict", {}).keys()),
        "struct":        set(getattr(pass1.father_ins, "struct_declaration", {}).keys()),
        "actor":         set(pass1.actor_registry.keys()),
        "enum":          set(),
        "action":        set(),
        "modifier":      set(),
        # Optional: scenario-level variables (to detect shadowing)
        "variable":      set(getattr(pass1.father_ins, "variables", {}).keys()),
    }

    # Collect anything Pass1 didn’t store explicitly by scanning the AST once
    for c in ast_root.get_children():
        if isinstance(c, ast_node.EnumDeclaration):
            buckets["enum"].add(_name_of(getattr(c, "enum_name", None)))
        elif isinstance(c, ast_node.StructDeclaration):
            buckets["struct"].add(_name_of(getattr(c, "struct_name", None)))
        elif isinstance(c, ast_node.ActorDeclaration):
            buckets["actor"].add(_name_of(getattr(c, "actor_name", None)))
        elif isinstance(c, (ast_node.ActionDeclaration, getattr(ast_node, "ActionInherts", tuple()))):
            # most grammars use qualified_behavior_name for actions
            n = getattr(c, "qualified_behavior_name", None)
            if n: buckets["action"].add(_name_of(n))
        elif isinstance(c, ast_node.ModifierDeclaration):
            buckets["modifier"].add(_name_of(getattr(c, "modifier_name", None)))

    # Build name -> kinds map
    name_to_kinds = {}
    for kind, names in buckets.items():
        for n in names:
            if n is None:
                continue
            s = str(n)
            name_to_kinds.setdefault(s, set()).add(kind)

    # Anything that appears in more than one bucket is a potential collision
    conflicts = {n: kinds for n, kinds in name_to_kinds.items() if len(kinds) > 1}

    if conflicts:
        print("\n[Symbol collisions detected]")
        for n, kinds in sorted(conflicts.items()):
            print(f"  '{n}' appears in: {', '.join(sorted(kinds))}")
        print()  # blank line
    else:
        print("\n[No cross-namespace name collisions found]\n")


check_namespace_collisions(config.ast_tree, pass1)

# PASS 2: lower to IR (choose entry scenarios, e.g., {"top"} or set() for all)
lower = IRLowering(config, actor_registry=pass1.actor_registry, entry_names={"top"})
scenarios = lower.lower(config.ast_tree)
print_ir(scenarios)





"""
config = MiniOSC2ScenarioConfig(PREFIX + osc_file)
for scen in config.pytree.values():
    print(render_scenario(scen))"""

"""visitor = ConfigInit(config)
visitor.visit(config.ast_tree)
py_tree = visitor.pytree

print("tree:")
print_pytree(py_tree)

map_constraints = {}
map_constraints["min_lanes"] = config.path.min_driving_lanes
print("map_constraints: ", map_constraints)

actor_constraints = pytree_to_actor_constraints(py_tree)
print("actor constraints: ", actor_constraints)"""

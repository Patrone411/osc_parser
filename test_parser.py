from . import MiniOSC2ScenarioConfig, ConfigInit, print_pytree, pytree_to_actor_constraints

PREFIX = "osc_parser/osc/"

osc_file = "test.osc"

config = MiniOSC2ScenarioConfig(PREFIX + osc_file)
visitor = ConfigInit(config)
visitor.visit(config.ast_tree)
py_tree = visitor.pytree

print("tree:")
print_pytree(py_tree)

map_constraints = {}
map_constraints["min_lanes"] = config.path.min_driving_lanes
print("map_constraints: ", map_constraints)

actor_constraints = pytree_to_actor_constraints(py_tree)
print("actor constraints: ", actor_constraints)

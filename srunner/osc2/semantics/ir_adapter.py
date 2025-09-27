# srunner/osc2/semantics/ir_adapter.py
from typing import Dict, List
from osc_parser.pytree.pytree import ScenarioNode, SerialBlock, ParallelBlock, ActionCall, ModifierCall
from .validator import ActionCall as VActionCall, ModifierAttach as VModifierAttach, ArgValue

def _to_argdict(args_list):
    """Convert IR ActionCall.args (list of scalars or (k,v)) into dict."""
    named = {}
    for a in args_list:
        if isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], str):
            named[a[0]] = a[1]
        # positional action args not used in your registry for now
    return named

def _mod_to_named(m: ModifierCall) -> Dict[str, object]:
    """Convert IR ModifierCall args into a named dict and map positionals to canonical names."""
    named = {}
    pos = []
    for a in m.args:
        if isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], str):
            named[a[0]] = a[1]
        else:
            pos.append(a)

    # Map the first positional to the canonical parameter where relevant
    if pos:
        if m.name in ("position", "lateral"):
            named.setdefault("distance", pos[0])
        elif m.name == "speed":
            named.setdefault("speed", pos[0])
        # add other mappings here as needed

    # Merge kwargs the IR already normalized
    for k, v in (m.kwargs or {}).items():
        named[k] = v
    return named

def validate_from_ir(scenarios: List[ScenarioNode], validator) -> None:
    for scn in scenarios:
        # Scenario-local actor type map
        actor_types = {name: inst.type for name, inst in scn.actors.items()}

        def walk_block(block):
            for ch in getattr(block, "children", []):
                if isinstance(ch, ActionCall):
                    inv_name = ch.actor
                    inv_type = actor_types.get(inv_name)
                    if not inv_type:
                        print(f"Unknown actor '{inv_name}' (not found in scenario '{scn.name}').")
                        continue  # or raise

                    # Action args -> ArgValue dict
                    a_named = _to_argdict(ch.args)
                    v_args = {k: ArgValue(k, v) for k, v in a_named.items()}

                    # Modifiers -> ModifierAttach with named args
                    v_mods = []
                    for m in ch.modifiers:
                        m_named = _mod_to_named(m)
                        v_mods.append(
                            VModifierAttach(
                                name=m.name,
                                args={k: ArgValue(k, v) for k, v in m_named.items()},
                                token=getattr(m, "token", None),
                            )
                        )

                    vcall = VActionCall(
                        invoker_name=inv_name,
                        invoker_type=inv_type,
                        method_name=ch.action,
                        args=v_args,
                        modifiers=v_mods,
                        token=getattr(ch, "token", None),
                    )
                    validator.validate_action_call(vcall)
                else:
                    # nested blocks
                    if isinstance(ch, (SerialBlock, ParallelBlock)):
                        walk_block(ch)

        for blk in scn.blocks:
            walk_block(blk)

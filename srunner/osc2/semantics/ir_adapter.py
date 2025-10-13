# srunner/osc2/semantics/ir_adapter.py
from typing import Dict, List
from osc_parser.pytree.pytree import ScenarioNode, SerialBlock, ParallelBlock, ActionCall, ModifierCall
from .validator import ActionCall as VActionCall, ModifierAttach as VModifierAttach, ArgValue
from osc_parser.srunner.osc2_dm.physical_types import Physical
from osc_parser.config_init import _GenericPath

# Map each modifier name to the canonical param its *first positional* represents
_POS0_PARAM = {
    "speed": "speed",
    "change_speed": "speed",
    "acceleration": "acceleration",
    "position": "distance",
    "distance": "distance",
    "lateral": "distance",
    "yaw": "angle",
    "lane": "lane",
    "change_lane": "lane",
}

def _first_positional_param_for(mod_name: str) -> str:
    return _POS0_PARAM.get(mod_name)


def _to_argdict(args_list):
    """Convert IR ActionCall.args (list of scalars or (k,v)) into dict."""
    named = {}
    pos = []
    for a in args_list:
        if isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], str):
            named[a[0]] = a[1]
        else:
            pos.append(a)

    if pos and "duration" not in named:
        named["duration"] = pos[0]

    return named

def _mod_to_named(m: ModifierCall) -> Dict[str, object]:
    named, pos = {}, []
    for a in m.args:
        if isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], str):
            named[a[0]] = a[1]
        else:
            pos.append(a)

    if pos:
        if m.name in ("position", "lateral", "distance"):
            named.setdefault("distance", pos[0])
        elif m.name in ("speed", "change_speed"):
            named.setdefault("speed", pos[0])
        elif m.name in ("acceleration", "change_acceleration"):
            named.setdefault("acceleration", pos[0])
        elif m.name == "along":
            named.setdefault("route", pos[0])
        elif m.name == "yaw":
            named.setdefault("angle", pos[0])
        elif m.name == "lane":
            named.setdefault("lane", pos[0])
        elif m.name == "change_lane":
            named.setdefault("lane", pos[0])
            if len(pos) >= 2:
                named.setdefault("side", pos[1])


    for k, v in (m.kwargs or {}).items():
        named[k] = v
    return named

def _infer_type_for_value(v, actor_types: dict) -> str:
    # Physicals -> underlying physical dimension
    if isinstance(v, Physical):
        try:
            return v.unit.physical.name  # e.g., "length", "time", "speed", "acceleration"
        except Exception:
            return None
    # GenericPath should be treated as 'route' (for along(route))
    if isinstance(v, _GenericPath):
        return "route"
    # Map common string literals to semantic types
    if isinstance(v, str):
        # actor name -> physical_object
        if v in actor_types:
            return "physical_object"
        low = v.lower()
        if low in ("start", "end"):         # used by many modifiers
            return "at"
        if low in ("left", "right"):        # used by lateral/lane side params
            return "side_left_right"
        # leave other strings untyped
        return None
    # Plain numerics/bools (only used where appropriate)
    if isinstance(v, bool):  return "bool"
    if isinstance(v, int):   return "int"
    if isinstance(v, float): return "float"
    return None

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
                    v_args = {k: ArgValue(k, v, type_name=_infer_type_for_value(v, actor_types))
                            for k, v in a_named.items()}

                    # Modifiers -> ModifierAttach with named args
                    v_mods = []
                    for m in ch.modifiers:
                        m_named = _mod_to_named(m)
                        v_mods.append(
                            VModifierAttach(
                                name=m.name,
                                args={k: ArgValue(k, v, type_name=_infer_type_for_value(v, actor_types))
                                    for k, v in m_named.items()},
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
                    # ⬇️ capture the result
                    result = validator.validate_action_call(vcall)

                    # modifiers: keep 1–1 by order
                    for (mod_idx, (m_name, _variant, m_args)) in enumerate(result.resolved_modifiers):
                        if mod_idx >= len(ch.modifiers):  # safety
                            break
                        irm = ch.modifiers[mod_idx]
                        if irm.name != m_name:
                            # fallback: try to find same-name modifier not yet processed
                            try:
                                irm = next(mm for mm in ch.modifiers if mm.name == m_name)
                            except StopIteration:
                                irm = ch.modifiers[mod_idx]

                        # Ensure args/kwargs containers exist
                        if getattr(irm, "args", None) is None:
                            irm.args = []
                        if getattr(irm, "kwargs", None) is None:
                            irm.kwargs = {}

                        # Handle the “pos0” canonical parameter to avoid duplicates
                        pos0_name = _first_positional_param_for(irm.name)
                        if pos0_name and pos0_name in m_args:
                            # If we already have a positional present, don't add named duplicate
                            if len(irm.args) > 0:
                                # drop it from write-back so we don't add it as a kwarg
                                m_args = {k: v for k, v in m_args.items() if k != pos0_name}
                            else:
                                # No positionals -> move this canonical param into positionals for pretty print
                                irm.args.insert(0, m_args[pos0_name])
                                m_args = {k: v for k, v in m_args.items() if k != pos0_name}

                        # Now add the rest of the (non-pos0) resolved args as kwargs,
                        # but NOT overwriting anything the user already supplied.
                        for k, v in m_args.items():
                            if k not in irm.kwargs:
                                irm.kwargs[k] = v


                else:
                    if isinstance(ch, (SerialBlock, ParallelBlock)):
                        walk_block(ch)

        for blk in scn.blocks:
            walk_block(blk)

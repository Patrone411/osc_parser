from typing import Any, List
from .pytree import ScenarioNode, SerialBlock, ParallelBlock, ActionCall, ModifierCall
from ..srunner.osc2_dm.physical_types import Physical, Range

def _unit_abbr(u) -> str:
    # Try common abbreviations; fall back to unit's name
    name = getattr(u, "unit_name", None) or getattr(u, "name", None) or str(u)
    table = {
        "meter": "m", "second": "s", "kilometer_per_hour": "kph",
        "meter_per_second": "mps", "mile_per_hour": "mph",
        "degree": "deg", "radian": "rad",
        "millisecond": "ms", "minute": "min", "hour": "h",
        "millimeter": "mm", "centimeter": "cm", "kilometer": "km",
    }
    return table.get(name, name)

def _num_to_str(x: Any) -> str:
    if isinstance(x, (int,)):
        return str(x)
    if isinstance(x, float):
        # trim trailing .0
        s = f"{x}"
        return s[:-2] if s.endswith(".0") else s
    return str(x)

def _fmt_value(v: Any) -> str:
    # Physical with Range => "[10m..20m]"
    if isinstance(v, Physical):
        unit = _unit_abbr(v.unit)
        if isinstance(v.num, Range):
            return f"[low: {_num_to_str(v.num.start)}{unit}, high: {_num_to_str(v.num.end)}{unit}]"
        return f"{_num_to_str(v.num)}{unit}"
    # bare Range (should be rare here)
    if isinstance(v, Range):
        return f"[{_num_to_str(v.start)}..{_num_to_str(v.end)}]"
    # strings as names/labels – keep quotes like your current output
    if isinstance(v, str):
        return repr(v)
    return _num_to_str(v)

def _print_block(block, indent="  "):
    pad = indent
    if isinstance(block, SerialBlock):
        label = f"{block.label}:" if block.label else ""
        print(f"{pad}serial{(', ' + label) if label else ':'}")
        for ch in block.children:
            _print_block(ch, indent + "  ")
    elif isinstance(block, ParallelBlock):
        head = "parallel"
        if block.duration is not None:
            head += f", duration: {_fmt_value(block.duration)}"
        label = f" ({block.label})" if block.label else ""
        print(f"{pad}{head}:{label}")
        for ch in block.children:
            _print_block(ch, indent + "  ")
    elif isinstance(block, ActionCall):
        print(f"{pad}{block.actor}.{block.action}()")
        for m in block.modifiers:
            # Build "name(pos1, pos2, k=v, ...)"
            parts: List[str] = []
            if getattr(m, "args", None):
                parts.extend(_fmt_value(a) for a in m.args)
            if getattr(m, "kwargs", None):
                parts.extend(f"{k}={_fmt_value(v)}" for k, v in m.kwargs.items())
            inside = ", ".join(parts)
            print(f"{pad}  with {m.name}({inside})")
    else:
        # Unknown child (defensive)
        print(f"{pad}{block}")

def print_ir(scenarios: List[ScenarioNode]) -> None:
    for scn in scenarios:
        print(f"Scenario: {scn.name}")
        # Actors
        if scn.actors:
            print("  Actors:")
            for a in scn.actors.values():
                print(f"    - {a.name}: {a.type}")
        # Vars
        if scn.vars:
            print("  Vars:")
            for v in scn.vars.values():
                ty = f": {v.type}" if getattr(v, "type", None) else ""
                print(f"    - {v.name}{ty} = {_fmt_value(v.value)}")
        # Events
        if scn.events:
            print("  Events:")
            for e in scn.events:
                print(f"    - {e.name}")
        # Do / blocks
        if scn.blocks:
            print("  Do:")
            for b in scn.blocks:
                _print_block(b, indent="    ")
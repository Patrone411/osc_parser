from typing import Dict, List, Optional

def extract_serial_of_parallel(
    scn_dict: Dict,
    serial_label: Optional[str] = None
) -> List[Dict]:
    """
    From constraints_from_ir(scenarios)[<name>], pull the first (or a named) top-level
    serial block and return an ordered list of its parallel children, each with
    'duration' and 'calls' — the exact shape match_serial_program(...) expects.
    """
    blocks = scn_dict.get("blocks", [])
    if serial_label is not None:
        serial_root = next(
            b for b in blocks if b.get("type") == "serial" and b.get("label") == serial_label
        )
    else:
        serial_root = next(b for b in blocks if b.get("type") == "serial")

    serial_blocks = []
    for child in serial_root.get("children", []):
        if child.get("type") != "parallel":
            continue
        serial_blocks.append({
            "type": "parallel",
            "duration": child.get("duration"),
            "calls": child.get("calls", []),
        })
    return serial_blocks

# OpenSCENARIO 2.0 Parser (Semantic Extension)

A Python-based **OpenSCENARIO 2.0 parser** built on a fork of CARLA’s  
`scenario_runner`, extended with **early semantic validation** and a  
**domain-aligned intermediate representation (IR)**.

The parser goes beyond syntax and validates scenarios against the  
OpenSCENARIO 2.0 **domain model** (actors, actions, physical units, scopes),  
making it suitable for **static analysis, filtering, and matching** of  
scenarios.

---

## Key Features

- ANTLR4-based OpenSCENARIO 2.0 frontend  
- Domain-aware **actor / action / modifier validation**  
- Canonical symbols and actor-aware scope resolution  
- **Physical units & dimension checking** (time, length, speed, …)  
- Multi-stage pipeline: **AST → IR → lowered IR**  
- Early rejection of semantically invalid scenarios  

---

## Usage

```python
from osc2_parser.parser import OSCProgram

osc_file = "change_lane.osc"
osc_prefix = "osc2_parser/osc/"
prog = OSCProgram(osc_path=osc_prefix + osc_file).compile()
```

---

## Quick API Overview

```python
prog = OSCProgram(...).compile()
```

`compile()` returns a `CompiledOSC` object exposing the validated scenario via  
a small, stable API:

### prog.constraints_by_scenario

Structured, per-scenario representation preserving:

- actor instances and their domain types  
- block hierarchy (serial / parallel), labels, and durations  
- validated actions and modifiers  
- typed physical values with explicit units  

### prog.calls

Flattened list of semantically validated action calls  
(derived from `constraints_by_scenario`), intended for downstream  
analysis, filtering, or matching.

### prog.min_lanes

Inferred minimum lane requirement of the scenario.

### prog.block_durations

Mapping from block labels to validated time durations.

### prog.validation_result, prog.validation_errors

Outcome of semantic validation against the OpenSCENARIO 2.0 domain model.

---

## Compilation Output (Structure)

```python
{
  "<scenario>": {
    "actors": {
      "ego_vehicle": {"type": "vehicle"}
    },
    "blocks": [
      {
        "type": "parallel",
        "label": "get_ahead",
        "duration": {"value": 15, "unit": "second"},
        "calls": [
          {
            "actor": "ego_vehicle",
            "action": "drive",
            "action_args": {
              "duration": {"value": 12, "unit": "second"}
            },
            "modifiers": [...]
          }
        ],
        "children": [...]
      }
    ],
    "calls_flat": [
      {
        "actor": "ego_vehicle",
        "action": "drive",
        "action_args": {...},
        "modifiers": [...],
        "block_label": "get_ahead",
        "block_type": "parallel"
      }
    ]
  }
}
```

All actions, parameters, and physical values appearing in this structure are  
**semantically validated and type-safe**.

---

## Scope

- Grammar: OpenSCENARIO 2.0 (unchanged)  
- Focus: static semantic correctness, not runtime execution  
- Designed as a foundation for downstream analysis and matching pipelines  

---

## Background

Developed as part of a master’s thesis on semantic validation, filtering, and  
matching of OpenSCENARIO 2.0 scenarios.

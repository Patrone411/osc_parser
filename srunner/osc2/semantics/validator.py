from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Set, Callable
from osc_parser.srunner.osc2_dm.physical_types import Physical

from .registry import SemanticsRegistry, ActionSpec, OverloadSpec, ModifierSpec, ModifierVariantSpec

# Fallback logging if your project logger isn't available
try:
    from osc_parser.srunner.osc2.utils.log_manager import LOG_ERROR, LOG_WARN
except Exception:
    def LOG_ERROR(msg, token=None): raise ValueError(msg)
    def LOG_WARN(msg, token=None): print("WARN:", msg)

# ---------- Public API types ----------

@dataclass
class ArgValue:
    """One actual argument in a call/modifier clause."""
    name: str
    value: Any
    type_name: Optional[str] = None  # optional; filled by type_of_expr hook if needed

@dataclass
class ModifierAttach:
    name: str
    args: Dict[str, ArgValue]
    token: Any = None

@dataclass
class ActionCall:
    invoker_name: str
    invoker_type: str
    method_name: str
    args: Dict[str, ArgValue]
    modifiers: List[ModifierAttach]
    token: Any = None

@dataclass
class ValidationResult:
    resolved_action_qname: str
    resolved_action_overload: str  # overload name or "<default>"
    resolved_args: Dict[str, Any]
    resolved_modifiers: List[Tuple[str, str, Dict[str, Any]]]  # (mod_name, variant_name, args)

def infer_type(arg: ArgValue):
    v = arg.value
    if isinstance(v, Physical):
        u = v.unit

        # Try common attribute names for the physical dimension object
        phys_obj = (
            getattr(u, "physical", None)
            or getattr(u, "physical_object", None)
            or getattr(u, "physical_type", None)
        )

        # Try to pull a name off the physical object
        phys_name = (
            getattr(phys_obj, "type_name", None)
            or getattr(phys_obj, "name", None)
            or getattr(phys_obj, "physical_name", None)
        )

        if phys_name:
            # e.g. "length", "time", "speed", "acceleration"
            print(f"[infer_type] {arg.name}={v} -> {phys_name} (unit={getattr(u,'unit_name',getattr(u,'name','?'))})")
            return phys_name

        # Fallback: infer by canonical unit name
        uname = getattr(u, "unit_name", None) or getattr(u, "name", None)

        time_units = {"second","millisecond","minute","hour"}
        length_units = {"meter","millimeter","centimeter","kilometer","inch","feet","mile","micrometer"}
        speed_units = {"meter_per_second","kilometer_per_hour","mile_per_hour"}
        accel_units = {"meter_per_sec_sqr","kilometer_per_hour_per_sec","mile_per_hour_per_sec","feet_per_sec_sqr"}

        if uname in time_units:   return "time"
        if uname in length_units: return "length"
        if uname in speed_units:  return "speed"
        if uname in accel_units:  return "acceleration"

        # Last resort: unknown physical
        # print(f"[infer_type] {arg.name}: unknown unit '{uname}' -> None")
        return None
    
    if isinstance(v, bool):  return "bool"
    if isinstance(v, int):   return "int"
    if isinstance(v, float): return "float"
    if isinstance(v, str):   return "string"
    return None


# ---------- Validator ----------

class SemanticValidator:
    """
    Stateless semantic checker using a SemanticsRegistry.
    Host feeds it ActionCall records (from IR adapter or AST adapter).
    """

    def __init__(self, registry: SemanticsRegistry, type_of_expr: Optional[Callable[[ArgValue], Optional[str]]] = None, debug_types: bool=False):
        self.registry = registry
        # Hook to infer type name from ArgValue if not given (e.g. Physical -> "length")
        self.type_of_expr = type_of_expr or (lambda a: a.type_name)
        self.debug_types = debug_types

    def _dbg(self, msg: str):
        if getattr(self, "debug_types", False):
            print(msg)
    # =====================================================================
    # Normalization helpers (dicts vs. objects)
    # =====================================================================
    def _arg_type_ok(self, expected: str, arg: "ArgValue") -> bool:
        actual = arg.type_name or self.type_of_expr(arg)
        self._dbg(f"[type-of] param={arg.name!r} expected={expected!r} actual={actual!r} value={arg.value!r}")
        if actual is None:
            # STRICT: if we can’t infer a type, treat it as mismatch
            return False

        # small convenience example
        if expected == "uint" and actual == "int":
            return True

        # allow enums to be passed as strings (optional)
        enums = {}
        if hasattr(self.registry, "enums"):
            enums = self.registry.enums
        elif hasattr(self.registry, "doc"):
            enums = (self.registry.doc or {}).get("enums", {})

        if expected in enums:
            # treat plain strings as OK for enums
            return actual in (expected, "string")

        return actual == expected
    
    def _get_action_spec(self, qname: str):
        """
        Return the action spec (dict or object) for a qualified action name.
        Works with several registry shapes.
        """
        # Preferred explicit getter
        if hasattr(self.registry, "get_action"):
            try:
                spec = self.registry.get_action(qname)
                if spec is not None:
                    return spec
            except Exception:
                pass

        # Common direct mapping attribute
        actions = getattr(self.registry, "actions", None)
        if isinstance(actions, dict) and qname in actions:
            return actions[qname]

        # JSON-esque document storage
        doc = getattr(self.registry, "doc", None)
        if isinstance(doc, dict):
            return doc.get("actions", {}).get(qname)

        # Last resort: registry could itself be dict-like
        if isinstance(self.registry, dict):
            return self.registry.get("actions", {}).get(qname, {})

        return None

    def _spec_inherits(self, spec) -> Optional[str]:
        """Read 'inherits' from dict or object spec."""
        if spec is None:
            return None
        if isinstance(spec, dict):
            return spec.get("inherits")
        return getattr(spec, "inherits", None)

    def _spec_overloads(self, spec):
        """Read 'overloads' from dict or object spec; always returns a list."""
        if spec is None:
            return []
        if isinstance(spec, dict):
            return spec.get("overloads", []) or []
        return getattr(spec, "overloads", []) or []

    def _overload_params_map(self, overload) -> Dict[str, Dict[str, Any]]:
        """
        Normalize an overload's params into a dict:
            name -> { "type": str|None, "optional": bool, ["default": Any] }
        Handles dict- or object-style overloads and param specs.
        """
        # 1) load raw container
        if isinstance(overload, dict):
            params = overload.get("params", {}) or {}
        else:
            params = getattr(overload, "params", {}) or {}

        out: Dict[str, Dict[str, Any]] = {}

        # Mapping: {name -> (dict|ParamSpec)}
        if isinstance(params, dict):
            for name, p in params.items():
                if isinstance(p, dict):
                    entry = {
                        "type": p.get("type"),
                        "optional": p.get("optional", False),
                    }
                    if "default" in p:
                        entry["default"] = p["default"]
                    out[name] = entry
                else:
                    # object-like ParamSpec
                    p_type = getattr(p, "type", None) or getattr(p, "param_type", None)
                    p_opt  = getattr(p, "optional", False) or getattr(p, "is_optional", False)
                    entry = {"type": p_type, "optional": p_opt}
                    if hasattr(p, "default"):
                        entry["default"] = getattr(p, "default")
                    out[name] = entry
            return out

        # Iterable: [ParamSpec or dict] with explicit 'name'
        if isinstance(params, (list, tuple)):
            for p in params:
                if isinstance(p, dict):
                    name = p.get("name")
                    if not name:
                        continue
                    entry = {
                        "type": p.get("type"),
                        "optional": p.get("optional", False),
                    }
                    if "default" in p:
                        entry["default"] = p["default"]
                    out[name] = entry
                else:
                    name = getattr(p, "name", None)
                    if not name:
                        continue
                    p_type = getattr(p, "type", None) or getattr(p, "param_type", None)
                    p_opt  = getattr(p, "optional", False) or getattr(p, "is_optional", False)
                    entry = {"type": p_type, "optional": p_opt}
                    if hasattr(p, "default"):
                        entry["default"] = getattr(p, "default")
                    out[name] = entry
            return out

        return out

    def _extract_candidate(self, item) -> Tuple[Optional[str], Optional[Any], Optional[str]]:
        """
        Normalize a candidate entry to (qualified_action_name, overload, overload_name).
        Supports:
          - (qname, overload) tuples
          - (qname, overload, overload_name) tuples
          - object with attributes: qualified_name/qname/name, overload/spec/overload_spec, overload_name
        """
        if isinstance(item, (tuple, list)):
            if len(item) == 3:
                return item[0], item[1], item[2]
            if len(item) == 2:
                return item[0], item[1], None
            return None, None, None

        # object-like
        qname = (
            getattr(item, "qualified_name", None)
            or getattr(item, "qname", None)
            or getattr(item, "name", None)
        )
        ov = (
            getattr(item, "overload", None)
            or getattr(item, "spec", None)
            or getattr(item, "overload_spec", None)
        )
        ovn = getattr(item, "overload_name", None)
        # fall back to overload.name if present
        if ovn is None and ov is not None:
            ovn = getattr(ov, "name", None)
        return qname, ov, ovn

    # =====================================================================
    # Inheritance parameter merge
    # =====================================================================

    def _params_from_inherits(self, action_qname: str) -> Dict[str, Dict[str, Any]]:
        """
        Walk 'inherits' chain and merge all ancestor overload params into one map.
        Child definitions will override parents later (we merge child last).
        """
        merged: Dict[str, Dict[str, Any]] = {}
        seen: Set[str] = set()
        cur = action_qname
        while cur and cur not in seen:
            seen.add(cur)
            spec = self._get_action_spec(cur)
            if not spec:
                break
            for ov in self._spec_overloads(spec):
                merged.update(self._overload_params_map(ov))
            cur = self._spec_inherits(spec)
        return merged

    # =====================================================================
    # Top-level
    # =====================================================================

    def validate_action_call(self, call: ActionCall) -> ValidationResult:
        # 1) candidates by method + invoker ancestry
        candidates = self.registry.find_actions_for_method_on(call.invoker_type, call.method_name)
        if not candidates:
            LOG_ERROR(
                f"No action '{call.method_name}' is defined for actor type '{call.invoker_type}' (or its bases).",
                call.token
            )

        # 2) choose concrete overload
        chosen_qname, chosen_spec, ov_name, resolved_args = self._choose_action_overload(call, candidates)

        # 3) modifiers
        resolved_mods: List[Tuple[str, str, Dict[str, Any]]] = []
        for m in call.modifiers:
            mod_spec: ModifierSpec = self.registry.get_modifier(m.name)
            if not mod_spec:
                LOG_ERROR(f"Modifier '{m.name}' is not defined.", m.token)
            family = self.registry.action_family_of(chosen_qname)
            if not self.registry.is_action_in_family(chosen_qname, mod_spec.applies_to):
                LOG_ERROR(
                    f"Modifier '{m.name}' does not apply to action '{chosen_qname}' "
                    f"(expected family '{mod_spec.applies_to}', got '{family}').",
                    m.token
                )
            v_name, v_args = self._choose_modifier_variant(m, mod_spec, call)
            resolved_mods.append((m.name, v_name, v_args))

        return ValidationResult(
            resolved_action_qname=chosen_qname,
            resolved_action_overload=ov_name or "<default>",
            resolved_args=resolved_args,
            resolved_modifiers=resolved_mods,
        )

    # =====================================================================
    # Overload resolution (actions)
    # =====================================================================

    def _choose_action_overload(self, call: ActionCall, candidates):
        """
        candidates: iterable of (qname, overload[, overload_name]) or object equivalents
        call.args: dict[str, ArgValue] of supplied named args (e.g., {'duration': ArgValue(...)})
        """
        supplied_names: Set[str] = set(call.args.keys())
        failures = []

        for item in candidates:
            qname, ov, ov_name = self._extract_candidate(item)
            if not qname or ov is None:
                continue

            parent_params = self._params_from_inherits(qname)
            own_params    = self._overload_params_map(ov)
            # Child wins on conflicts
            effective_params = {**parent_params, **own_params}

            param_names = set(effective_params.keys())

            # unknown args?
            unknown = supplied_names - param_names
            if unknown:
                failures.append((qname, f"unknown args: {sorted(unknown)}"))
                continue

            # missing required?
            required = {k for k, v in effective_params.items() if not v.get("optional", False)}
            missing = required - supplied_names
            if missing:
                failures.append((qname, f"missing required: {sorted(missing)}"))
                continue

            # ---- NEW: type checks for supplied args ----
            type_mismatch = []
            for k in supplied_names:
                expected = effective_params.get(k, {}).get("type")
                if expected:
                    if not self._arg_type_ok(expected, call.args[k]):
                        got = call.args[k].type_name or self.type_of_expr(call.args[k])
                        type_mismatch.append((k, expected, got))
                    else:
                        self._dbg(f"[type-check ✓] action={qname} param={k!r} ok")
                else:
                    self._dbg(f"[type-check ?] action={qname} param={k!r} has no 'type' in registry")
            if type_mismatch:
                self._dbg(f"[type-check ✗] action={qname} mismatches={type_mismatch}")
                failures.append((qname, f"type mismatch: {type_mismatch}"))
                continue
            # ---- END NEW ----

            # SUCCESS -> return normalized values (not ArgValue wrappers)
            resolved_args = {k: v.value for k, v in call.args.items()}
            return qname, ov, ov_name, resolved_args

        # no match -> keep your current error behavior
        LOG_ERROR(
            f"No overload of '{call.method_name}' matches the supplied arguments: {sorted(supplied_names)}.",
            call.token
        )

    # =====================================================================
    # Variant resolution (modifiers)
    # =====================================================================

    def _choose_modifier_variant(self, m: ModifierAttach, spec: ModifierSpec, call: ActionCall) -> Tuple[str, Dict[str, Any]]:
        matches: List[Tuple[str, Dict[str, Any]]] = []
        var_list = spec.variants or [ModifierVariantSpec(None, {}, {})]
        for var in var_list:
            ok, resolved = self._args_match_variant(m, var, call)
            if ok:
                matches.append((var.name or "<default>", resolved))
        if not matches:
            LOG_ERROR(f"No variant of modifier '{m.name}' matches supplied arguments: {list(m.args.keys())}", m.token)
        if len(matches) > 1 and spec.rules.get("overloads_mutually_exclusive", False):
            LOG_ERROR(f"Ambiguous modifier '{m.name}': multiple variants match: {[x[0] for x in matches]}", m.token)
        return matches[0]

    def _args_match_variant(self, m: ModifierAttach, var: ModifierVariantSpec, call: ActionCall) -> Tuple[bool, Dict[str, Any]]:
        provided: Set[str] = set(m.args.keys())
        resolved: Dict[str, Any] = {}

        for pn, ps in var.params.items():
            if not ps.optional and pn not in provided:
                return False, {}

        unknown = provided - set(var.params.keys())
        if unknown:
            return False, {}

        for pn, ps in var.params.items():
            if pn in m.args:
                arg = m.args[pn]
                arg_t = arg.type_name or self.type_of_expr(arg)
                self._dbg(f"[mod-type-of] mod={m.name} param={pn!r} expected={ps.type!r} actual={arg_t!r} value={m.args[pn].value!r}")
                if arg_t and ps.type and arg_t != ps.type:
                    return False, {}
                resolved[pn] = arg.value
                if ps.ignored_if_present:
                    for bad in ps.ignored_if_present:
                        if bad in provided:
                            LOG_WARN(f"Modifier '{m.name}': parameter '{pn}' is ignored when '{bad}' is present.", m.token)
            else:
                if ps.optional and ps.default is not None:
                    resolved[pn] = call.invoker_name if ps.default == "<actor>" else ps.default

        if not self._check_rules_groups(var.rules, resolved, provided):
            return False, {}
        return True, resolved

    # =====================================================================
    # Common rules
    # =====================================================================

    def _check_rules_groups(self, rules: Dict[str, List[List[str]]], resolved: Dict[str, Any], provided: Set[str]) -> bool:
        for group in rules.get("exactly_one_of", []):
            cnt = sum(1 for k in group if k in provided)
            if cnt != 1:
                return False
        for group in rules.get("at_least_one_of", []):
            cnt = sum(1 for k in group if k in provided)
            if cnt < 1:
                return False
        for pair in rules.get("requires", []):
            if len(pair) == 2:
                a, b = pair
                if a in provided and b not in provided:
                    return False
        return True

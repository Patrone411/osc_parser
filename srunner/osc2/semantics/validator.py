from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Set, Callable

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

# ---------- Validator ----------

class SemanticValidator:
    """
    Stateless semantic checker using a SemanticsRegistry.
    Host feeds it ActionCall records (from IR adapter or AST adapter).
    """

    def __init__(self, registry: SemanticsRegistry, type_of_expr: Optional[Callable[[ArgValue], Optional[str]]] = None):
        self.registry = registry
        # Hook to infer type name from ArgValue if not given (e.g. Physical -> "length")
        self.type_of_expr = type_of_expr or (lambda a: a.type_name)

    # ---- Top-level ----

    def validate_action_call(self, call: ActionCall) -> ValidationResult:
        # 1) candidates by method + invoker ancestry
        candidates = self.registry.find_actions_for_method_on(call.invoker_type, call.method_name)
        if not candidates:
            LOG_ERROR(f"No action '{call.method_name}' is defined for actor type '{call.invoker_type}' (or its bases).", call.token)

        # 2) choose concrete overload
        chosen_qname, chosen_spec, ov_name, resolved_args = self._choose_action_overload(call, candidates)

        # 3) modifiers
        resolved_mods = []
        for m in call.modifiers:
            mod_spec = self.registry.get_modifier(m.name)
            if not mod_spec:
                LOG_ERROR(f"Modifier '{m.name}' is not defined.", m.token)
            family = self.registry.action_family_of(chosen_qname)
            if not self.registry.is_action_in_family(chosen_qname, mod_spec.applies_to):
                LOG_ERROR(
                    f"Modifier '{m.name}' does not apply to action '{chosen_qname}' "
                    f"(expected family '{mod_spec.applies_to}', got '{family}').", m.token
                )
            v_name, v_args = self._choose_modifier_variant(m, mod_spec, call)
            resolved_mods.append((m.name, v_name, v_args))

        return ValidationResult(
            resolved_action_qname=chosen_qname,
            resolved_action_overload=ov_name or "<default>",
            resolved_args=resolved_args,
            resolved_modifiers=resolved_mods,
        )

    # ---- Overload resolution (actions) ----

    def _choose_action_overload(
        self,
        call: ActionCall,
        candidates: List[Tuple[str, ActionSpec]],
    ) -> Tuple[str, ActionSpec, Optional[str], Dict[str, Any]]:

        matches: List[Tuple[str, ActionSpec, Optional[str], Dict[str, Any]]] = []

        for qname, spec in candidates:
            if spec.abstract:
                continue
            ov_list = spec.overloads or [OverloadSpec(None, {}, {})]
            for ov in ov_list:
                ok, resolved = self._args_match_overload(call, ov)
                if ok:
                    matches.append((qname, spec, ov.name, resolved))

        if not matches:
            LOG_ERROR(f"No overload of '{call.method_name}' matches the supplied arguments: {list(call.args.keys())}.", call.token)

        if len(matches) > 1:
            alts = [f"{q}.{ov or '<default>'}" for q, _, ov, _ in matches]
            LOG_ERROR(f"Ambiguous call '{call.method_name}': multiple overloads match: {alts}", call.token)

        return matches[0]

    def _args_match_overload(self, call: ActionCall, ov: OverloadSpec) -> Tuple[bool, Dict[str, Any]]:
        provided: Set[str] = set(call.args.keys())
        resolved: Dict[str, Any] = {}

        # required params
        for pn, ps in ov.params.items():
            if not ps.optional and pn not in provided:
                return False, {}

        # unknown params
        unknown = provided - set(ov.params.keys())
        if unknown:
            return False, {}

        # type checks + defaults
        for pn, ps in ov.params.items():
            if pn in call.args:
                arg = call.args[pn]
                arg_t = arg.type_name or self.type_of_expr(arg)
                if arg_t and ps.type and arg_t != ps.type:
                    return False, {}
                resolved[pn] = arg.value
                # ignored_if_present warnings
                if ps.ignored_if_present:
                    for bad in ps.ignored_if_present:
                        if bad in provided:
                            LOG_WARN(f"Parameter '{pn}' is ignored when '{bad}' is present.", call.token)
            else:
                if ps.optional and ps.default is not None:
                    resolved[pn] = call.invoker_name if ps.default == "<actor>" else ps.default

        # group rules
        if not self._check_rules_groups(ov.rules, resolved, provided):
            return False, {}

        return True, resolved

    # ---- Variant resolution (modifiers) ----

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

    # ---- common ----

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

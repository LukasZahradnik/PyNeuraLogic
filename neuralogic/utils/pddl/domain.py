from typing import Any
from neuralogic.core.constructs.factories import R, V, C
from neuralogic.core.constructs.relation import BaseRelation
from neuralogic.core.constructs.rule import Rule


class Action:
    """Represents a PDDL Action and its mapping to NeuraLogic Rules."""
    def __init__(self, name: str, parameters: list[tuple], precondition: Any, effect: Any):
        self.name = name
        self.parameters = parameters
        self.precondition = precondition
        self.effect = effect

    def to_rules(self) -> list[Rule]:
        """Convert PDDL action to NeuraLogic rules (effect <= precondition)."""
        body = self._parse_literal(self.precondition)
        heads = self._parse_literal(self.effect)

        if not isinstance(heads, list):
            heads = [heads]

        return [head <= body for head in heads]

    def _parse_literal(self, sexpr: Any) -> BaseRelation | list[BaseRelation]:
        """Parse PDDL literal or conjunction into NeuraLogic relations."""
        if not sexpr:
            return []

        if sexpr[0] == "and":
            literals = []
            for sub in sexpr[1:]:
                res = self._parse_literal(sub)
                if isinstance(res, list):
                    literals.extend(res)
                else:
                    literals.append(res)
            return literals

        if sexpr[0] == "not":
            res = self._parse_literal(sexpr[1])
            if isinstance(res, list):
                return [~r for r in res]
            return ~res

        # Atomic literal
        pred_name = sexpr[0]
        args = []
        for arg in sexpr[1:]:
            if arg.startswith("?"):
                args.append(V.get(arg[1:]))
            else:
                args.append(C.get(arg))

        return R.get(pred_name)(*args)


class PDDLDomain:
    """Represents a PDDL Domain and its components."""
    def __init__(self, sexpr: Any):
        self.name = ""
        self.requirements = []
        self.types = {}
        self.predicates = []
        self.actions = []
        self._parse(sexpr)

    def _parse(self, sexpr: Any) -> None:
        if not sexpr or sexpr[0] != "define":
            return

        for item in sexpr[1:]:
            if not isinstance(item, list):
                continue
            
            tag = item[0]
            if tag == "domain":
                self.name = item[1]
            elif tag == ":requirements":
                self.requirements = item[1:]
            elif tag == ":types":
                # Basic type support (TODO: hierarchy)
                self.types = item[1:]
            elif tag == ":predicates":
                self.predicates = item[1:]
            elif tag == ":action":
                self._parse_action(item)

    def _parse_action(self, item: list[Any]) -> None:
        name = item[1]
        params = []
        precondition = None
        effect = None

        i = 2
        while i < len(item):
            if item[i] == ":parameters":
                params = item[i+1] # e.g. (?x - type1 ?y - type2)
                i += 2
            elif item[i] == ":precondition":
                precondition = item[i+1]
                i += 2
            elif item[i] == ":effect":
                effect = item[i+1]
                i += 2
            else:
                i += 1
        
        self.actions.append(Action(name, params, precondition, effect))

    def to_rules(self) -> list[Rule]:
        """Returns all rules corresponding to domain actions."""
        rules: list[Rule] = []
        for action in self.actions:
            rules.extend(action.to_rules())
        return rules

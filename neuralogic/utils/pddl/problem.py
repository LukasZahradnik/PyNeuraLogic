from typing import Any
from neuralogic.core.constructs.factories import R, C
from neuralogic.core.constructs.relation import BaseRelation


class PDDLProblem:
    """Represents a PDDL Problem and its components (objects, init, goal)."""
    def __init__(self, sexpr: Any):
        self.name = ""
        self.domain = ""
        self.objects = []
        self.init = []
        self.goal = []
        self._parse(sexpr)

    def _parse(self, sexpr: Any) -> None:
        if not sexpr or sexpr[0] != "define":
            return

        for item in sexpr[1:]:
            if not isinstance(item, list):
                continue
            
            tag = item[0]
            if tag == "problem":
                self.name = item[1]
            elif tag == "domain":
                self.domain = item[1]
            elif tag == ":objects":
                self.objects = item[1:]
            elif tag == ":init":
                self.init = [self._parse_literal(lit) for lit in item[1:]]
            elif tag == ":goal":
                self.goal = self._parse_goal(item[1])

    def _parse_goal(self, sexpr: Any) -> list[BaseRelation]:
        """Parse PDDL goal into NeuraLogic relations."""
        if not sexpr:
            return []
        
        if sexpr[0] == "and":
            return [self._parse_literal(sub) for sub in sexpr[1:]]
        
        return [self._parse_literal(sexpr)]

    def _parse_literal(self, sexpr: Any) -> BaseRelation:
        """Parse PDDL literal into NeuraLogic relation."""
        if sexpr[0] == "not":
            return ~self._parse_literal(sexpr[1])
        
        pred_name = sexpr[0]
        args = [C.get(arg) for arg in sexpr[1:]]
        return R.get(pred_name)(*args)

    def get_objects(self) -> list[Any]:
        """Returns the constants representing PDDL objects."""
        return [C.get(obj) for obj in self.objects]

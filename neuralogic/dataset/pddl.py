import os
from typing import Optional
from neuralogic.dataset.logic import Dataset, Sample
from neuralogic.dataset.base import ConvertibleDataset
from neuralogic.utils.pddl.parser import PDDLReader
from neuralogic.utils.pddl.domain import PDDLDomain
from neuralogic.utils.pddl.problem import PDDLProblem


class PDDLDataset(ConvertibleDataset):
    """
    PDDLDataset converts PDDL domain and problem files into a logic dataset.
    It supports creating samples from the initial state and using the goal state as a query.
    """
    def __init__(
        self, 
        domain: str, 
        problems: str | list[str], 
        include_actions: bool = True
    ):
        """
        Parameters
        ----------
        domain : str
            Path to the PDDL domain file or the domain string content.
        problems : Union[str, List[str]]
            Path/string or list of paths/strings to PDDL problem files/content.
        include_actions : bool, optional
            Whether to include action definitions as rules in the example. Default: True.
        """
        self.domain_input = domain
        self.problem_inputs: list[str] = [problems] if isinstance(problems, str) else problems
        self.include_actions = include_actions

        self.domain: PDDLDomain | None = None
        self.problems: list[PDDLProblem] = []

    def _load(self) -> None:
        """Load and parse PDDL from paths or strings."""
        if self.domain is None:
            if os.path.isfile(self.domain_input):
                domain_sexpr = PDDLReader.read(self.domain_input)
            else:
                domain_sexpr = PDDLReader.parse_string(self.domain_input)
            self.domain = PDDLDomain(domain_sexpr)
        
        if not self.problems:
            for problem_input in self.problem_inputs:
                if os.path.isfile(problem_input):
                    problem_sexpr = PDDLReader.read(problem_input)
                else:
                    problem_sexpr = PDDLReader.parse_string(problem_input)
                self.problems.append(PDDLProblem(problem_sexpr))

    def to_dataset(self) -> Dataset:
        """
        Converts the PDDL domain and problems into a NeuraLogic Dataset.

        Returns
        -------
        Dataset
            The created Dataset object containing Samples.
        """
        self._load()
        samples = []

        # Domain rules (actions)
        domain_rules = self.domain.to_rules() if self.include_actions else []

        for problem in self.problems:
            # Example: Initial state + Domain rules
            example = problem.init + domain_rules
            
            # Query: Goal state
            # If multiple goal literals, we currently take them as a list of queries
            query = problem.goal
            
            samples.append(Sample(query, example))

        return Dataset(samples)

    def __str__(self) -> str:
        return f"PDDLDataset(n_problems={len(self.problem_inputs)})"

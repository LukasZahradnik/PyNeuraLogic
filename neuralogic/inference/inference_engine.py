from typing import List, Union, Optional, Tuple, Dict

import jpype

from neuralogic import is_initialized, initialize
from neuralogic.core import Template, Settings, R
from neuralogic.core.constructs.java_objects import JavaFactory
from neuralogic.core.builder import DatasetBuilder
from neuralogic.core.constructs.relation import BaseRelation
from neuralogic.core.constructs.rule import Rule


class InferenceEngine:
    def __init__(self, template: Template, settings: Settings = None):
        if not is_initialized():
            initialize()

        self.settings = Settings().create_disconnected_proxy() if settings is None else settings.create_proxy()
        self.java_factory = JavaFactory()

        self.parsed_template = template.get_parsed_template(self.settings, self.java_factory)
        self.dataset_builder = DatasetBuilder(self.parsed_template, self.java_factory)

        self.examples: List[Union[BaseRelation, Rule]] = []

        self.grounder = jpype.JClass("cz.cvut.fel.ida.logic.grounding.Grounder").getGrounder(self.settings.settings)
        self.matching = jpype.JClass("cz.cvut.fel.ida.logic.subsumption.Matching")()
        self.examples_builder = jpype.JClass("cz.cvut.fel.ida.logic.constructs.building.ExamplesBuilder")
        self.queries_builder = jpype.JClass("cz.cvut.fel.ida.logic.constructs.building.QueriesBuilder")
        self.grounding_sample = jpype.JClass("cz.cvut.fel.ida.logic.grounding.GroundingSample")
        self.horn_clause = jpype.JClass("cz.cvut.fel.ida.logic.HornClause")

        self.empty_example = jpype.JClass("cz.cvut.fel.ida.logic.constructs.example.LiftedExample")()

    def set_knowledge(self, examples: List[Union[BaseRelation, Rule]]) -> None:
        self.examples = examples

    def get_queries(self, examples: Optional[List[Union[BaseRelation, Rule]]] = None):
        if examples is None:
            examples = self.examples

        examples_builder = self.examples_builder(self.settings.settings)

        self.java_factory.weight_factory = self.java_factory.get_new_weight_factory()
        built_examples = self.dataset_builder.build_examples([examples], examples_builder)[0]
        sample = built_examples[0]

        gs = self.grounding_sample(sample, self.parsed_template)

        lifted_example = gs.query.evidence
        template = gs.template

        ground_template = self.grounder.groundRulesAndFacts(lifted_example, template)

        ground_rules = ground_template.groundRules.values()
        for ground_rule in ground_rules:
            for head in ground_rule.keys():
                ground_head = head.groundHead

                yield R.get(str(ground_head.predicateName()))([str(term.name()) for term in ground_head.arguments()])

    def q(self, query: BaseRelation, examples: Optional[List[Union[BaseRelation, Rule]]] = None):
        return self.query(query, examples)

    def query(self, query: BaseRelation, examples: Optional[List[Union[BaseRelation, Rule]]] = None):
        if examples is None:
            examples = self.examples

        examples_builder = self.examples_builder(self.settings.settings)
        query_builder = self.queries_builder(self.settings.settings)
        query_builder.setFactoriesFrom(examples_builder)

        self.java_factory.weight_factory = self.java_factory.get_new_weight_factory()
        queries, one_query_per_example = self.dataset_builder.build_queries([query], query_builder)

        self.java_factory.weight_factory = self.java_factory.get_new_weight_factory()
        examples = self.dataset_builder.build_examples([examples], examples_builder)[0]

        logic_samples = self.dataset_builder.merge_queries_with_examples(
            queries, examples, examples_builder, one_query_per_example
        )
        sample = logic_samples[0]

        gs = self.grounding_sample(sample, self.parsed_template)

        lifted_example = gs.query.evidence
        template = gs.template

        ground_template = self.grounder.groundRulesAndFacts(lifted_example, template)

        clause = self.java_factory.atom_to_clause(query)
        name = str(query.predicate)
        results: List[Dict[str, str]] = []
        variables = [(index, term) for index, term in enumerate(query.terms) if str(term)[0].isupper()]

        self._get_substitutions(clause, name, variables, ground_template.groundRules, results)
        self._get_substitutions(clause, name, variables, ground_template.groundFacts, results)

        if len(results) == 0:
            return {}

        if len(variables) == 0:
            return iter([])

        return results

    def _get_substitutions(
        self, clause, query_signature: str, variables: List[Tuple[int, str]], literals, substitutions: List
    ):
        for literal in literals:
            if str(literal.predicate().toString()) == query_signature and self.matching.subsumption(
                clause, self.java_factory.clause(literal)
            ):
                terms = literal.arguments()
                substitutions.append({str(label): str(terms[index]) for index, label in variables})

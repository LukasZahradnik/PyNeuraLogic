from typing import List, Union, Optional

import jpype

from neuralogic import is_initialized, initialize
from neuralogic.core import Template, JavaFactory, Settings
from neuralogic.core.builder import DatasetBuilder
from neuralogic.core.constructs.atom import AtomType
from neuralogic.core.constructs.rule import Rule


class InferenceEngine:
    def __init__(self, template: Template):
        if not is_initialized():
            initialize()

        self.settings = Settings().create_disconnected_proxy()
        self.java_factory = JavaFactory()

        self.settings.settings.inferTemplateFacts = False

        self.parsed_template = template.get_parsed_template(self.settings, self.java_factory)
        self.dataset_builder = DatasetBuilder(self.parsed_template, self.java_factory)

        self.examples: List[Union[AtomType, Rule]] = []

        self.grounder = jpype.JClass("cz.cvut.fel.ida.logic.grounding.Grounder").getGrounder(self.settings.settings)
        field = self.grounder.getClass().getDeclaredField("herbrandModel")
        field.setAccessible(True)

        self.herbrand_model = field.get(self.grounder)

        self.examples_builder = jpype.JClass("cz.cvut.fel.ida.logic.constructs.building.ExamplesBuilder")
        self.queries_builder = jpype.JClass("cz.cvut.fel.ida.logic.constructs.building.QueriesBuilder")
        self.grounding_sample = jpype.JClass("cz.cvut.fel.ida.logic.grounding.GroundingSample")
        self.horn_clause = jpype.JClass("cz.cvut.fel.ida.logic.HornClause")

        self.empty_example = jpype.JClass("cz.cvut.fel.ida.logic.constructs.example.LiftedExample")()

    def set_knowledge(self, examples: List[Union[AtomType, Rule]]) -> None:
        self.examples = examples

    def q(self, query: AtomType, examples: Optional[List[Union[AtomType, Rule]]] = None):
        return self.query(query, examples)

    def query(self, query: AtomType, examples: Optional[List[Union[AtomType, Rule]]] = None):
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
        horn_clause = self.horn_clause(clause)
        substitutions = self.herbrand_model.groundingSubstitutions(horn_clause)

        labels = list(substitutions.r)
        substitutions_sets = list(substitutions.s)

        def generator():
            for substitution_set in substitutions_sets:
                yield {str(label): str(substitution) for label, substitution in zip(labels, substitution_set)}

        if len(substitutions_sets) == 0:
            return {}
        return generator()

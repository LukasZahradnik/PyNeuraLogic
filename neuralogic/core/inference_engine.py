from typing import List, Union, Dict, Generator, Optional

from py4j.java_collections import ListConverter
from py4j.java_gateway import get_field, set_field

from neuralogic import get_neuralogic, get_gateway
from neuralogic.core import Template, JavaFactory, Settings
from neuralogic.core.builder import DatasetBuilder
from neuralogic.core.constructs.atom import AtomType
from neuralogic.core.constructs.rule import Rule


class InferenceEngine:
    def __init__(self, template: Template):
        self.settings = Settings().create_disconnected_proxy()
        self.java_factory = JavaFactory()

        set_field(self.settings.settings, "inferTemplateFacts", False)

        self.parsed_template = template.get_parsed_template(self.settings, self.java_factory)
        self.dataset_builder = DatasetBuilder(self.parsed_template, self.java_factory)

        self.examples: List[Union[AtomType, Rule]] = []

        self.grounder = get_neuralogic().cz.cvut.fel.ida.logic.grounding.Grounder.getGrounder(self.settings.settings)
        field = self.grounder.getClass().getDeclaredField("herbrandModel")
        field.setAccessible(True)

        self.herbrand_model = field.get(self.grounder)
        self.empty_example = get_neuralogic().cz.cvut.fel.ida.logic.constructs.example.LiftedExample()

    def set_knowledge(self, examples: List[Union[AtomType, Rule]]) -> None:
        self.examples = examples

    def q(self, query: AtomType, examples: Optional[List[Union[AtomType, Rule]]] = None):
        return self.query(query, examples)

    # -> Generator[Dict[str, Union[float, int, str]]]

    def query(self, query: AtomType, examples: Optional[List[Union[AtomType, Rule]]] = None):
        if examples is None:
            examples = self.examples

        namespace = get_neuralogic().cz.cvut.fel.ida.logic.constructs.building
        examples_builder = namespace.ExamplesBuilder(self.settings.settings)
        query_builder = namespace.QueriesBuilder(self.settings.settings)
        query_builder.setFactoriesFrom(examples_builder)

        self.java_factory.weight_factory = self.java_factory.get_new_weight_factory()
        queries = self.dataset_builder.build_queries([query], query_builder)

        self.java_factory.weight_factory = self.java_factory.get_new_weight_factory()
        examples = ListConverter().convert(
            self.dataset_builder.build_examples([examples], examples_builder), get_gateway()._gateway_client
        )

        logic_samples = DatasetBuilder.merge_queries_with_examples(queries, examples)
        logic_samples = ListConverter().convert(logic_samples, get_gateway()._gateway_client)

        sample = logic_samples[0]

        gs = get_neuralogic().cz.cvut.fel.ida.logic.grounding.GroundingSample(sample, self.parsed_template)

        lifted_example = get_field(get_field(gs, "query"), "evidence")
        template = get_field(gs, "template")

        ground_template = self.grounder.groundRulesAndFacts(lifted_example, template)

        clause = self.java_factory.atom_to_clause(query)
        horn_clause = get_neuralogic().cz.cvut.fel.ida.logic.HornClause(clause)
        substitutions = self.herbrand_model.groundingSubstitutions(horn_clause)

        labels = list(get_field(substitutions, "r"))
        substitutions_sets = list(get_field(substitutions, "s"))

        def generator():
            for substitution_set in substitutions_sets:
                yield {str(label): str(substitution) for label, substitution in zip(labels, substitution_set)}

        if len(substitutions_sets) == 0:
            return {}
        return generator()

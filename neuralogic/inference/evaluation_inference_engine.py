from typing import List, Union, Optional

from neuralogic.core import Template, Settings
from neuralogic.core.constructs.relation import BaseRelation
from neuralogic.core.constructs.rule import Rule

from neuralogic.dataset import Dataset


class EvaluationInferenceEngine:
    def __init__(self, template: Template):
        self.settings = Settings()
        self.model = template.build(self.settings)

        self.examples: List[Union[BaseRelation, Rule]] = []
        self.dataset = Dataset()
        self.dataset.examples = [[]]

    def set_knowledge(self, examples: List[Union[BaseRelation, Rule]]) -> None:
        self.dataset.examples = [examples]

    def q(self, query: BaseRelation, examples: Optional[List[Union[BaseRelation, Rule]]] = None):
        return self.query(query, examples)

    def query(self, query: BaseRelation, examples: Optional[List[Union[BaseRelation, Rule]]] = None):
        global_examples = self.dataset.examples

        if examples is not None:
            self.dataset.examples = [examples]

        self.dataset.queries = [query]
        variables = [(name, index) for index, name in enumerate(query.terms) if str(name).isupper()]

        try:
            built_dataset = self.model.build_dataset(self.dataset)
            results = self.model(built_dataset.samples, train=False)
        except Exception:
            self.dataset.examples = global_examples
            return {}

        self.dataset.examples = global_examples

        if len(built_dataset.samples) != len(results):
            raise Exception

        def generator():
            for result, sample in zip(results, built_dataset.samples):
                sub_query = str(sample.java_sample.query.neuron.getName())
                sub_query = sub_query.split("(")[1].strip()[:-1]

                substitutions = sub_query.split(",")
                yield result, {label: substitutions[position].strip() for label, position in variables}

        return generator()

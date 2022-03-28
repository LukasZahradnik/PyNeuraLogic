from typing import List, Union, Optional

from neuralogic.core import Template, Settings, Backend, Dataset
from neuralogic.core.constructs.atom import AtomType
from neuralogic.core.constructs.rule import Rule


class EvaluationInferenceEngine:
    def __init__(self, template: Template):
        self.settings = Settings()
        self.model = template.build(Backend.JAVA, self.settings)

        self.examples: List[Union[AtomType, Rule]] = []
        self.dataset = Dataset()
        self.dataset.examples = [[]]

    def set_knowledge(self, examples: List[Union[AtomType, Rule]]) -> None:
        self.dataset.examples = [examples]

    def q(self, query: AtomType, examples: Optional[List[Union[AtomType, Rule]]] = None):
        return self.query(query, examples)

    def query(self, query: AtomType, examples: Optional[List[Union[AtomType, Rule]]] = None):
        global_examples = self.dataset.examples

        if examples is not None:
            self.dataset.examples = [examples]

        self.dataset.queries = [query]
        variables = [(name, index) for index, name in enumerate(query.terms) if str(name).isupper()]

        try:
            built_dataset = self.model.build_dataset(self.dataset)
            results = self.model(built_dataset.samples, train=False)
        except Exception as e:
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
                yield result[1], {label: substitutions[position].strip() for label, position in variables}

        return generator()

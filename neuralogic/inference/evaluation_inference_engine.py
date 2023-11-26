from typing import List, Union, Optional

from neuralogic.core import Template, Settings
from neuralogic.core.constructs.relation import BaseRelation
from neuralogic.core.constructs.rule import Rule

from neuralogic.dataset import Dataset, Sample


class EvaluationInferenceEngine:
    def __init__(self, template: Template, settings: Settings = None):
        self.settings = Settings() if settings is None else settings
        self.model = template.build(self.settings)
        self.dataset = Dataset(Sample(None, None))

    def set_knowledge(self, examples: List[Union[BaseRelation, Rule]]) -> None:
        self.dataset[0].example = examples

    def q(self, query: BaseRelation, examples: Optional[List[Union[BaseRelation, Rule]]] = None):
        return self.query(query, examples)

    def query(self, query: BaseRelation, examples: Optional[List[Union[BaseRelation, Rule]]] = None):
        global_examples = self.dataset[0].example

        if examples is not None:
            self.dataset[0].example = examples

        self.dataset[0].query = query
        variables = [(name, index) for index, name in enumerate(query.terms) if str(name).isupper()]

        try:
            built_dataset = self.model.build_dataset(self.dataset)
            results = self.model(built_dataset.samples, train=False)
        except Exception:
            self.dataset[0].example = global_examples
            return {}

        self.dataset[0].example = global_examples

        if len(built_dataset.samples) != len(results):
            raise Exception

        def generator():
            for result, sample in zip(results, built_dataset.samples):
                sub_query = str(sample.java_sample.query.neuron.getName())
                sub_query = sub_query.split("(")[1].strip()[:-1]

                substitutions = sub_query.split(",")
                yield result, {label: substitutions[position].strip() for label, position in variables}

        return generator()

from typing import Union, List
from neuralogic.model.atom import BaseAtom, WeightedAtom
from neuralogic.model.rule import Rule


TemplateEntries = Union[BaseAtom, WeightedAtom, Rule]


class Template:
    def __init__(self):
        self.template: List[TemplateEntries] = []
        self.examples: List[TemplateEntries] = []
        self.queries: List[TemplateEntries] = []

    def add_rule(self, rule: TemplateEntries):
        self.add_rules([rule])

    def add_rules(self, rules: List[TemplateEntries]):
        self.template.extend(rules)

    def add_example(self, example: TemplateEntries):
        self.add_examples([example])

    def add_examples(self, examples: List[TemplateEntries]):
        self.examples.extend(examples)

    def add_query(self, query: TemplateEntries):
        self.add_queries([query])

    def add_queries(self, queries: List[TemplateEntries]):
        self.queries.extend(queries)

    def build(self):
        pass

    def __str__(self):
        return "\n".join(str(r) for r in self.template)

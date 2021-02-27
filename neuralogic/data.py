from typing import Optional, Union, List
from pathlib import Path
import os
from antlr4 import InputStream, CommonTokenStream
from neuralogic.grammar import NeuralogicLexer, NeuralogicParser
from neuralogic import neuralogic_jvm
from neuralogic.builder import Weight, Sample, Model
from neuralogic.settings import Settings
from neuralogic.sources import Sources
from neuralogic.error import DatasetAlreadyLoadedException, InvalidRuleException


PathType = Optional[Union[Path, str]]


class Dataset:
    def __init__(
        self,
        source_dir: PathType = None,
        template: PathType = None,
        examples: PathType = None,
        queries: PathType = None,
    ):
        self.source_dir = source_dir

        self.template = template
        self.examples = examples
        self.queries = queries

        self.template_list: List[str] = []
        self.examples_list: List[str] = []
        self.queries_list: List[str] = []

        self.loaded = False
        self.__weights: List[Weight] = []
        self.__samples: List[Sample] = []

        self.settings: Optional[Settings] = None
        self.sources: Optional[Sources] = None

    def load(self, args: Optional[List] = None):
        self.loaded = True

        args = [] if args is None else args

        if self.source_dir is not None:
            args.extend(["-sd", str(self.source_dir)])
        if self.template is not None:
            args.extend(["-t", str(self.template)])
        if self.queries is not None:
            args.extend(["-q", str(self.queries)])
        if self.examples is not None:
            args.extend(["-e", str(self.examples)])

        with neuralogic_jvm():
            settings = Settings()
            sources = Sources.from_args(args, settings)
            weights, samples = Model.from_neuralogic(settings, sources)

        self.__weights = weights
        self.__samples = samples

    def add_rules(self, rules: List[str]) -> "Dataset":
        if self.loaded:
            raise DatasetAlreadyLoadedException()
        if all(Dataset.validate_template(rule) for rule in rules):
            self.template_list.extend(rules)
            return self
        raise InvalidRuleException()

    def add_queries(self, queries: List[str]) -> "Dataset":
        if self.loaded:
            raise DatasetAlreadyLoadedException()
        if all(Dataset.validate_query(rule) for rule in queries):
            self.queries_list.extend(queries)
            return self
        raise InvalidRuleException()

    def add_examples(self, examples: List[str]) -> "Dataset":
        if self.loaded:
            raise DatasetAlreadyLoadedException()
        if all(Dataset.validate_query(rule) for rule in examples):
            self.examples_list.extend(examples)
            return self
        raise InvalidRuleException()

    @property
    def samples(self) -> List[Sample]:
        if not self.loaded:
            self.load()
        return self.__samples

    @property
    def weights(self) -> List[Weight]:
        if not self.loaded:
            self.load()
        return self.__weights

    @staticmethod
    def get_parser(input: str) -> NeuralogicParser:
        lexer = NeuralogicLexer(InputStream(input))
        stream = CommonTokenStream(lexer)
        return NeuralogicParser(stream)

    @staticmethod
    def validate_example(rule: str) -> bool:
        parser = Dataset.get_parser(rule)
        parser.examplesFile()

        return parser.getNumberOfSyntaxErrors() == 0

    @staticmethod
    def validate_template(rule: str) -> bool:
        parser = Dataset.get_parser(rule)
        parser.templateFile()

        return parser.getNumberOfSyntaxErrors() == 0

    @staticmethod
    def validate_query(rule: str) -> bool:
        parser = Dataset.get_parser(rule)
        parser.queriesFile()

        return parser.getNumberOfSyntaxErrors() == 0


base_path = os.path.abspath(os.path.dirname(__file__))

XOR = Dataset(source_dir=os.path.join(base_path, "..", "dataset", "simple", "xor", "naive"))
Mutagenesis = Dataset(source_dir=os.path.join(base_path, "..", "dataset", "molecules", "mutagenesis"))

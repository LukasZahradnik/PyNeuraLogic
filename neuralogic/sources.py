from . import get_neuralogic, get_gateway
from .settings import Settings
from .error import InvalidLearningModeException

from enum import Enum

from typing import Union
from pathlib import Path


class LearningMode(Enum):
    CROSSVALIDATION = 0
    TRAIN_TEST = 1
    TRAIN_ONLY = 2
    TEST_ONLY = 3


class Sources:
    @staticmethod
    def from_dir(source_dir: Union[Path, str], settings: Settings) -> "Sources":
        neuralogic = get_neuralogic()
        gateway = get_gateway()

        args = ["-sd", str(source_dir), "-ts", "100"]
        jargs = gateway.new_array(gateway.jvm.java.lang.String, len(args))

        for i, item in enumerate(args):
            jargs[i] = item

        sources = neuralogic.cz.cvut.fel.ida.neuralogic.cli.utils.Runner.getSources(jargs, settings.settings)
        return Sources(sources)

    @staticmethod
    def from_str(text: str, settings: Settings, mode: LearningMode) -> "Sources":
        neuralogic = get_neuralogic()
        reader = neuralogic.java.io.StringReader(text)

        sources = neuralogic.cz.cvut.fel.ida.setup.Sources(settings.settings)
        # sources.setTemplateReader(reader)

        if mode == LearningMode.CROSSVALIDATION:
            sources.crossvalidation = True
        elif mode == LearningMode.TRAIN_TEST:
            sources.trainTest = True
        elif mode == LearningMode.TRAIN_ONLY:
            sources.trainOnly = True
        elif mode == LearningMode.TEST_ONLY:
            sources.testOnly = True
        else:
            raise InvalidLearningModeException()
        return Sources(sources)

    def __init__(self, sources):
        self.sources = sources

    def to_json(self) -> str:
        return self.sources.exportToJson()

    # def __init__(self, settings: Settings, source_dir: Union[Path, str]):
    #     neuralogic = get_neuralogic()
    #     gateway = get_gateway()
    #
    #     args = ["-sd", str(source_dir), "-ts", "100"]
    #     jargs = gateway.new_array(gateway.jvm.java.lang.String, len(args))
    #
    #     for i, item in enumerate(args):
    #         jargs[i] = item
    #
    #     self.sources = (
    #         neuralogic.cz.cvut.fel.ida.neuralogic.cli.utils.Runner.getSources(
    #             jargs, settings.settings
    #         )
    #     )

from typing import List

import jpype

from neuralogic import is_initialized, initialize
from neuralogic.core.settings import SettingsProxy


class Sources:
    @staticmethod
    def from_settings(settings: SettingsProxy) -> "Sources":
        if not is_initialized():
            initialize()

        sources = jpype.JClass("cz.cvut.fel.ida.setup.Sources")(settings.settings)
        return Sources(sources)

    @staticmethod
    def from_args(args: List[str], settings: SettingsProxy) -> "Sources":
        if not is_initialized():
            initialize()

        runner = jpype.JClass("cz.cvut.fel.ida.neuralogic.cli.utils.Runner")
        sources = runner.getSources(args, settings.settings)
        settings._setup_random_generator()

        return Sources(sources)

    def __init__(self, sources):
        self.sources = sources

    def to_json(self) -> str:
        return self.sources.exportToJson()

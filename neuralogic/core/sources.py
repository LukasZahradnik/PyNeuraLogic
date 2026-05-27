from typing import Any

import jpype

from neuralogic.core.settings import SettingsProxy
from neuralogic.setup import initialize, is_initialized


class Sources:
    """
    Represents the logic sources (models, examples, queries) for the NeuraLogic backend.
    """

    @staticmethod
    def from_settings(settings: SettingsProxy) -> "Sources":
        """
        Creates Sources from the provided settings.

        Parameters
        ----------
        settings : SettingsProxy
            The settings proxy.

        Returns
        -------
        Sources
            The created Sources object.
        """
        if not is_initialized():
            initialize()

        sources = jpype.JClass("cz.cvut.fel.ida.setup.Sources")(settings.settings)
        return Sources(sources)

    @staticmethod
    def from_args(args: list[str], settings: SettingsProxy) -> "Sources":
        """
        Creates Sources from command line arguments and settings.

        Parameters
        ----------
        args : List[str]
            The command line arguments.
        settings : SettingsProxy
            The settings proxy.

        Returns
        -------
        Sources
            The created Sources object.
        """
        if not is_initialized():
            initialize()

        runner = jpype.JClass("cz.cvut.fel.ida.neuralogic.cli.utils.Runner")
        sources = runner.getSources(args, settings.settings)
        settings._setup_random_generator()

        return Sources(sources)

    def __init__(self, sources: Any):
        self.sources = sources

    def to_json(self) -> str:
        """
        Exports the sources to a JSON string.

        Returns
        -------
        str
            The JSON representation of the sources.
        """
        return self.sources.exportToJson()

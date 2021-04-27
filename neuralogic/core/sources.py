from neuralogic import get_neuralogic, get_gateway
from neuralogic.core.settings import Settings
from typing import List


class Sources:
    @staticmethod
    def from_settings(settings: Settings) -> "Sources":
        neuralogic = get_neuralogic()
        sources = neuralogic.cz.cvut.fel.ida.setup.Sources(settings.settings)
        return Sources(sources)

    @staticmethod
    def from_args(args: List[str], settings: Settings) -> "Sources":
        neuralogic = get_neuralogic()
        gateway = get_gateway()

        jargs = gateway.new_array(gateway.jvm.java.lang.String, len(args))

        for i, item in enumerate(args):
            jargs[i] = item

        sources = neuralogic.cz.cvut.fel.ida.neuralogic.cli.utils.Runner.getSources(jargs, settings.settings)
        return Sources(sources)

    def __init__(self, sources):
        self.sources = sources

    def to_json(self) -> str:
        return self.sources.exportToJson()

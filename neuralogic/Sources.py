from . import get_neuralogic, get_gateway
from .Settings import Settings

from typing import Union
from pathlib import Path


class Sources:
    def __init__(self, settings: Settings, source_dir: Union[Path, str]):
        neuralogic = get_neuralogic()
        gateway = get_gateway()

        args = ["-sd", str(source_dir), "-ts", "100"]
        jargs = gateway.new_array(gateway.jvm.java.lang.String, len(args))

        for i, item in enumerate(args):
            jargs[i] = item

        self.sources = (
            neuralogic.cz.cvut.fel.ida.neuralogic.cli.utils.Runner.getSources(
                jargs, settings.settings
            )
        )

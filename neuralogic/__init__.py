import os
from typing import Optional, List

import jpype

from neuralogic.logging import _init_logging


_is_initialized = False
_seed = int.from_bytes(os.urandom(4), byteorder="big")
_initial_seed = _seed
_rnd_generator = None

jvm_params = {
    "classpath": os.path.join(os.path.abspath(os.path.dirname(__file__)), "jar", "NeuraLogic.jar"),
}

jvm_options = ["-Xms1g", "-Xmx64g"]


def initial_seed() -> int:
    """Returns the initial random seed for a random number generator used in the backend"""
    return _initial_seed


def seed() -> int:
    """Sets the seed for a random number generator used in the backend to a random seed and returns the seed."""
    global _seed

    _seed = int.from_bytes(os.urandom(4), byteorder="big")

    if _rnd_generator is not None:
        _rnd_generator.setSeed(_seed)
    return _seed


def manual_seed(seed: int):
    """
    Sets the seed for a random number generator used in the backend to the passed ``seed``.

    :param seed:
    """
    global _seed

    _seed = seed

    if _rnd_generator is not None:
        _rnd_generator.setSeed(_seed)


def set_jvm_options(options: List[str]) -> None:
    """
    Set the jvm options - by default ["-Xms1g", "-Xmx64g"],
    """
    global jvm_options
    jvm_options = options


def set_jvm_path(path: Optional[str]) -> None:
    global jvm_params

    if path is None:
        jvm_params.pop("jvmpath", None)
    else:
        jvm_params["jvmpath"] = path


def is_initialized() -> bool:
    return _is_initialized


def initialize(
    debug_mode: bool = False, debug_port: int = 12999, is_debug_server: bool = True, debug_suspend: bool = True
):
    global _is_initialized

    if _is_initialized:
        raise Exception("NeuraLogic already initialized")

    _is_initialized = True
    if debug_mode:
        port = int(debug_port)
        server = "y" if is_debug_server else "n"
        suspend = "y" if debug_suspend else "n"

        debug_params = [
            "-Xint",
            "-Xdebug",
            "-Xnoagent",
            f"-Xrunjdwp:transport=dt_socket,server={server},address={port},suspend={suspend}",
        ]

        jpype.startJVM(*jvm_options, *debug_params, **jvm_params)
    else:
        jpype.startJVM(*jvm_options, **jvm_params)
    _init_logging()

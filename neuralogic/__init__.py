import os
from typing import Optional, List

import jpype

from neuralogic.logging import _init_logging


_is_initialized = False
_seed = int.from_bytes(os.urandom(4), byteorder="big")
_initial_seed = _seed
_rnd_generator = None
_max_memory_size = None

jvm_params = {
    "classpath": os.path.join(os.path.abspath(os.path.dirname(__file__)), "jar", "NeuraLogic.jar"),
}

jvm_options = ["-Xms1g"]


def set_max_memory_size(size: int):
    """
    Set maximum memory size that can be utilized by the backend (in gigabytes)

    Parameters
    ----------
    size : int
        The maximum memory size (in gigabytes)
    """
    global _max_memory_size
    _max_memory_size = size


def initial_seed() -> int:
    """
    Returns the initial/current random seed for a random number generator used in the backend.
    """
    return _initial_seed


def seed() -> int:
    """
    Sets the seed for a random number generator used in the backend to a random seed and returns the seed.
    """
    global _seed

    _seed = int.from_bytes(os.urandom(4), byteorder="big")

    if _rnd_generator is not None:
        _rnd_generator.setSeed(_seed)
    return _seed


def manual_seed(seed: int):
    """
    Sets the seed for a random number generator used in the backend to the passed ``seed``.

    Parameters
    ----------
    seed : int
        The seed for the random number generator.
    """
    global _seed

    _seed = seed

    if _rnd_generator is not None:
        _rnd_generator.setSeed(_seed)


def set_jvm_options(options: List[str]):
    """
    Set the jvm options - by default ``["-Xms1g"]``.

    Parameters
    ----------
    options : List[str]
        List of JVM options
    """
    global jvm_options
    jvm_options = options


def set_jvm_path(path: Optional[str]):
    """
    Set the JVM path.

    Parameters
    ----------
    path : Optional[str]
        The JVM path
    """
    global jvm_params

    if path is None:
        jvm_params.pop("jvmpath", None)
    else:
        jvm_params["jvmpath"] = path


def is_initialized() -> bool:
    """
    Check whether the NeuraLogic backend has been initialized
    """
    return _is_initialized


def initialize(
    debug_mode: bool = False, debug_port: int = 12999, is_debug_server: bool = True, debug_suspend: bool = True
):
    """
    Initialize the NeuraLogic backend. This function is called implicitly when needed and should be called
    manually only for debugging.

    Parameters
    ----------
    debug_mode : bool
        Enable/Disable JVM debug mode.
    debug_port : int
        Port for the debugger to listen on. Default: ``12999``.
    is_debug_server : bool
        Act like server and listen for the debugger. Default: ``True``
    debug_suspend : bool
        Wait until the debugger is connected. Default: ``True``
    """
    global _is_initialized

    if _is_initialized:
        raise Exception("NeuraLogic already initialized")

    _is_initialized = True
    options = [*jvm_options]

    if _max_memory_size is not None:
        options.append(f"-Xmx{_max_memory_size}g")

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

        jpype.startJVM(*options, *debug_params, **jvm_params)
    else:
        jpype.startJVM(*options, **jvm_params)
    _init_logging()

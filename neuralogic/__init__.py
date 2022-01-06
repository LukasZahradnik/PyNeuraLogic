import os

import jpype


os.environ["CLASSPATH"] = os.path.join(os.path.abspath(os.path.dirname(__file__)), "jar", "NeuraLogic.jar")
_is_initialized = False


def is_initialized() -> bool:
    return _is_initialized


def initialize(
    debug_mode: bool = False, debug_port: int = 12999, is_debug_server: bool = True, debug_suspend: bool = False
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

        jpype.startJVM(*debug_params, classpath=[os.environ["CLASSPATH"]])
    else:
        jpype.startJVM(classpath=[os.environ["CLASSPATH"]])

    jpype.java.lang.System.setOut(jpype.java.io.PrintStream(jpype.java.io.File("/dev/null")))
    jpype.java.lang.System.setErr(jpype.java.io.PrintStream(jpype.java.io.File("/dev/null")))

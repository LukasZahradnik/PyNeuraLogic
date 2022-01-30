import os
from typing import Optional

import jpype


_is_initialized = False
_std_out = None
_std_err = None

jvm_params = {
    "classpath": os.path.join(os.path.abspath(os.path.dirname(__file__)), "jar", "NeuraLogic.jar"),
}


class TextIOWrapper:
    def __init__(self, wrapped_text_io):
        self.wrapped_text_io = wrapped_text_io

    def write(self, string):
        self.wrapped_text_io.write(str(string))


def set_system_output(output, system_output_setter) -> None:
    java_output_stream = os.devnull

    if output is not None:
        wrapped = TextIOWrapper(output)

        java_io_wrapper = jpype.JProxy("cz.cvut.fel.ida.utils.python.PythonOutputStream.TextIOWrapper", inst=wrapped)
        java_output_stream = jpype.JClass("cz.cvut.fel.ida.utils.python.PythonOutputStream")(java_io_wrapper)
    system_output_setter(jpype.java.io.PrintStream(java_output_stream))


def set_stdout(out_io=None) -> None:
    global _std_out
    _std_out = out_io

    if not _is_initialized:
        return

    set_system_output(_std_out, jpype.java.lang.System.setOut)


def set_stderr(err_io=None) -> None:
    global _std_err
    _std_err = err_io

    if not _is_initialized:
        return

    set_system_output(_std_err, jpype.java.lang.System.setErr)


def set_jvm_path(path: Optional[str]) -> None:
    global jvm_params

    if path is None:
        jvm_params.pop("jvmpath", None)
    else:
        jvm_params["jvmpath"] = path


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

        jpype.startJVM(*debug_params, **jvm_params)
    else:
        jpype.startJVM(**jvm_params)

    set_stderr(_std_err)
    set_stdout(_std_out)

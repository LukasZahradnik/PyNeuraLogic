import os
from enum import Enum

import jpype


_is_logging_initialized = False
_loggers_buffer = []


class TextIOWrapper:
    def __init__(self, wrapped_text_io):
        self.wrapped_text_io = wrapped_text_io

    def write(self, string):
        self.wrapped_text_io.write(str(string))


class Level(Enum):
    """Logging level"""

    OFF = "OFF"
    SEVERE = "SEVERE"
    WARNING = "WARNING"
    INFO = "INFO"
    CONFIG = "CONFIG"
    FINE = "FINE"
    FINER = "FINER"
    FINEST = "FINEST"
    ALL = "ALL"


class Formatter(Enum):
    """Logged information formatters"""

    COLOR = "color"
    NORMAL = "normal"


def _init_logging():
    global _is_logging_initialized

    jpype.java.lang.System.setOut(jpype.java.io.PrintStream(os.devnull))
    jpype.java.lang.System.setErr(jpype.java.io.PrintStream(os.devnull))

    jpype.JClass("cz.cvut.fel.ida.logging.Logging").initLogging(jpype.JClass("cz.cvut.fel.ida.setup.Settings")())
    _is_logging_initialized = True

    for handler_settings in _loggers_buffer:
        add_handler(*handler_settings)
    _loggers_buffer.clear()


def add_handler(output, level: Level = Level.FINER, formatter: Formatter = Formatter.COLOR):
    """
    Add logger handler for an insight into the java backend

    :param output: File-like object (has write(text: str) method)
    :param level: The logging level
    :param formatter: The log formatter
    """
    if not _is_logging_initialized:
        _loggers_buffer.append((output, level, formatter))
        return

    root_logger = jpype.java.util.logging.Logger.getLogger("")
    wrapped = TextIOWrapper(output)

    java_io_wrapper = jpype.JProxy("cz.cvut.fel.ida.utils.python.PythonOutputStream.TextIOWrapper", inst=wrapped)
    java_output_stream = jpype.JClass("cz.cvut.fel.ida.utils.python.PythonOutputStream")(java_io_wrapper)

    if formatter == Formatter.COLOR:
        java_formatter = jpype.JClass("cz.cvut.fel.ida.logging.ColoredFormatter")()
    elif formatter == Formatter.NORMAL:
        java_formatter = jpype.JClass("cz.cvut.fel.ida.logging.NormalFormatter")()
    else:
        raise NotImplementedError(f"Unknown formatter {formatter}")

    print_stream = jpype.java.io.PrintStream(java_output_stream)
    stream_handler = jpype.JClass("cz.cvut.fel.ida.logging.FlushStreamHandler")(print_stream, java_formatter)
    stream_handler.setLevel(getattr(jpype.java.util.logging.Level, str(level.value)))

    root_logger.addHandler(stream_handler)


def clear_handlers():
    """Clear all handlers"""
    if not _is_logging_initialized:
        _loggers_buffer.clear()
        return

    root_logger = jpype.java.util.logging.Logger.getLogger("")

    for handler in root_logger.getHandlers():
        root_logger.removeHandler(handler)

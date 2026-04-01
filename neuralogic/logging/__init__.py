import dataclasses
import os
import sys
from enum import Enum
from typing import Any

import jpype


_is_logging_initialized = False
_default_logging = True
_loggers_buffer: list[tuple] = []


class TextIOWrapper:
    """
    TextIOWrapper is a wrapper for text IO objects to ensure strings are written correctly.
    """
    def __init__(self, wrapped_text_io: Any):
        """
        Parameters
        ----------
        wrapped_text_io : Any
            The text IO object to wrap.
        """
        self.wrapped_text_io = wrapped_text_io

    def write(self, string: Any) -> None:
        self.wrapped_text_io.write(str(string))


class Level(Enum):
    """
    Logging level enum.
    """

    OFF = 2147483647
    SEVERE = 1000
    WARNING = 900
    INFO = 800
    CONFIG = 700
    FINE = 500
    FINER = 400
    FINEST = 300
    ALL = -2147483648


class Formatter(Enum):
    """
    Logged information formatters enum.
    """

    COLOR = "color"
    NORMAL = "normal"


@dataclasses.dataclass
class LogHandler:
    """
    LogHandler stores settings for a log handler.
    """
    output: Any
    level: Level = Level.FINER
    formatter: Formatter = Formatter.COLOR


def _init_logging() -> None:
    global _is_logging_initialized

    jpype.java.lang.System.setOut(jpype.java.io.PrintStream(os.devnull))
    jpype.java.lang.System.setErr(jpype.java.io.PrintStream(os.devnull))

    settings_class = jpype.JClass("cz.cvut.fel.ida.setup.Settings")
    settings = settings_class()
    settings.supressConsoleOutput = True
    settings.supressLogFileOutput = True
    settings.loggingLevel = jpype.JClass("java.util.logging.Level").OFF

    jpype.JClass("cz.cvut.fel.ida.logging.Logging").initLogging(settings)
    _is_logging_initialized = True

    if _default_logging:
        add_handler(sys.stdout, Level.INFO)

    for handler_settings in _loggers_buffer:
        add_handler(*handler_settings)
    _loggers_buffer.clear()


def add_log_handler(handler: LogHandler) -> None:
    """
    Add logger handler for an insight into the java backend. Overrides the default logger to stdout.

    Parameters
    ----------
    handler : LogHandler
        The log handler to add.
    """
    return add_handler(handler.output, handler.level, handler.formatter)


def add_handler(output: Any, level: Level = Level.FINER, formatter: Formatter = Formatter.COLOR) -> None:
    """
    Add logger handler for an insight into the java backend. Overrides the default logger to stdout.

    Parameters
    ----------
    output : Any
        File-like object (has ``write(text: str)`` method)
    level : Level
        The logging level. Default: Level.FINER.
    formatter : Formatter
        The log formatter. Default: Formatter.COLOR.
    """
    global _default_logging
    _default_logging = False

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
    stream_handler.setLevel(getattr(jpype.java.util.logging.Level, str(level.name)))

    if level.value < root_logger.getLevel().intValue():
        root_logger.setLevel(getattr(jpype.java.util.logging.Level, str(level.name)))

    root_logger.addHandler(stream_handler)


def clear_handlers() -> None:
    """Removes all log handlers."""
    global _default_logging
    _default_logging = False

    if not _is_logging_initialized:
        _loggers_buffer.clear()
        return

    root_logger = jpype.java.util.logging.Logger.getLogger("")

    for handler in root_logger.getHandlers():
        root_logger.removeHandler(handler)


__all__ = ["add_handler", "add_log_handler", "clear_handlers", "Level", "Formatter", "TextIOWrapper", "LogHandler"]

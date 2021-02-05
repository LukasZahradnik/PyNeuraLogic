from typing import Union
from pathlib import Path
import sys
import os
from py4j.java_gateway import JavaGateway

neuralogic = None
gateway = None


def set_environment(java_home: Union[Path, str], class_path: Union[Path, str]):
    """
    Set class path and java home environment variables

    :param java_home:
    :param class_path:
    :return:
    """
    os.environ["JAVA_HOME"] = java_home
    os.environ["CLASS_PATH"] = class_path


def initialize(path: Union[Path, str], std_out=sys.stdout, std_err=sys.stderr, die_on_exit=True):
    global gateway, neuralogic

    gateway = JavaGateway.launch_gateway(
        classpath=path,
        redirect_stdout=std_out,
        redirect_stderr=std_err,
        die_on_exit=die_on_exit,
    )
    neuralogic = gateway.jvm


def get_gateway():
    return gateway


def get_neuralogic():
    return neuralogic

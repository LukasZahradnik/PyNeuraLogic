from typing import Union, Optional
from pathlib import Path
import sys
import os
from py4j.java_gateway import JavaGateway, JVMView

neuralogic: Optional[JVMView] = None
gateway: Optional[JavaGateway] = None


def set_java_home(java_home: Union[Path, str]):
    """
    Set java home environment variable

    :param java_home:
    :return:
    """
    os.environ["JAVA_HOME"] = str(java_home)


def initialize(std_out=sys.stdout, std_err=sys.stderr, die_on_exit=True):
    """
    Initialize the gateway for the Java process. Has to be reinitialized if some settings (JAVA_HOME) changes

    :param std_out:
    :param std_err:
    :param die_on_exit:
    :return:
    """
    global gateway, neuralogic

    if gateway is not None:
        gateway.shutdown()

    gateway = JavaGateway.launch_gateway(
        classpath=os.environ["CLASSPATH"],
        redirect_stdout=std_out,
        redirect_stderr=std_err,
        die_on_exit=die_on_exit,
    )

    neuralogic = gateway.jvm


def get_gateway() -> JavaGateway:
    """
    Get the gateway for the Java process

    :return:
    """
    return gateway


def get_neuralogic() -> JVMView:
    """
    Get the jvm view of the Java process

    :return:
    """
    return neuralogic


os.environ["CLASSPATH"] = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "jar", "NeuraLogic.jar")
initialize()

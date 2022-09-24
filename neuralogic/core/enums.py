from enum import Enum


class Backend(Enum):
    DYNET = "dynet"
    JAVA = "java"
    TORCH = "torch"


class Grounder(Enum):
    BUP = "BUP"
    GRINGO = "GRINGO"

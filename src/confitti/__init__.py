from importlib.metadata import version
from .confitti import *

__version__ = version("confitti")


def hello() -> str:
    return "Hello from confitti🎉🎊!"

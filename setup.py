from setuptools import setup
import sys

setup(
    name="pyneuralogic",
    version=sys.argv[-1],
    packages=["neuralogic", "neuralogic.error", "neuralogic.grammar"],
    url="https://github.com/LukasZahradnik/PyNeuraLogic",
    license="MIT",
    author="Lukáš Zahradník",
    author_email="lukaszahradnik96@seznam.cz",
    description="PyNeuraLogic is a framework which combines relational and deep learning.",
)

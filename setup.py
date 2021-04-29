#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os

from typing import Dict
from setuptools import find_packages, setup

# Package meta-data.
NAME = "neuralogic"
DESCRIPTION = "PyNeuraLogic is a framework which combines relational and deep learning."
URL = "https://github.com/LukasZahradnik/PyNeuraLogic"
EMAIL = "lukaszahradnik96@seznam.cz"
AUTHOR = "Lukáš Zahradník"
REQUIRES_PYTHON = ">=3.7.0"
VERSION = None

REQUIRED = [
    "py4j >= 0.10.9.1",
    "antlr4-python3-runtime >= 4.8",
    "numpy >= 1.19.0",
]

EXTRAS: Dict = {}

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

about: Dict = {}
with open(os.path.join(here, NAME, "__version__.py")) as f:  # type: ignore
    exec(f.read(), about)

setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*", "examples"]),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    package_data={
        "neuralogic.jar": ["NeuraLogic.jar"],
        "neuralogic.utils.data": ["datasets/**/*", "datasets/**/**/*", "datasets/**/**/**/**/*", "datasets/**/**/**/*"],
    },
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

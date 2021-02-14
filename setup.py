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
REQUIRES_PYTHON = ">=3.8.0"
VERSION = None

REQUIRED = [
    "py4j",
    "antlr4-python3-runtime",
]

EXTRAS: Dict = {}

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

about: Dict = {}
if not VERSION:
    with open(os.path.join(here, "neuralogic", "__version__.py")) as f:  # type: ignore
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
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
)

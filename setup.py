#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os

from typing import Dict
from setuptools import find_packages, setup

# Package meta-data.
NAME = "neuralogic"
DESCRIPTION = "PyNeuraLogic lets you use Python to create Differentiable Logic Programs."
URL = "https://github.com/LukasZahradnik/PyNeuraLogic"
EMAIL = "lukaszahradnik96@seznam.cz"
AUTHOR = "Lukáš Zahradník"
REQUIRES_PYTHON = ">=3.7.0"
VERSION = None

REQUIRED = [
    "JPype1 >=1.3.0",
    "numpy >= 1.20.4",
    "matplotlib",
    "tqdm",
]

EXTRAS: Dict = {}

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

about: Dict = {}

try:
    with open(os.path.join(here, NAME, "__version__.py")) as f:  # type: ignore
        exec(f.read(), about)
except Exception:
    about["__version__"] = "0.0.0"  # development

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
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

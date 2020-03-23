#!/usr/bin/env python3
# -*- Coding: UTF-8 -*-
import os
from setuptools import setup, find_packages


def read(filename):
    return open(os.path.join(os.path.dirname(__file__), filename)).read()

setup(
    name="blockmatching",
    license="Apache License 2.0",
    version='1.0.0',
    author='Eduardo S. Pereira',
    author_email='pereira.somoza@gmail.com',
    packages=find_packages("src"),
    package_dir={"":"src"},
    description="Block Matching Algorithm",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/duducosmos/blockmatching",
    install_requires=["numpy",
                      "numba",
                      "opencv-python",
                      "scipy",
                      "Pillow",
                      "networkx",
                      "sk-video"]
)

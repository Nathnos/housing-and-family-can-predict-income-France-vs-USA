#! /usr/bin/env python3
# coding: utf-8

import setuptools

with open("ReadMe.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Housing and Family can predict income",
    version="0.0.1",
    author="Lorenzo VILLARD",
    author_email="villard.lorenzo.pro@pm.me",
    description="A Data Cleaning Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Nathnos/housing-and-family-can-predict-income-France-vs-USA",
    license="GNUv3",
    python_requires=">=3.8.2",
)

try:
    from setuptools import setup

    kw = {"test_suite": "tests"}
except ImportError:
    from distutils.core import setup

    kw = {}

import os

versionfile = os.path.join('beniget', 'version.py')
exec(open(versionfile).read())

setup(
    name="beniget",  # gast, beniget!
    version=__version__,
    packages=["beniget"],
    description="Extract semantic information about static Python code",
    long_description="""
A static analyzer for Python2 and Python3 code.

Beniget provides a static over-approximation of the global and
local definitions inside Python Module/Class/Function.
It can also compute def-use chains from each definition.""",
    author="serge-sans-paille",
    author_email="serge.guelton@telecom-bretagne.eu",
    url="https://github.com/serge-sans-paille/beniget/",
    license="BSD 3-Clause",
    install_requires=open("requirements.txt").read().splitlines(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, !=3.5.*",
    **kw
)

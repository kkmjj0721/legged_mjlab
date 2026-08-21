"""Compatibility entry point for legacy ``python setup.py`` commands.

The package metadata is maintained in ``pyproject.toml`` (PEP 621).  Keeping
this file as a metadata-free shim lets older tooling invoke setuptools without
creating a second dependency declaration.
"""

from setuptools import setup


if __name__ == "__main__":
    setup()

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mjlab==1.6.0",
    "mujoco-warp==3.11.0",
    "scipy==1.17.0"
]

# Installation operation
setup(
    name="legged_mjlab",
    packages=find_packages(include=["legged_mjlab", "legged_mjlab.*"]),
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)

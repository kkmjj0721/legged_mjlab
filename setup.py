from setuptools import find_packages, setup


PROJECT_PACKAGES = find_packages(
    include=["legged_mjlab", "legged_mjlab.*"]
)
LOCAL_RSL_RL_SOURCE = "rsl_rl/rsl_rl"
LOCAL_RSL_RL_SUBPACKAGES = find_packages(where=LOCAL_RSL_RL_SOURCE)
LOCAL_RSL_RL_PACKAGES = ["rsl_rl"] + [
    f"rsl_rl.{name}" for name in LOCAL_RSL_RL_SUBPACKAGES
]

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mjlab==1.6.0",
    "mujoco-warp==3.11.0",
    "scipy==1.17.0"
]

# Installation operation
setup(
    name="legged_mjlab",
    packages=PROJECT_PACKAGES + LOCAL_RSL_RL_PACKAGES,
    package_dir={"rsl_rl": LOCAL_RSL_RL_SOURCE},
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)

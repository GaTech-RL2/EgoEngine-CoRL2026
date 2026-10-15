from setuptools import setup, find_packages

setup(
    name="robosuite",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # list any dependencies you need, e.g.:
        "mujoco>=2.3",
        "numpy",
        "gymnasium",
        # …
    ],
)

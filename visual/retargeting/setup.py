from setuptools import setup, find_packages

setup(
    name="retargeting",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["pybullet", "open3d", "pytorch_kinematics", "loop-rate-limiters", "chumpy", "pyglet==1.5.31"],
)

from setuptools import setup, find_packages

setup(
    name="isaaclab_envs",
    version="0.1.0",
    description="G1 Motion Imitation Environments for Isaac Lab",
    author="TWIST2",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "numpy",
        "pyyaml",
    ],
)

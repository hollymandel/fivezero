from setuptools import find_packages, setup


setup(
    name="fivezero",
    version="0.1.0",
    description="AlphaZero-style training utilities for a 5x5 connect game",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "torch",
    ],
)

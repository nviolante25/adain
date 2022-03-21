from setuptools import setup

setup(
    name="adain",
    version="0.1",
    packages=["adain"],
    install_requires=[
        "numpy"
    ],
    extras_require={
        "dev": ["black"],
    },
)
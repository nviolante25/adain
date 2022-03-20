from setuptools import setup

setup(
    name="adain",
    version="0.1",
    packages=["src"],
    install_requires=[
        "numpy"
    ],
    extras_require={
        "dev": ["black"],
    },
)
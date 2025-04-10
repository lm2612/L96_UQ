from setuptools import setup, find_packages

setup(
    name="your_package",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        # add other dependencies
    ],
    extras_require={
        "dev": [
            "pytest",
            # add other development dependencies
        ],
    },
) 
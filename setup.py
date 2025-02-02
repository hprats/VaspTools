from setuptools import setup, find_packages

setup(
    name="vasptools",
    version="0.1.0",
    packages=find_packages(),  # automatically finds python packages inside
    install_requires=[
        "ase",
        "pymatgen",
    ],
)

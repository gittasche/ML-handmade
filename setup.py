from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = "0.1.0"

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mlhandmade",
    version=__version__,
    description="Implemented some ML routines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="gittasche",
    python_reqires=">=3.9",
    setup_requires=[
        "numpy>=1.22.3",
        "scipy>=1.8.0",
        "matplotlib>=3.5.1"
    ],
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False
)
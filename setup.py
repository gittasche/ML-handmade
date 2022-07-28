from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = "0.0.13"

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    all_reqs = f.read().split("\n")

install_requires = [req.strip() for req in all_reqs if "git+" not in req]
dependency_links = [req.strip().replace("git+", "") for req in all_reqs if req.startswith("git+")]

setup(
    name="mlhandmade",
    version=__version__,
    description="Implemented some ML routines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="gittasche",
    python_reqires=">=3.9",
    install_requires=install_requires,
    setup_requires=["numpy>=1.22.3", "scipy>=1.8.0", "matplotlib>=3.5.1"],
    dependency_links=dependency_links,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False
)
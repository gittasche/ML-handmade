from setuptools import setup, find_packages

setup(
    name="mlhandmade",
    version="0.0.4",
    description="Implemented some ML routines",
    author="gittasche",
    python_reqires=">=3.7",
    install_requires=["numpy", "matplotlib"],
    packages=find_packages(),
    zip_safe=False
)
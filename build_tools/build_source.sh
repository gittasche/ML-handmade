#!/bin/bash

set -e
set -x

cd ../../

python -m venv build_env
source build_env/bin/activate

python -m pip install numpy, scipy, matplotlib
python -m pip install twine

cd ML-handmade/ML-handmade
python setup.py sdist

twine check dist/*.tar.gz
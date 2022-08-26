#!/bin/bash

set -e
set -x

cd ../../

python -m venv build_env
source build_env/bin/activate

ls
cd gittasche/ML-handmade
python -m pip install .[dev]
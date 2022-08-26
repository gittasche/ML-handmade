#!/bin/bash

set -e
set -x

cd ../../

python -m venv build_env
source build_env/bin/activate

cd ML-handmade
ls
python -m pip install .[dev]
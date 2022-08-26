#!/bin/bash

set -e
set -x

cd ../../

python -m venv examples_env
source examples_env/bin/activate

python -m pip install numpy scipy matplotlib jupyter pandas scikit-learn

cd ML-handmade/ML-handmade/examples
for note in *.ipynb
    do jupyter nbconvert --to html --execute $note
    if [ $? != 0 ]
        then echo $note
        exit 1
    fi
done
        
#!/bin/bash

set -e
set -x

cd ../../

python -m env examples_env
source examples_env/bin/activate

python -m pip install jupyter, pandas, scikit-learn

cd gittasche/mlhandmade/examples
for note in *.ipynb do
    jupyter nbconvert --to html --execute $note
    if [ $? != 0 ] then
        echo $note
        exit 1
    fi
done
        
#!/bin/bash
WORK_DIR=$(pwd)
cd Software
python3 -m venv pythia_tutorial
if test -f ./pythia_tutorial/bin/activate; then
    echo "File exists."
    source ./pythia_tutorial/bin/activate
    pip install --upgrade pip
    pip install -r pip_requirements.txt
else
    echo "Virtual Environment could not be created..."
fi
cd $WORK_DIR

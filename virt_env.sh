#!/bin/bash
WORK_DIR=$(pwd)
cd Software
pwd > Software_Path.txt
python3 -m venv pythia_tutorial
source ./pythia_tutorial/bin/activate
pip install --upgrade pip
pip install -r pip_requirements.txt
cd $WORK_DIR

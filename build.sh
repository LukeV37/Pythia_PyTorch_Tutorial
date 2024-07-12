#!/bin/bash
WORK_DIR=$(pwd)
cd Software
pwd > Software_Path.txt

cd pythia8312
export CC=gcc
export CXX=g++
./configure --with-python-config=python3-config | tee config.log
CHECK=$(cat config.log | awk 'END{print $3}')

if [[ "$CHECK" == *"PYTHON"* ]]
then
    echo "Python3 has been sucessfully configured with Pythia!"
    echo "How many cores would you like to use to compile Pythia?"
    read -p "Please enter an integer: " CORES
    make -j$CORES
else
    echo "Python3 has not been configured with Pythia!"
    echo "Please ask your neighbor for help..."
fi
cd ..

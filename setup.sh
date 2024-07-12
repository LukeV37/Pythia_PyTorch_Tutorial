#!/bin/bash
WORK_DIR=$(pwd)
cd Software
pwd > Software_Path.txt

echo "Welcome to the Pythia8 setup script!"
###            Author: Luke Vaughan        ###
echo "Would you like to download pythia source code? (Only for first-time setup)"
read -p "Please Respond [y|n]: " CHECK
if [[ "$CHECK" == "y" ]]; then
    curl -O https://www.pythia.org/download/pythia83/pythia8312.tgz
    tar xfz pythia8312.tgz
    rm pythia8312.tgz
fi

echo "Would you like to compile pythia?"
read -p "Please Respond [y|n]: " CHECK
if [[ "$CHECK" == "y" ]]; then
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
fi

echo "Would you like to setup python virtual env?"
read -p "Please Respond [y|n]: " CHECK
if [[ "$CHECK" == "y" ]]; then
    python3 -m venv pythia_tutorial
    if test -f ./pythia_tutorial/bin/activate; then
        echo "File exists."
        source ./pythia_tutorial/bin/activate
        pip install --upgrade pip
        pip install -r pip_requirements.txt
    else
        echo "Virtual Environment could not be created..."
    fi

fi
echo "Would you like to run a jupyter notebook?"
read -p "Please Respond [y|n]: " CHECK
if [[ "$CHECK" == "y" ]]; then
    ipython3 kernel install --user --name=pythia_tutorial
    cd $WORK_DIR 
    jupyter notebook
fi
cd $WORK_DIR
echo "Setup Complete!"

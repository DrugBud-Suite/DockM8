#!/bin/bash --login

echo "Installing DockM8 environment ...\n"

#Force stop if any link fails
set -e

#Set this directory to be current directory
PROJECT_DIR=$PWD

# creates the conda environment
conda env create --file $PROJECT_DIR/environment.yml --force

# activate the conda env before installing PyTorch Geometric via pip
conda init bash
source activate dockm8

#Specify your CUDA toolkit
CUDA=cpu

python -m pip install torch==1.9.1
python -m pip install torch-scatter==2.1.0 -f https://data.pyg.org/whl/torch-1.9.1+${CUDA}.html
python -m pip install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.9.1+${CUDA}.html
python -m pip install torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.9.1+${CUDA}.html
python -m pip install torch-cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.9.1+${CUDA}.html









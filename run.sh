#!/bin/bash --login

echo "Installing relevant packages for DockM8"

#Force stop if any link fails
set -e

#Set this dir to be current dir
PROJECT_DIR=$PWD

# creates the conda environment
mamba env create --file $PROJECT_DIR/environment.yml --force

# activate the conda env before installing PyTorch Geometric via pip
conda init bash
source activate dockm82

#Specify your CUDA toolkit
CUDA=cpu

python -m pip install torch==1.9.1
python -m pip install torch-scatter==2.1.0 -f https://data.pyg.org/whl/torch-1.9.1+${CUDA}.html
python -m pip install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.9.1+${CUDA}.html
python -m pip install torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.9.1+${CUDA}.html
python -m pip install torch-cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.9.1+${CUDA}.html
#pip install torch-geometric==2.3.1 -f https://data.pyg.org/whl/torch-1.9.1+${CUDA}.html
#python -m pip install torch-geometric
#python -m pip install torch_scatter torch_sparse torch_cluster torch_spline_conv  -f https://data.pyg.org/whl/torch-1.9.1+${CUDA}.html


#python -m pip install torch-scatter==2.1.0 torch-sparse==0.6.12 torch-spline-conv==1.2.1 torch-cluster==1.6.0 torch-geometric==2.0.1 -f https://data.pyg.org/whl/torch-1.9.1+cpu.html







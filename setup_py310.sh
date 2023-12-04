#!/bin/bash

# define the directory SCORCH is cloned into
BASEDIR=$PWD

echo -e """
###############################################################
# Installing DockM8
###############################################################
"""

# install dependencies for xgboost, GWOVina & MGLTools
if [[ "$OSTYPE" == "darwin"* || "$OSTYPE" == "msys"* ]]; then
    # dependencies for mac and windows
    echo -e "\nDockM8 is not compatible with Mac OS or Windows!"
    exit

elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo -e "\nDetected Linux OS!"

fi

echo -e """
###############################################################
# Verifying conda install or installing miniconda3 if not found
###############################################################
"""

# check if conda is installed, and install miniconda3 if not

# if conda is not a recognised command then download and install
if ! command -v conda &> /dev/null; then
    echo -e "\nNo conda found - installing..."
    cd utils && mkdir miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3/miniconda.sh --no-check-certificate
    # install miniconda3
    cd miniconda3 && chmod -x miniconda.sh
    cd $BASEDIR && bash utils/miniconda3/miniconda.sh -b -u -p utils/miniconda3

    # remove the installer
    rm -f utils/miniconda3/miniconda.sh

    # define conda installation paths
    CONDA_PATH="utils/miniconda3/bin/conda"
    CONDA_BASE=$BASEDIR/utils/miniconda3
    CONDA_SH="utils/miniconda3/etc/profile.d/conda.sh"
else
    echo -e "\nFound existing conda install!"
    # if conda not installed then find location of existing installation
    CONDA_PATH=$(which conda)
    CONDA_BASE=$(conda info --base)
    CONDA_SH=$CONDA_BASE/etc/profile.d/conda.sh
fi

echo -e """
###############################################################
# installing the DockM8 conda environment
###############################################################
"""
# source the bash files to enable conda command in the same session
if test -f ~/.bashrc; then
    source ~/.bashrc
fi

if test -f ~/.bash_profile; then
    source ~/.bash_profile
fi

# initiate conda
$CONDA_PATH init bash

# source the conda shell script once initiated
source $CONDA_SH

# configure conda to install environment quickly and silently
$CONDA_PATH config --set auto_activate_base false

# create the conda environment
conda create -n dockm8 python=3.10 -y

conda activate dockm8

conda config --add channels conda-forge

conda install rdkit ipykernel scipy spyrmsd kneed scikit-learn-extra molvs seaborn xgboost openbabel -q -y

pip install pymesh espsim oddt biopandas redo MDAnalysis==2.0.0 prody==2.1.0 dgl Pebble tensorflow meeko chembl_structure_pipeline posebusters streamlit

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install torch_scatter torch_sparse torch_spline_conv torch_cluster torch_geometric

if [[ -f dockm8.py || -d DockM8 ]]; then
    echo -e "\nDockM8 repository found in current folder. Installation finished"
else
    echo -e """
###############################################################
# Downloading DockM8 repository...
###############################################################
"""
    wget https://github.com/Tonylac77/DockM8/main.zip -O DockM8.zip --no-check-certificate
    unzip DockM8.zip
    rm DockM8.zip
    echo -e "\nDockM8 repository downloaded. Installation finished"
fi

echo -e """
###############################################################
# DockM8 installation complete
###############################################################
"""


#!/bin/bash

# define the directory SCORCH is cloned into
BASEDIR=$PWD

###############################################################

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

###############################################################

echo -e """
###############################################################
# Verifying conda install or installing miniconda3 if not found
###############################################################
"""

# check if conda is installed, and install miniconda3 if not

# if conda is not a recognised command then download and install
if ! command -v conda &> /dev/null; then
    
    echo -e "\nNo conda found - installing..."
    mkdir -p $HOME/miniconda3
    cd $HOME/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh --no-check-certificate
    # install miniconda3
    cd miniconda3 && chmod -x miniconda.sh
    cd $BASEDIR && bash $HOME/miniconda3/miniconda.sh -b -u -p $HOME/miniconda3

    # remove the installer
    rm -f $HOME/miniconda3/miniconda.sh

    # define conda installation paths
    CONDA_PATH="$HOME/miniconda3/bin/conda"
    CONDA_BASE=$BASEDIR/$HOME/miniconda3
    CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo -e "\nFound existing conda install!"
    # if conda not installed then find location of existing installation
    CONDA_PATH=$(which conda)
    CONDA_BASE=$(conda info --base)
    CONDA_SH=$CONDA_BASE/etc/profile.d/conda.sh
fi

###############################################################

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
ENV_NAME="dockm8"

conda create -n $ENV_NAME python=3.10 -y

conda activate $ENV_NAME

conda config --add channels conda-forge

conda install rdkit ipykernel scipy spyrmsd kneed scikit-learn-extra molvs seaborn xgboost openbabel docopt -q -y

pip install pymesh espsim oddt biopandas redo MDAnalysis==2.0.0 prody==2.1.0 dgl Pebble tensorflow meeko chembl_structure_pipeline posebusters streamlit

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install torch_scatter torch_sparse torch_spline_conv torch_cluster torch_geometric

###############################################################

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

###############################################################

echo -e """
###############################################################
# Downloading Executables...
###############################################################
"""
if [[ ! -d ./software ]]; then
    mkdir ./software
    cd ./software
else
    cd ./software
fi

if [[ ! -f ./gnina ]]; then
    echo -e "\nDownloading GNINA!"
    wget https://github.com/gnina/gnina/releases/latest/download/gnina --no-check-certificate  -q --show-progress
    chmod +x gnina
fi

if [[ ! -f ./qvina-w ]]; then
    echo -e "\nDownloading QVINA-W!"
    wget https://github.com/QVina/qvina/raw/master/bin/qvina-w --no-check-certificate -q --show-progress
    chmod +x qvina-w
fi

if [[ ! -f ./qvina2.1 ]]; then
    echo -e "\nDownloading QVINA2!"
    wget https://github.com/QVina/qvina/raw/master/bin/qvina2.1 --no-check-certificate -q --show-progress
    chmod +x qvina2.1
fi

if [[ ! -f ./KORP-PL ]]; then
    echo -e "\nDownloading KORP-PL!"
    wget https://files.inria.fr/NanoDFiles/Website/Software/KORP-PL/0.1.2/Linux/KORP-PL-LINUX-v0.1.2.2.tar.gz --no-check-certificate -q --show-progress
    tar -xvf KORP-PL-LINUX-v0.1.2.2.tar.gz
    rm KORP-PL-LINUX-v0.1.2.2.tar.gz
    chmod +x KORP-PL
fi

if [[ ! -f ./Convex-PL ]]; then
    echo -e "\nDownloading Convex-PLR!"
    wget https://files.inria.fr/NanoDFiles/Website/Software/Convex-PL/Files/Convex-PL-Linux-v0.5.tar.zip --no-check-certificate -q --show-progress
    unzip Convex-PL-Linux-v0.5.tar.zip
    tar -xvf Convex-PL-Linux-v0.5.tar
    rm Convex-PL-Linux-v0.5.tar.zip
    rm Convex-PL-Linux-v0.5.tar
    rm -r __MACOSX
    chmod +x Convex-PL
fi

if [[ ! -f ./smina.static ]]; then
    echo -e "\nDownloading Lin_F9!"
    wget https://github.com/cyangNYU/Lin_F9_test/raw/master/smina.static --no-check-certificate -q --show-progress
    chmod +x smina.static
fi

if [[ ! -d ./AA-Score-Tool-main ]]; then
    echo -e "\nDownloading AA-Score!"
    wget https://github.com/Xundrug/AA-Score-Tool/archive/refs/heads/main.zip --no-check-certificate -q --show-progress
    unzip main.zip
    rm main.zip
fi

if [[ ! -d ./gypsum_dl-1.2.1 ]]; then
    echo -e "\nDownloading GypsumDL!"
    wget https://github.com/durrantlab/gypsum_dl/archive/refs/tags/v1.2.1.tar.gz --no-check-certificate -q --show-progress
    tar -xvf v1.2.1.tar.gz
    rm v1.2.1.tar.gz
fi

if [[ ! -d ./SCORCH ]]; then
    echo -e "\nDownloading SCORCH!"
    wget https://github.com/SMVDGroup/SCORCH/archive/refs/tags/v1.0.0.tar.gz --no-check-certificate -q --show-progress
    tar -xvf v1.0.0.tar.gz
    rm v1.0.0.tar.gz
fi

if [[ ! -d ./rf-score-vs ]]; then
    echo -e "\nDownloading RF-Score-VS!"
    wget https://github.com/oddt/rfscorevs_binary/releases/download/1.0/rf-score-vs_v1.0_linux_2.7.zip -q --show-progress --no-check-certificate
    unzip rf-score-vs_v1.0_linux_2.7.zip
    rm rf-score-vs_v1.0_linux_2.7.zip
    rm -r ./test
    rm README.md
    chmod +x rf-score-vs
fi

if [[ ! -d ./RTMScore-main ]]; then
    echo -e "\nDownloading RTMScore!"
    wget https://github.com/sc8668/RTMScore/archive/refs/heads/main.zip --no-check-certificate -q --show-progress
    unzip main.zip
    rm main.zip
    rm ./RTMScore-main/scripts -r
    rm ./RTMScore-main/121.jpg

fi

if [[ ! -f ./models/DeepCoy* ]]; then
    echo -e "\nDownloading DeepCoy models!"
    cd ./models
    wget https://opig.stats.ox.ac.uk/data/downloads/DeepCoy_pretrained_models.tar.gz
    tar -xvf DeepCoy_pretrained_models.tar.gz
    rm DeepCoy_pretrained_models.tar.gz
fi

echo -e """
###############################################################
# DockM8 installation complete
###############################################################
"""
###############################################################
echo -e """
###############################################################
# Checking installation success
###############################################################
"""

# Check if conda environment is present in the list of environments
if conda env list | grep -q $ENV_NAME; then
    echo -e "\nDockM8 conda environment is present!"
else
    echo -e "\nDockM8 conda environment is not present!"
fi

# Check if required packages are installed in the $ENV_NAME environment
required_packages=("rdkit" "ipykernel" "scipy" "spyrmsd" "kneed" "scikit-learn-extra" "molvs" "seaborn" "xgboost" "openbabel" "pymesh" "espsim" "oddt" "biopandas" "redo" "MDAnalysis==2.0.0" "prody==2.1.0" "dgl" "Pebble" "tensorflow" "meeko" "chembl_structure_pipeline" "posebusters" "streamlit" "torch" "torchvision" "torchaudio" "torch_scatter" "torch_sparse" "torch_spline_conv" "torch_cluster" "torch_geometric")

for package in "${required_packages[@]}"; do
    if conda list -n $ENV_NAME "$package" &> /dev/null; then
        echo -e "$package is installed in the $ENV_NAME environment!"
    else
        echo -e "\nINSTALLATION ERROR : $package is not installed in the $ENV_NAME environment!"
    fi
done

# Check if DockM8 repository is present
if [[ -f dockm8.py || -d DockM8 ]]; then
    echo -e "DockM8 repository is present!"
else
    echo -e "\nINSTALLATION ERROR : DockM8 repository is not present!"
fi
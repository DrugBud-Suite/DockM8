g#!/bin/bash

USE_GIT=0

while getopts "g" opt; do
  case $opt in
    g)
      USE_GIT=1
      ;;
    \?)
      echo "Usage: $0 [-g]" >&2
      exit 1
      ;;
  esac
done
BASEDIR=$PWD


# Check for the existence of required utilities

function check_dependency() {
    local command_name="$1"
    if ! command -v "$command_name" &> /dev/null; then
        echo "ERROR: $command_name is not installed, please install manually. Please run : "sudo apt-get install $command_name""
        exit 1
    fi
}

check_dependency "wget"
if [ $USE_GIT -eq 1 ]; then
    check_dependency "git"
fi
check_dependency "unzip"
check_dependency "gcc"


###############################################################

echo -e """
###############################################################
# Installing DockM8
###############################################################
"""

# install dependencies for xgboost, GWOVina & MGLTools
if [[ "$OSTYPE" == "darwin"* || "$OSTYPE" == "msys"* ]]; then
    # dependencies for mac and windows
    echo -e "DockM8 is not compatible with Mac OS or Windows!"
    exit

elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo -e "Detected Linux OS!"

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
    
    echo -e "No conda found - installing..."
    mkdir -p $HOME/miniconda3
    cd $HOME/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh --no-check-certificate -q --show-progress
    # install miniconda3
    cd $HOME/miniconda3 && chmod -x miniconda.sh
    cd $BASEDIR && bash $HOME/miniconda3/miniconda.sh -b -u -p $HOME/miniconda3

    # remove the installer
    rm -f $HOME/miniconda3/miniconda.sh

    # define conda installation paths
    CONDA_PATH="$HOME/miniconda3/bin/conda"
    CONDA_BASE=$BASEDIR/$HOME/miniconda3
    CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo -e "Found existing conda install!"
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
$CONDA_PATH config --set ssl_verify False

# create the conda environment
ENV_NAME="dockm8"

if conda env list | grep -q "^$ENV_NAME\s"; then
    echo "Conda environment '$ENV_NAME' already exists. Skipping creation."
    conda activate dockm8
else
    conda create -n $ENV_NAME python=3.10 -y
    conda deactivate
    conda activate $ENV_NAME

    conda config --add channels conda-forge

    conda install rdkit=2023.09 ipykernel scipy spyrmsd kneed scikit-learn-extra molvs seaborn xgboost openbabel docopt chembl_structure_pipeline tqdm pytest pdbfixer -q -y

    echo -e """
    ###############################################################
    # Installing Pip packages, please wait...
    ###############################################################
    """

    pip install pymesh espsim oddt biopandas redo MDAnalysis==2.0.0 prody==2.1.0 dgl Pebble tensorflow meeko posebusters streamlit terrace wandb roma omegaconf py3Dmol -q

    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q

    pip install torch_scatter torch_sparse torch_spline_conv torch_cluster torch_geometric -q

    pip install pytorch_lightning==1.9.1

    echo -e """
    ###############################################################
    # Finished installing pip packages
    ###############################################################
    """
fi

###############################################################
DOCKM8_FOLDER=""
if [[ -f dockm8.py ]]; then
    echo -e "\nDockM8 repository found in current folder."
    DOCKM8_FOLDER=$(pwd)
else
    if [ $USE_GIT -eq 1 ]; then
        # Replace wget logic with Git logic for DockM8 repository download
        echo -e "\nDownloading DockM8 repository using Git..."
        rm -rf ./DockM8
        git clone https://gitlab.com/Tonylac77/DockM8.git DockM8
        DOCKM8_FOLDER=$(pwd)/DockM8
        echo -e "\nDockM8 repository downloaded using Git."
    else
        # Use wget logic for DockM8 repository download (as in your original script)
        echo -e "\nDownloading DockM8 repository using wget..."
        rm -rf ./DockM8
        wget https://gitlab.com/Tonylac77/DockM8/-/archive/main/DockM8-main.tar.gz -O DockM8.tar.gz --no-check-certificate -q --show-progress
        tar -xf DockM8.tar.gz
        mv -f DockM8-main DockM8
        DOCKM8_FOLDER=$(pwd)/DockM8
        rm DockM8.tar.gz
        echo -e "\nDockM8 repository downloaded using wget."
    fi
fi

cd $DOCKM8_FOLDER
###############################################################

echo -e """
###############################################################
# Downloading Executables...
###############################################################
"""
if [[ ! -d $DOCKM8_FOLDER/software ]]; then
    mkdir $DOCKM8_FOLDER/software
    cd $DOCKM8_FOLDER/software
else
    cd $DOCKM8_FOLDER/software
fi

if [[ ! -f $DOCKM8_FOLDER/software/gnina ]]; then
    echo -e "\nDownloading GNINA!"
    wget https://github.com/gnina/gnina/releases/latest/download/gnina --no-check-certificate  -q --show-progress
    chmod +x gnina
fi

if [[ ! -f $DOCKM8_FOLDER/software/qvina-w ]]; then
    echo -e "\nDownloading QVINA-W!"
    wget https://github.com/QVina/qvina/raw/master/bin/qvina-w --no-check-certificate -q --show-progress
    chmod +x qvina-w
fi

if [[ ! -f $DOCKM8_FOLDER/software/qvina2.1 ]]; then
    echo -e "\nDownloading QVINA2!"
    wget https://github.com/QVina/qvina/raw/master/bin/qvina2.1 --no-check-certificate -q --show-progress
    chmod +x qvina2.1
fi

if [[ ! -f $DOCKM8_FOLDER/software/PLANTS ]]; then
    echo -e "\nPLANTS not found in software folder, if you want to use it, please see documentation for a link to register and download it!"
fi

if [[ ! -f $DOCKM8_FOLDER/software/plantain/inference.py ]]; then
    echo -e "\nDownloading PLANTAIN!"
    wget https://github.com/molecularmodelinglab/plantain/archive/refs/heads/main.zip --no-check-certificate -q --show-progress
    unzip main.zip
    rm main.zip
    mv plantain-main plantain
    rm -rf ./plantain/analysis ./plantain/prior_work ./plantain/training ./plantain/validation ./plantain/baselines
    rm ./plantain/create_env.sh ./plantain/dev_requirements.txt ./plantain/requirements.txt ./plantain/train.py 
fi

if [[ ! -f $DOCKM8_FOLDER/software/KORP-PL ]]; then
    echo -e "\nDownloading KORP-PL!"
    wget https://files.inria.fr/NanoDFiles/Website/Software/KORP-PL/0.1.2/Linux/KORP-PL-LINUX-v0.1.2.2.tar.gz --no-check-certificate -q --show-progress
    tar -xf KORP-PL-LINUX-v0.1.2.2.tar.gz
    rm KORP-PL-LINUX-v0.1.2.2.tar.gz
    chmod +x KORP-PL
fi

if [[ ! -f $DOCKM8_FOLDER/software/Convex-PL ]]; then
    echo -e "\nDownloading Convex-PLR!"
    wget https://files.inria.fr/NanoDFiles/Website/Software/Convex-PL/Files/Convex-PL-Linux-v0.5.tar.zip --no-check-certificate -q --show-progress
    unzip Convex-PL-Linux-v0.5.tar.zip
    tar -xf Convex-PL-Linux-v0.5.tar
    rm Convex-PL-Linux-v0.5.tar.zip
    rm Convex-PL-Linux-v0.5.tar
    rm -r __MACOSX
    chmod +x Convex-PL
fi

if [[ ! -f $DOCKM8_FOLDER/software/smina.static ]]; then
    echo -e "\nDownloading Lin_F9!"
    wget https://github.com/cyangNYU/Lin_F9_test/raw/master/smina.static --no-check-certificate -q --show-progress
    chmod +x smina.static
fi

if [[ ! -d $DOCKM8_FOLDER/software/AA-Score-Tool-main ]]; then
    echo -e "\nDownloading AA-Score!"
    wget https://github.com/Xundrug/AA-Score-Tool/archive/refs/heads/main.zip --no-check-certificate -q --show-progress
    unzip -q main.zip
    rm main.zip
fi

if [[ ! -d $DOCKM8_FOLDER/software/gypsum_dl-1.2.1 ]]; then
    echo -e "\nDownloading GypsumDL!"
    wget https://github.com/durrantlab/gypsum_dl/archive/refs/tags/v1.2.1.tar.gz --no-check-certificate -q --show-progress
    tar -xf v1.2.1.tar.gz
    rm v1.2.1.tar.gz
fi

if [[ ! -d $DOCKM8_FOLDER/software/SCORCH ]]; then
    echo -e "\nDownloading SCORCH!"
    wget https://github.com/SMVDGroup/SCORCH/archive/refs/tags/v1.0.0.tar.gz --no-check-certificate -q --show-progress
    tar -xf v1.0.0.tar.gz
    rm v1.0.0.tar.gz
fi

if [[ ! -f $DOCKM8_FOLDER/software/rf-score-vs ]]; then
    echo -e "\nDownloading RF-Score-VS!"
    wget https://github.com/oddt/rfscorevs_binary/releases/download/1.0/rf-score-vs_v1.0_linux_2.7.zip -q --show-progress --no-check-certificate
    unzip -q rf-score-vs_v1.0_linux_2.7.zip
    rm rf-score-vs_v1.0_linux_2.7.zip
    rm -r $DOCKM8_FOLDER/software/test
    rm README.md
    chmod +x rf-score-vs
fi

if [[ ! -d $DOCKM8_FOLDER/software/RTMScore-main ]]; then
    echo -e "\nDownloading RTMScore!"
    wget https://github.com/sc8668/RTMScore/archive/refs/heads/main.zip --no-check-certificate -q --show-progress
    unzip -q main.zip 
    rm main.zip
    rm $DOCKM8_FOLDER/software/RTMScore-main/scripts -r
    rm $DOCKM8_FOLDER/software/RTMScore-main/121.jpg

fi

cd $BASEDIR

if [[ ! -f $DOCKM8_FOLDER/software/models/DeepCoy* ]]; then
    echo -e "\nDownloading DeepCoy models!"
    cd $DOCKM8_FOLDER/software/models
    wget https://opig.stats.ox.ac.uk/data/downloads/DeepCoy_pretrained_models.tar.gz
    tar -xf DeepCoy_pretrained_models.tar.gz -C $DOCKM8_FOLDER/software/
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
    echo -e "\nINSTALLATION ERROR : DockM8 conda environment is not present!"
fi

# Check if required packages are installed in the $ENV_NAME environment
required_packages=("rdkit" "ipykernel" "scipy" "spyrmsd" "kneed" "scikit-learn-extra" "molvs" "seaborn" "xgboost" "openbabel" "pymesh" "espsim" "oddt" "biopandas" "redo" "MDAnalysis==2.0.0" "prody==2.1.0" "dgl" "Pebble" "tensorflow" "meeko" "chembl_structure_pipeline" "posebusters" "streamlit" "torch" "torchvision" "torchaudio" "torch_scatter" "torch_sparse" "torch_spline_conv" "torch_cluster" "torch_geometric")

for package in "${required_packages[@]}"; do
    if conda list -n $ENV_NAME "$package" &> /dev/null; then
        echo -e "$package is installed in the $ENV_NAME environment!"
    else
        echo -e "\nINSTALLATION ERROR : $package is not installed in the $ENV_NAME environment!"
    fi

conda activate $ENV_NAME
cd $DOCKM8_FOLDER

done
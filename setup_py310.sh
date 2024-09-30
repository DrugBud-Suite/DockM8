#!/bin/bash

# DockM8 Setup Script
# This script installs DockM8 and its dependencies on a Linux system.
# It checks for conda, installs it if necessary, creates a conda environment,
# installs required packages, and downloads necessary software.

set -x  # Exit immediately if a command exits with a non-zero status

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Error: This script is designed to run on Linux only."
    exit 1
fi

# Check for the existence of required utilities

function check_dependency() {
    local command_name="$1"
    if ! command -v "$command_name" &> /dev/null; then
        echo "ERROR: $command_name is not installed. Please install it before proceeding."
        exit 1
    fi
}


check_dependency "wget"
check_dependency "unzip"
check_dependency "gcc"
check_dependency "rsync"


###############################################################

echo -e """
###############################################################
# Installing DockM8
###############################################################
"""

###############################################################

echo -e """
###############################################################
# Verifying conda install or installing miniconda3 if not found
###############################################################
"""

# Function to check for conda installation
check_conda() {
    if command -v conda &>/dev/null; then
        echo "Conda is already installed."
        return 0
    fi

    # Check common conda installation paths
    local conda_paths=("$HOME/anaconda3/bin/conda" "$HOME/miniconda3/bin/conda" "/opt/conda/bin/conda")
    for path in "${conda_paths[@]}"; do
        if [[ -x "$path" ]]; then
            echo "Conda found at $path"
            export PATH="$(dirname "$path"):$PATH"
            return 0
        fi
    done

    echo "Conda not found. Will install Miniconda."
    return 1
}

# Function to install Miniconda
install_miniconda() {
    echo "Installing Miniconda..."
    local miniconda_installer="Miniconda3-latest-Linux-x86_64.sh"
    local miniconda_url="https://repo.anaconda.com/miniconda/$miniconda_installer"

    # Download Miniconda installer with retry
    for i in {1..3}; do
        if wget "$miniconda_url" -O "$miniconda_installer" -q --show-progress; then
            break
        fi
        echo "Download failed. Retrying in 5 seconds..."
        sleep 5
    done

    if [[ ! -f "$miniconda_installer" ]]; then
        echo "Error: Failed to download Miniconda installer after 3 attempts."
        exit 1
    fi

    bash "$miniconda_installer" -b -p "$HOME/miniconda3"
    rm "$miniconda_installer"
    # Inform the user to add Miniconda to PATH
    echo 'Please add Miniconda to your PATH by adding the following line to your .bashrc or .bash_profile:'
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"'
    export PATH="$HOME/miniconda3/bin:$PATH"
}

# Check for conda or install it
if ! check_conda; then
    install_miniconda
fi

# Source conda.sh to make 'conda' command available
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Cannot find conda.sh, conda may not be properly installed."
    exit 1
fi

conda config --set auto_activate_base false
conda config --set ssl_verify true

CONDA_PATH=$(which conda)

# Function to download DockM8
download_dockm8() {
    local target_dir="$1"
    echo "Downloading DockM8 repository..."
    mkdir -p "$target_dir"
    local download_url="https://api.github.com/repos/DrugBud-Suite/DockM8/tarball"
    if wget "$download_url" -O - -q --show-progress | tar -xzf - -C "$target_dir" --strip-components=1; then
        echo "DockM8 repository downloaded and extracted to $target_dir."
    else
        echo "Failed to download DockM8. Please check your internet connection and try again."
        exit 1
    fi
}

# Function to verify DockM8 installation
verify_dockm8() {
    local dir="$1"
    if [[ ! -f "$dir/dockm8.py" ]]; then
        echo "DockM8 installation in $dir appears to be incomplete or corrupted."
        return 1
    fi
    echo "DockM8 installation in $dir verified."
    return 0
}

# Function to update DockM8
update_dockm8() {
    local dir="$1"
    echo "Updating DockM8 in $dir..."
    local temp_dir
    temp_dir=$(mktemp -d)
    if download_dockm8 "$temp_dir"; then
        rsync -a --delete "$temp_dir/" "$dir/"
        rm -rf "$temp_dir"
        echo "DockM8 updated successfully."
    else
        rm -rf "$temp_dir"
        echo "Failed to update DockM8. The existing installation will be kept."
        exit 1
    fi
}

# Main logic
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Check for dockm8.py in the current folder
if [[ -f "./dockm8.py" ]]; then
    DOCKM8_FOLDER="$PWD"
    echo "DockM8 found in current directory: $DOCKM8_FOLDER"
    if ! verify_dockm8 "$DOCKM8_FOLDER"; then
        while true; do
            read -p "DockM8 installation may be corrupted. Would you like to update it? (y/n) " -n 1 -r
            echo
            case $REPLY in
                [Yy]) 
                    update_dockm8 "$DOCKM8_FOLDER"
                    break
                    ;;
                [Nn]) 
                    break
                    ;;
                *)
                    echo "Please enter y or n."
                    ;;
            esac
        done
    fi
else
    # Check for DockM8 folder in the script directory or one level up
    for dir in "$SCRIPT_DIR" "$SCRIPT_DIR/.."; do
        if [[ -d "$dir/DockM8" ]]; then
            DOCKM8_FOLDER="$dir/DockM8"
            echo "Existing DockM8 folder found: $DOCKM8_FOLDER"
            if verify_dockm8 "$DOCKM8_FOLDER"; then
                while true; do
                    read -p "Would you like to update DockM8? (y/n) " -n 1 -r
                    echo
                    case $REPLY in
                        [Yy]) 
                            update_dockm8 "$DOCKM8_FOLDER"
                            break
                            ;;
                        [Nn]) 
                            break
                            ;;
                        *)
                            echo "Please enter y or n."
                            ;;
                    esac
                done
            else
                while true; do
                    read -p "DockM8 installation may be corrupted. Would you like to reinstall it? (y/n) " -n 1 -r
                    echo
                    case $REPLY in
                        [Yy]) 
                            rm -rf "$DOCKM8_FOLDER"
                            download_dockm8 "$DOCKM8_FOLDER"
                            break
                            ;;
                        [Nn]) 
                            exit 1
                            ;;
                        *)
                            echo "Please enter y or n."
                            ;;
                    esac
                done
            fi
            break
        fi
    done

    # If no DockM8 folder found, offer to download
    if [[ -z "${DOCKM8_FOLDER:-}" ]]; then
        echo "No existing DockM8 installation found."
        while true; do
            read -p "Would you like to download DockM8 to $SCRIPT_DIR/DockM8? (y/n) " -n 1 -r
            echo
            case $REPLY in
                [Yy]) 
                    DOCKM8_FOLDER="$SCRIPT_DIR/DockM8"
                    download_dockm8 "$DOCKM8_FOLDER"
                    break
                    ;;
                [Nn]) 
                    echo "DockM8 installation skipped."
                    exit 0
                    ;;
                *)
                    echo "Please enter y or n."
                    ;;
            esac
        done
    fi
fi

if [[ -n "${DOCKM8_FOLDER:-}" ]]; then
    cd "$DOCKM8_FOLDER"
    echo "Current working directory: $PWD"
else
    echo "DockM8 folder not set. Installation may have failed or been skipped."
    exit 1
fi

cd $DOCKM8_FOLDER
###############################################################

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

# Create or update conda environment
ENV_NAME="dockm8_v1"
ENV_FILE="$DOCKM8_FOLDER/environment.yml"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: environment.yml not found in $DOCKM8_FOLDER"
    exit 1
fi

if conda env list | grep -q "^$ENV_NAME "; then
    echo "Updating existing $ENV_NAME environment..."
    if ! conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune; then
        echo "Error: Failed to update $ENV_NAME environment"
        exit 1
    fi
else
    echo "Creating new $ENV_NAME environment..."
    if ! conda env create -n "$ENV_NAME" -f "$ENV_FILE"; then
        echo "Error: Failed to create $ENV_NAME environment"
        exit 1
    fi
fi

echo "$ENV_NAME environment setup completed successfully"

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
    wget https://github.com/gnina/gnina/releases/latest/download/gnina -q --show-progress
    chmod +x gnina
fi

if [[ ! -f $DOCKM8_FOLDER/software/qvina-w ]]; then
    echo -e "\nDownloading QVINA-W!"
    wget https://github.com/QVina/qvina/raw/master/bin/qvina-w -q --show-progress
    chmod +x qvina-w
fi

if [[ ! -f $DOCKM8_FOLDER/software/qvina2.1 ]]; then
    echo -e "\nDownloading QVINA2!"
    wget https://github.com/QVina/qvina/raw/master/bin/qvina2.1 -q --show-progress
    chmod +x qvina2.1
fi

if [[ ! -f $DOCKM8_FOLDER/software/PLANTS ]]; then
    echo -e "\nPLANTS not found in software folder, if you want to use it, please see documentation for a link to register and download it!"
fi

if [[ ! -f $DOCKM8_FOLDER/software/KORP-PL ]]; then
    echo -e "\nDownloading KORP-PL!"
    wget https://files.inria.fr/NanoDFiles/Website/Software/KORP-PL/0.1.2/Linux/KORP-PL-LINUX-v0.1.2.2.tar.gz -q --show-progress
    tar -xf KORP-PL-LINUX-v0.1.2.2.tar.gz
    rm KORP-PL-LINUX-v0.1.2.2.tar.gz
    chmod +x KORP-PL
fi

if [[ ! -f $DOCKM8_FOLDER/software/Convex-PL ]]; then
    echo -e "\nDownloading Convex-PLR!"
    wget https://files.inria.fr/NanoDFiles/Website/Software/Convex-PL/Files/Convex-PL-Linux-v0.5.tar.zip -q --show-progress
    unzip Convex-PL-Linux-v0.5.tar.zip
    tar -xf Convex-PL-Linux-v0.5.tar
    rm Convex-PL-Linux-v0.5.tar.zip
    rm Convex-PL-Linux-v0.5.tar
    rm -r __MACOSX
    chmod +x Convex-PL
fi

if [[ ! -f $DOCKM8_FOLDER/software/smina.static ]]; then
    echo -e "\nDownloading Lin_F9!"
    wget https://github.com/cyangNYU/Lin_F9_test/raw/master/smina.static -q --show-progress
    chmod +x smina.static
fi

if [[ ! -d $DOCKM8_FOLDER/software/AA-Score-Tool-main ]]; then
    echo -e "\nDownloading AA-Score!"
    wget https://github.com/Xundrug/AA-Score-Tool/archive/refs/heads/main.zip -q --show-progress
    unzip -q main.zip
    rm main.zip
fi

if [[ ! -d $DOCKM8_FOLDER/software/gypsum_dl-1.2.1 ]]; then
    echo -e "\nDownloading GypsumDL!"
    wget https://github.com/durrantlab/gypsum_dl/archive/refs/tags/v1.2.1.tar.gz -q --show-progress
    tar -xf v1.2.1.tar.gz
    rm v1.2.1.tar.gz
fi

if [[ ! -d $DOCKM8_FOLDER/software/SCORCH-1.0.0 ]]; then
    echo -e "\nDownloading SCORCH!"
    wget https://github.com/SMVDGroup/SCORCH/archive/refs/tags/v1.0.0.tar.gz -q --show-progress
    tar -xf v1.0.0.tar.gz
    rm v1.0.0.tar.gz
fi

if [[ ! -f $DOCKM8_FOLDER/software/rf-score-vs ]]; then
    echo -e "\nDownloading RF-Score-VS!"
    wget https://github.com/oddt/rfscorevs_binary/releases/download/1.0/rf-score-vs_v1.0_linux_2.7.zip -q --show-progress
    unzip -q rf-score-vs_v1.0_linux_2.7.zip
    rm rf-score-vs_v1.0_linux_2.7.zip
    rm -r $DOCKM8_FOLDER/software/test
    rm README.md
    chmod +x rf-score-vs
fi

if [[ ! -d $DOCKM8_FOLDER/software/RTMScore-main ]]; then
    echo -e "\nDownloading RTMScore!"
    wget https://github.com/sc8668/RTMScore/archive/refs/heads/main.zip -q --show-progress
    unzip -q main.zip 
    rm main.zip
    rm $DOCKM8_FOLDER/software/RTMScore-main/scripts -r
    rm $DOCKM8_FOLDER/software/RTMScore-main/121.jpg

fi

cd $SCRIPT_DIR

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

# Installation verification
conda list -n "$ENV_NAME" | grep -E "rdkit|ipykernel|scipy|spyrmsd|kneed|scikit-learn-extra|molvs|seaborn|xgboost|openbabel|torch|torch-geonetric"
if [ $? -eq 0 ]; then
    echo "All required packages are installed."
else
    echo "Error: Some required packages are missing."
    exit 1
fi

conda init bash
conda activate $ENV_NAME
cd $DOCKM8_FOLDER

done
# Changelog

All notable changes to this project will be documented in this file.

## v1.1.0
- **Added**: PoseBusters integration for docking pose quality validation
- **Added**: GNINA_GPU docking support for GPU-accelerated molecular docking
- **Added**: Ensemble docking mode for multi-receptor virtual screening
- **Added**: Decoy optimization workflow for automated parameter tuning
- **Added**: Numba-optimized consensus analyzer with batch processing
- **Added**: Analysis pipeline for reproducing all paper benchmark figures (`analysis/`)
- **Added**: New scoring modules: CnnAffinity, CnnScore, GninaAffinity (split from monolithic Gnina scorer)
- **Added**: Fast SDF loader/writer for improved I/O performance
- **Added**: CITATION.cff for machine-readable citation metadata
- **Added**: THIRD_PARTY_LICENSES.md for bundled software license documentation
- **Changed**: Complete CLI rewrite (`dockm8.py`) with refactored module imports and v1.1 pipeline
- **Changed**: Streamlit GUI rewrite with input validation, advanced options, and command preview
- **Changed**: All Jupyter notebooks rewritten for v1.1 API with step-by-step documentation
- **Changed**: DeepCoy upgraded to Python 3 and TensorFlow 2 compatibility
- **Changed**: Vectorized consensus scoring with RESCORING_FUNCTIONS integration
- **Changed**: Refactored pose selection to use fast SDF loader/writer
- **Changed**: Conda environment renamed to `dockm8`
- **Changed**: Multi-column scoring support with optimized set-based score lookups
- **Deprecated**: AAScore (disabled — requires separate Python 3.6 environment)
- **Deprecated**: PLECScore (disabled — removed from active scoring functions)
- **Removed**: Docker support
- **Fixed**: Path handling across all modules (use Path types consistently)
- **Fixed**: File splitting API rewritten for SDF and PDBQT formats
- **Fixed**: Test data and fixtures updated for v1.1 API
- **Fixed**: Setup script verification typos and dependency checks

## v1.0.2
- **Added**: setup script to release
- **Changed**: None
- **Deprecated**: None
- **Removed**: None
- **Fixed**: Links in PDF guides
- **Security**: None

## v1.0.1
- **Added**: None
- **Changed**: None
- **Deprecated**: None
- **Removed**: None
- **Fixed**: Links in README file
- **Security**: None

## v1.0.0 - First release of DockM8
- **Added**: Ability to define a manual pocket
- **Changed**: None
- **Deprecated**: None
- **Removed**: None
- **Fixed**: None
- **Security**: None

## v0.2.3
- **Added**: Output docking poses for pose selection methods
- **Changed**: Improved ID handling for numeric or `_`-containing IDs and change License to GNU GPLv3
- **Deprecated**: None
- **Removed**: None
- **Fixed**: None
- **Security**: None

## v0.2.2
- **Added**: Decoy generation options
- **Changed**: Performance calculation handling
- **Deprecated**: None
- **Removed**: None
- **Fixed**: None
- **Security**: None

## v0.2.1
- **Added**: Conformer generation options (RDKit MMFF or GypsumDL)
- **Added**: Streamlit GUI
- **Changed**: Setup script for WSL2 compatibility
- **Deprecated**: None
- **Removed**: None
- **Fixed**: None
- **Security**: None

## v0.2
- **Added**: Setup.sh for easy installation (includes conda env and software download)
- **Changed**: Defined constants for docking programs, rescoring functions, consensus methods, and clustering metrics
- **Changed**: Updated to GypsumDL 1.2.1
- **Deprecated**: None
- **Removed**: pkasolver removed due to Python 3.10 incompatibility
- **Fixed**: None
- **Security**: None

## v0.1 (Initial Commit)
- **Added**: Initial project setup and files
- **Changed**: None
- **Deprecated**: None
- **Removed**: None
- **Fixed**: None
- **Security**: None
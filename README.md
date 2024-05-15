![DockM8 Logo](./media/DockM8_white_horizontal_smaller.png)

[![Forks](https://img.shields.io/github/forks/DrugBud-Suite/DockM8?style=for-the-badge&logo=github)](https://github.com/DrugBud-Suite/DockM8)
[![Stars](https://img.shields.io/github/stars/DrugBud-Suite/DockM8?style=for-the-badge&logo=github)](https://github.com/DrugBud-Suite/DockM8)
[![Watching](https://img.shields.io/github/watchers/DrugBud-Suite/DockM8?style=for-the-badge&logo=github)](https://github.com/DrugBud-Suite/DockM8)
![License](https://img.shields.io/github/license/DrugBud-Suite/DockM8?style=for-the-badge)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?style=for-the-badge&logo=github)](https://github.com/DrugBud-Suite/DockM8/pulls)
[![Last Commit](https://img.shields.io/github/last-commit/DrugBud-Suite/DockM8?style=for-the-badge&logo=github)](https://github.com/DrugBud-Suite/DockM8)
[![Commit Activity](https://img.shields.io/github/commit-activity/m/DrugBud-Suite/DockM8?style=for-the-badge&logo=github)](https://github.com/DrugBud-Suite/DockM8)
[![Open Issues](https://img.shields.io/github/issues/DrugBud-Suite/DockM8?style=for-the-badge&logo=github)](https://github.com/DrugBud-Suite/DockM8/issues)
[![Closed Issues](https://img.shields.io/github/issues-closed/DrugBud-Suite/DockM8?style=for-the-badge&logo=github)](https://github.com/DrugBud-Suite/DockM8/issues)

**DockM8 is and all-in-one Structure-Based Virtual Screening workflow based on the concept of consensus docking. The workflow takes care of library and protein preparation, docking, pose selection, rescoring and ranking. We actively encourage the community to participate in the continued development of DockM8. Please see the [**contribution guide**](https://github.com/DrugBud-Suite/DockM8/blob/main/CONTRIBUTING.md) for details.**

DockM8 only runs on Linux systems. However, we have tested the installation on Windows Subsystem for Linux v2 and using VirtualBox virtual machines.

## Automatic installation (Python 3.10 / Ubuntu 22.04)

For automatic installation, download and run [**setup_py310.sh**](https://github.com/DrugBud-Suite/DockM8/releases/download/v1.0.2/setup_py310.sh) This will create the required conda environment and download the respository if not done already. Make sure the installation script can be executed by running `chmod +x setup_py310.sh` and then `./setup_py310.sh`.

## Manual Installation (Python 3.10 / Ubuntu 22.04)

Please refer to the [**Installation Guide**](https://github.com/DrugBud-Suite/DockM8/blob/main/DockM8_Installation_Guide.pdf) provided.

## Running DockM8 (via streamlit GUI)

DockM8 comes with a simple form-based GUI for ease of use. To launch it, run the following command :

`streamlit run **PATH_TO**/gui.py`

You can click the `localhost` link to access the GUI.

## Running DockM8 (via command-line / dockm8.py script)

Please refer to the [**Usage Guide**](https://github.com/DrugBud-Suite/DockM8/blob/main/DockM8_Usage_Guide.pdf) provided.

## Running DockM8 (via Jupyter Notebook)

1. Open dockm8.ipynb, dockm8_ensemble.ipynb or dockm8_decoys.ipynb in your favorite IDE, depending on which DockM8 mode you want to use.

2. Follow the instructions in the Markdown cells

## Issues and bug reports

Please you the issue system built into github to report issues. They will be resolved as soon as possible.

## Acknowledgements

We acknowledge and thank the authors of the packages used in DockM8. Please see the publication for citations.

## Citation

Coming Soon

## License

This project is licensed under the GNU GPL v3.0 License - see the [LICENSE.md](https://github.com/DrugBud-Suite/DockM8/blob/main/LICENSE) file for details.

## Contributing

We highly encourage contributions from the community - see the [CONTRIBUTING.md](https://github.com/DrugBud-Suite/DockM8/blob/main/CONTRIBUTING.md) file for details.

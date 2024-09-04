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

<a href="https://www.buymeacoffee.com/tonylac77" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>

**DockM8 is an all-in-one Structure-Based Virtual Screening workflow based on the concept of consensus docking. The workflow takes care of library and protein preparation, docking, pose selection, rescoring and ranking. We actively encourage the community to participate in the continued development of DockM8. Please see the [contribution guide](https://gitlab.com/Tonylac77/DockM8/-/blob/main/CONTRIBUTING.md) for details.**

DockM8 only runs on Linux systems. However, we have tested the installation on Windows Subsystem for Linux v2 and using VirtualBox virtual machines.

<details>
<summary><b>Installation</b></summary>

### Automatic Installation (Python 3.10 / Ubuntu 22.04)

For automatic installation, download and run [setup_py310.sh](https://gitlab.com/Tonylac77/DockM8/-/blob/main/setup_py310.sh). This script will set up the required conda environments and install all necessary packages. Make sure the installation script can be executed by running:

```bash
chmod +x setup_py310.sh
./setup_py310.sh
```

The script will create the main `dockm8` environment and additional specialized environments.

### Manual Installation (Python 3.10 / Ubuntu 22.04)

Create the main `dockm8` environment and install the following packages:

```bash
conda create -n dockm8 python=3.10
conda install -c conda-forge rdkit=2023.09 spyrmsd kneed molvs xgboost openbabel docopt pdbfixer smina lightning -y
conda install ipykernel scipy seaborn tqdm pytest pydantic -y
conda install -c mx reduce -y

pip3 install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pymesh espsim oddt biopandas redo MDAnalysis==2.0.0 prody==2.1.0 dgl Pebble tensorflow meeko posebusters streamlit prolif datamol yapf medchem molgrid
pip install torch_scatter torch_sparse torch_spline_conv torch_cluster torch_geometric
pip install streamlit_molstar sh scikit-learn
pip install roma pytorch_lightning omegaconf terrace dgllife scikit-learn-extra posecheck
```

#### Installing binaries

DockM8 will automatically download and install binaries/programs when required.


</details>

<details>
<summary><b>Running DockM8</b></summary>

### Via Streamlit GUI

DockM8 comes with a simple form-based GUI for ease of use. To launch it, run:

```bash
streamlit run **PATH_TO**/gui.py
```

Click the `localhost` link to access the GUI.

### Via Command-line (dockm8.py script)

DockM8 now uses a configuration file (`config.yml`) to specify all parameters for the workflow when running via command-line. This file should be located in the root directory of DockM8.

#### Configuration File

Here's a sample of what the `config.yml` file might look like:

```yaml
general:
  software: "/path/to/DockM8/software"
  mode: "single"
  n_cpus: 0

receptor(s):
  - "/path/to/receptor.pdb"

docking_library: "/path/to/library.sdf"

docking:
  docking_programs:
    - "SMINA"
    - "GNINA"
  n_poses: 10
  exhaustiveness: 8

# ... other configuration options ...
```

#### Configuration Sections

- `general`: Overall settings for DockM8
- `decoy_generation`: Settings for generating decoy compounds
- `receptor(s)`: Path(s) to receptor file(s)
- `docking_library`: Path to the compound library for docking
- `protein_preparation`: Settings for preparing the protein structure
- `ligand_preparation`: Settings for preparing the ligands
- `pocket_detection`: Settings for detecting the binding pocket
- `docking`: Settings for the docking process
- `post_docking`: Settings for post-docking processing
- `pose_selection`: Settings for selecting poses
- `rescoring`: List of rescoring methods to use
- `consensus`: Method for consensus scoring
- `threshold`: Threshold for ensemble and active learning modes

Refer to the `config.yml` file in the DockM8 repository for a complete list of available options and their descriptions.

#### Running DockM8

1. Ensure you have set up the `config.yml` file with your desired parameters.

2. Open a terminal and activate the dockm8 python environment:
   ```bash
   conda activate dockm8
   ```

3. Run the following command:
   ```bash
   python **PATH_TO**/dockm8.py --config **PATH_TO**/config.yml
   ```

   DockM8 will automatically read the configuration and warn you of any errors in the configuration file.

### Via Jupyter Notebook

1. Open `dockm8.ipynb`, `dockm8_ensemble.ipynb`, or `dockm8_decoys.ipynb` in your favorite IDE, depending on which DockM8 mode you want to use.

2. Ensure that the notebook is configured to use the `config.yml` file.

3. Follow the instructions in the Markdown cells.

</details>

## Acknowledgements

We acknowledge and thank the authors of the packages used in DockM8. Please see the publication for citations.

## Citation

Coming Soon

## License

This project is licensed under the GNU GPL v3.0 License - see the [LICENSE.md](https://gitlab.com/Tonylac77/DockM8/-/blob/main/LICENSE) file for details.

## Contributing

We highly encourage contributions from the community - see the [CONTRIBUTING.md](https://gitlab.com/Tonylac77/DockM8/-/blob/main/CONTRIBUTING.md) file for details.
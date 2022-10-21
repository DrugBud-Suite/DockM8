# WoConDock
<!-- markdown-link-check-disable-next-line -->
Open source workflow for structure-based virtual screening (SBVS).
WoConDock, a SBVS workflow, is developed to fill the demand between all researchers, regardless of their background. we are doing it with the goal of allowing researchers from all backgrounds to be able to perform SBVS on the wealth of 3D protein structures that are now available.

The code is divied into 4 parts :
1. Predocking
2. Docking
3. Postdocking
4. Ranking

## Installation on Linux by Jupyter Notebook(Ubuntu 22.04)
<!-- markdown-link-check-disable-next-line -->

1. Anaconda should be installed to be able to create local environment [For more info](https://docs.anaconda.com/anaconda/install/index.html)

2. Create and activate a WoConDock conda environment:
`conda create -n wocondock python=3.8`
`conda activate wocondock`

3. Install required packages using the following commands:
`conda install -c conda-forge rdkit chembl_structure_pipeline ipykernel scipy -y`
`pip install pymesh spyrmsd kneed scikit-learn-extra espsim`
`snap install openbabel`
`pip install torch -f https://data.pyg.org/whl/torch-1.9.1+cpu.html`
`pip install torch-scatter -f https://data.pyg.org/whl/torch-1.9.1+cpu.html #different if CPU or GPU`
`pip install torch-sparse`
`pip install torch-geometric==2.0.1`
`pip install cairosvg svgutils molvs`
`pip install -q git+https://github.com/mayrf/pkasolver.git`


2. After activating conda environment, open [setup.ipynb](https://gitlab.com/hibrahim21/CADD22/-/blob/main/consensus_docking_python/setup.ipynb) and install packages

## Running WoConDock

1. 

## License
<!-- markdown-link-check-disable-next-line -->
This project is licensed under the MIT License - see the [LICENSE.md](https://gitlab.com/hibrahim21/CADD22/-/blob/main/LICENSE) file for details.



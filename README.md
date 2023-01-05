# WoConDock
<!-- markdown-link-check-disable-next-line -->
Open source workflow for structure-based virtual screening (SBVS).
WoConDock, a SBVS workflow, is developed to fill the demand between all researchers, regardless of their background. we are doing it with the goal of allowing researchers from all backgrounds to be able to perform SBVS on the wealth of 3D protein structures that are now available.

The code is divied into 4 parts :
1. Predocking
2. Docking
3. Postdocking
4. Ranking

## Installation (Python 3.8 / Ubuntu 22.04)
<!-- markdown-link-check-disable-next-line -->

1. Anaconda should be installed to be able to create local environment [For more info](https://docs.anaconda.com/anaconda/install/index.html)

2. Create and activate a WoConDock conda environment:  
`conda create -n wocondock python=3.8`  
`conda activate wocondock`  

3. Install required packages using the following commands:  
`conda install -c conda-forge rdkit chembl_structure_pipeline ipykernel scipy spyrmsd kneed scikit-learn-extra cairosvg svgutils molvs jupyter notebook seaborn -y`  
`pip install pymesh espsim oddt biopandas`  
`snap install openbabel` (alternatively install from Ubuntu Software manager)  
`pip install torch==1.9.1+cpu`  
`pip install torch-scatter==2.1.0 -f https://data.pyg.org/whl/torch-1.9.1+cpu.html #different if CPU or GPU`  
`pip install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.9.1+cpu.html`  
`pip install torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.9.1+cpu.html`  
`pip install torch-cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.9.1+cpu.html`  
`pip install torch-geometric==2.0.1 -f https://data.pyg.org/whl/torch-1.9.1+cpu.html`  
`pip install -q git+https://github.com/mayrf/pkasolver.git`    

4. Clone repository to your machine:  
`git clone https://gitlab.com/hibrahim21/CADD22.git`  

5. Ensure you have permissions to run the scripts required
On Linux, right-click the script file, and ensure 'allow executing file as program' is ticked. This applies to gnina.sh, PLANTS.sh and rf-score-vs.sh.  

## Running WoConDock (via wocondock_refactored folder)

1. Ensure required files are in the wocondock_refactored folder
- protein/receptor file (with hydrogens added) as a .pdb file
- reference ligand file as a .mol2 file
- docking library as a .sdf file

2. Open wocondock.ipynb

3. Make sure to set the variables to point to your protein, reference and docking library files. You also need to specify the location where the software files are located. Finally, if the docking library contains a unique identifier, specify the name of the column (id_column) to keep track of this.

4. The notebook can then be run in sequence. Further explanations and instructions as present in the notebook.

## License
<!-- markdown-link-check-disable-next-line -->
This project is licensed under the MIT License - see the [LICENSE.md](https://gitlab.com/hibrahim21/CADD22/-/blob/main/LICENSE) file for details.




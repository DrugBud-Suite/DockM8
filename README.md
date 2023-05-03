# DockM8
<!-- markdown-link-check-disable-next-line -->
**Workflow for Consensus Docking**  
DockM8 is and all-in-one Structure-Based Virtual Screening workflow based on the concept of consensus docking. The workflow takes care of library preparation, docking, clustering of docking poses, rescoring and ranking.

## Installation (Python 3.8 / Ubuntu 22.04)
<!-- markdown-link-check-disable-next-line -->

1. Anaconda should be installed to be able to create a local environment [For more info](https://docs.anaconda.com/anaconda/install/index.html)

2. Clone repository to your machine:  
`git clone https://gitlab.com/Tonylac77/DockM8.git` 

3. Create and activate a DockM8 conda environment:  
`conda create -n dockm8 python=3.8`  
`conda activate dockm8`  

4. Install required packages using the following commands:  
`conda install -c conda-forge rdkit chembl_structure_pipeline ipykernel scipy spyrmsd kneed scikit-learn-extra cairosvg svgutils molvs jupyter notebook seaborn -y`  
`pip install pymesh espsim oddt biopandas redo MDAnalysis==2.0.0 prody==2.1.0 dgl`  
`pip install torch==1.9.1`
`pip install torch-scatter==2.1.0 -f https://data.pyg.org/whl/torch-1.9.1+cpu.html` #different if CPU or GPU  
`pip install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.9.1+cpu.html`  
`pip install torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.9.1+cpu.html`  
`pip install torch-cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.9.1+cpu.html`  
`pip install torch-geometric==2.0.1 -f https://data.pyg.org/whl/torch-1.9.1+cpu.html`  
`pip install -q git+https://github.com/mayrf/pkasolver.git`  

    If you want to run the AA-score or the delta_LinF9_XGB scoring functions, you should build OpenBabel from source with Python bindings.  
`git clone https://github.com/openbabel/openbabel.git`  
`cd openbabel`  
`git checkout openbabel-3-1-1 `  
`mkdir build`  
`cd build`  
`cmake -DWITH_MAEPARSER=OFF -DWITH_COORDGEN=OFF -DPYTHON_BINDINGS=ON -DRUN_SWIG=ON ..`  
`make`  
`make install`  
`conda install -c conda-forge openbabel`  

    If not you can simply install OpenBabel using `snap install openbabel` (alternatively install from Ubuntu Software manager)  

    Navigate to the pkasolver-main folder in the /software directory and run `python setup.py install`  

6. If GNINA does not run, you may need to run the following command to point GNINA to the lib folder in the anaconda installation directory : `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/anaconda3/lib/`  

5. (Optional) Ensure you have permissions to run the scripts required
On Linux, right-click the script file, and ensure 'allow executing file as program' is ticked. This applies to gnina.sh, PLANTS.sh and rf-score-vs.sh.  

## Running DockM8 (via command-line / dockm8.py script)

1. Create a working directory on your machine and ensure the required files are present:
- protein/receptor file (with hydrogens added) as a .pdb file
- If using a ligand to define the binding pocket : reference ligand file as an .sdf file
- docking library as a .sdf file

2. Open a terminal and activate the dockm8 python environment

3. Run the following command:

´python /path/to/dockm8.py --args´

--software: The path to the software folder.  
--proteinfile: The path to the protein file (.pdb).  
--pocket: The method to use for pocket determination. Must be one of 'reference' or 'dogsitescorer'.  
--dockinglibrary: The path to the docking library file (.sdf).  
--idcolumn: The unique identifier column used in the docking library.  
--protonation: The method to use for compound protonation. Must be one of 'pkasolver', 'GypsumDL', or 'None'.  
--docking: The method(s) to use for docking. Must be one or more of 'GNINA', 'SMINA', or 'PLANTS'.  
--metric: The method(s) to use for pose clustering. Must be one or more of 'RMSD', 'spyRMSD', 'espsim', 'USRCAT', '3DScore', 'bestpose', 'bestpose_GNINA', 'bestpose_SMINA', or 'bestpose_PLANTS'.  
--nposes: The number of poses to generate for each docking software. Default=10  
--exhaustiveness: The precision used if docking with SMINA/GNINA. Default=8  
--parallels: Whether or not to run the workflow in parallel. Default=1 (on). Can be set to 1 (on) or 0 (off).  
-- ncpus: The number of cpus to use for the workflow. Default behavior is to use half of the available cpus.  
-- clustering: Which algorithm to use for clustering. Must be one of 'KMedoids', 'Aff_prop'.  
--rescoring: Which scoring functions to use for rescoring. Must be one or more of 'gnina', 'AD4', 'chemplp', 'rfscorevs', 'LinF9', 'vinardo', 'plp', 'AAScore'.  

## Running DockM8 (via Jupyter Notebook)

1. Open dockm8.ipynb in your favorite IDE

2. Follow the instructions in the Markdown cells


## License (NEEDS CHECKING...)
<!-- markdown-link-check-disable-next-line -->
This project is licensed under the MIT License - see the [LICENSE.md](https://gitlab.com/Tonylac77/DockM8/-/blob/main/LICENSE) file for details.




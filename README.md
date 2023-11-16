# DockM8
<!-- markdown-link-check-disable-next-line -->
**Workflow for Consensus Docking**  
DockM8 is and all-in-one Structure-Based Virtual Screening workflow based on the concept of consensus docking. The workflow takes care of library preparation, docking, clustering of docking poses, rescoring and ranking.

## Installation (tested on Python 3.8 and 3.9 / Ubuntu 22.04)
<!-- markdown-link-check-disable-next-line -->

1. Anaconda should be installed to be able to create a local environment [For more info](https://docs.anaconda.com/anaconda/install/index.html)

2. Clone repository to your machine:  
`git clone https://gitlab.com/Tonylac77/DockM8.git` 

3. Create and activate a DockM8 conda environment:  
`conda create -n dockm8 python=3.8` OR `conda create -n dockm8 python=3.9`  
`conda activate dockm8`  

4. Install required packages using the following commands:  
`conda install -c conda-forge rdkit ipykernel scipy spyrmsd kneed scikit-learn-extra cairosvg svgutils molvs seaborn xgboost -y`  
`pip install pymesh espsim oddt biopandas redo MDAnalysis==2.0.0 prody==2.1.0 dgl Pebble tensorflow meeko chembl_structure_pipeline posebusters`  
`pip install torch==1.9.1`  
`pip install torch-scatter==2.1.0 torch-sparse==0.6.12 torch-spline-conv==1.2.1 torch-cluster==1.6.0 torch-geometric==2.0.1 -f https://data.pyg.org/whl/torch-1.9.1+cpu.html`  
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

6. If GNINA does not run, you may need to run the following command to point GNINA to the lib folder in the anaconda installation directory : `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/anaconda3/lib/`  

5. (Optional) Ensure you have permissions to run the scripts required
On Linux, right-click the script file, and ensure 'allow executing file as program' is ticked. This applies to gnina.sh, PLANTS.sh and rf-score-vs.sh.  


## Installation (Python 3.10 / Ubuntu 22.04)
<!-- markdown-link-check-disable-next-line -->

1. Anaconda should be installed to be able to create a local environment [For more info](https://docs.anaconda.com/anaconda/install/index.html)

2. Clone repository to your machine:  
`git clone https://gitlab.com/Tonylac77/DockM8.git` 

3. Create and activate a DockM8 conda environment:  
`conda create -n dockm8 python=3.10`  
`conda activate dockm8`  

4. Install required packages using the following commands:  
`conda install -c conda-forge rdkit ipykernel scipy spyrmsd kneed scikit-learn-extra molvs seaborn xgboost openbabel -y`  
`pip install pymesh espsim oddt biopandas redo MDAnalysis==2.0.0 prody==2.1.0 dgl Pebble tensorflow meeko chembl_structure_pipeline`  
`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`  
`pip install torch_scatter torch_sparse torch_spline_conv torch_cluster torch_geometric`  
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

    If not you can simply install OpenBabel using `snap install openbabel` (alternatively install from Ubuntu Software manager)  

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

`python /path/to/dockm8.py --args`  


`--software`: The path to the software folder.  
`--mode`: Choose mode with which to run dockm8. Options are:
  - 'single' : Regular docking on one receptor.
  - 'ensemble' : Ensemble docking on multiple receptor conformations.  

`--receptor`: The path to the protein file (.pdb) or multiple paths if using ensemble mode.  
`--pocket`: The method to use for pocket determination. Must be one of:
  - 'reference' : Uses reference ligand to define pocket.
  - 'RoG' (radius of gyration) : Uses reference ligand's radius of gyration to define pocket.  
  - 'dogsitescorer' : API call to DogSiteScorer webserver to determine pocket.  

`--reffile`: The path to the reference ligand to use for pocket determination. Must be set if using 'reference' or 'RoG' pocket mode.  
`--docking_library`: The path to the docking library file (.sdf).  
`--idcolumn`: The unique identifier column used in the docking library.  
`--protonation`: The method to use for compound protonation. Must be one of:
  - 'pkasolver' : Use pkasolver library to protonate library
  - 'GypsumDL' : Use GypsumDL library to protonate library
  - 'None' : Do not protonate library  

`--docking_programs`: The method(s) to use for docking. Must be one or more of:
  - 'GNINA'
  - 'SMINA'
  - 'QVINAW'
  - 'QVINA2'
  - 'PLANTS'  

`--clustering_metric`: The method(s) to use for pose clustering. Must be one or more of:
  - 'RMSD' : Cluster compounds on RMSD matrix of poses
  - 'spyRMSD' : Cluster compounds on symmetry-corrected RMSD matrix of poses
  - 'espsim' : Cluster compounds on electrostatic shape similarity matrix of poses
  - 'USRCAT' : Cluster compounds on shape similarity matrix of poses
  - '3DScore' : Selects pose with the lowest average RMSD to all other poses
  - 'bestpose' : Takes the best pose from each docking program
  - 'bestpose_GNINA' : Takes the best pose from GNINA docking program
  - 'bestpose_SMINA' : Takes the best pose from SMINA docking program
  - 'bestpose_QVINAW' : Takes the best pose from QVINAW docking program
  - 'bestpose_QVINA2' : Takes the best pose from QVINA2 docking program
  - 'bestpose_PLANTS' : Takes the best pose from PLANTS docking program  

`--nposes`: The number of poses to generate for each docking software. Default=10  
`--exhaustiveness`: The precision used if docking with SMINA/GNINA/QVINA. Default=8  
`--ncpus`: The number of cpus to use for the workflow. Default behavior is to use half of the available cpus.  
`--clustering_method`: Which algorithm to use for clustering. Must be one of 'KMedoids', 'Aff_prop'. Must be set when using 'RMSD', 'spyRMSD', 'espsim', 'USRCAT' clustering metrics.  
`--rescoring`: Which scoring functions to use for rescoring. Must be one or more of 'GNINA_Affinity', 'CNN-Score', 'CNN-Affinity', 'AD4', 'CHEMPLP', 'RFScoreVS', 'LinF9', 'Vinardo', 'PLP', 'AAScore', 'ECIF', 'SCORCH', 'RTMScore', 'NNScore', 'PLECnn', 'KORPL', 'ConvexPLR'.  
`--consensus`: Which consensus method to use. Must be one of 'method1', 'method2', 'method3', 'method4', 'method5', 'method6', 'method7'.  
`--threshold`: Threshold in % to use when using 'ensemble' mode. Will find the hits in common in the x% of top ranked compounds in all of the conformations.  

## Running DockM8 (via Jupyter Notebook)

1. Open dockm8.ipynb in your favorite IDE

2. Follow the instructions in the Markdown cells


## License (NEEDS CHECKING...)
<!-- markdown-link-check-disable-next-line -->
This project is licensed under the MIT License - see the [LICENSE.md](https://gitlab.com/Tonylac77/DockM8/-/blob/main/LICENSE) file for details.




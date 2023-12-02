# DockM8

**Workflow for Consensus Docking**  
DockM8 is and all-in-one Structure-Based Virtual Screening workflow based on the concept of consensus docking. The workflow takes care of library preparation, docking, clustering of docking poses, rescoring and ranking.

![](https://gitlab.com/Tonylac77/DockM8/blob/main/media/banner_withtitle.png)

## Automatic installation

For automatic installation, download and run [**setup.sh**](LINK_TO_SETUP.SH_DOWNLOAD) This will create the required conda environment and download the respository if not done already. Make sure the installation script can be executed by running `chmod +x setup.sh` and then `./setup.sh`.

## Manual Installation (Python 3.10 / Ubuntu 22.04)

1. Anaconda or Miniconda should be installed to be able to create a local environment [For more info](https://docs.anaconda.com/anaconda/install/index.html)

2. Download DockM8 repository:  
`wget https://github.com/Tonylac77/DockM8/main.zip -O DockM8.zip --no-check-certificate`  
`unzip DockM8.zip`  
`rm DockM8.zip`  
or by using git : `git clone https://github.com/Tonylac77/DockM8/main.zip`  

3. Create and activate a DockM8 conda environment:  
  - From environment file:  
      `conda env create -f environment.yml`  
      `conda activate dockm8`
  - Manual installation:  
      `conda create -n dockm8 python=3.10`  
      `conda activate dockm8`  
    - Install required packages using the following commands:  
    `conda install -c conda-forge rdkit ipykernel scipy spyrmsd kneed scikit-learn-extra molvs seaborn xgboost openbabel -y`  
    `pip install pymesh espsim oddt biopandas redo MDAnalysis==2.0.0 prody==2.1.0 dgl Pebble tensorflow meeko chembl_structure_pipeline`  
    `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`  
    `pip install torch_scatter torch_sparse torch_spline_conv torch_cluster torch_geometric`  
    `pip install -q git+https://github.com/mayrf/pkasolver.git`  

6. If GNINA does not run, you may need to run the following command to point GNINA to the lib folder in the anaconda installation directory : `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/anaconda3/lib/`  

5. **Optional** : Ensure you have permissions to run the scripts required:
On Linux, right-click the script file, and ensure 'allow executing file as program' is ticked. This applies to gnina.sh, PLANTS.sh, rf-score-vs.sh, Convex-PL.sh, KORP-PL.sh, qvina-w.sh, qvina2.1.sh, and smina.static.  Alternatively you can use the following command to give execute permissions to all files in the DockM8 folder: `chmod +x DockM8/software`

## Running DockM8 (via command-line / dockm8.py script)

1. Create a working directory on your machine and ensure the required files are present:
- protein/receptor file (with hydrogens added) as a .pdb file
- If using a ligand to define the binding pocket : reference ligand file as an .sdf file
- docking library as a .sdf file

2. Open a terminal and activate the dockm8 python environment

3. Run the following command:

`python /path/to/dockm8.py --args`  

`--software`: The path to the software folder. In most cases this is where the DockM8 repository was downloaded to (`path/to/DockM8/software`)  
`--mode`: Choose mode with which to run dockm8. Options are:
  - 'single' : Regular docking on one receptor.
  - 'ensemble' : Ensemble docking on multiple receptor conformations.  

`--receptor`: The path to the protein file (.pdb) or multiple paths if using ensemble mode.  
`--pocket`: The method to use for pocket determination. Must be one of:
  - 'reference' : Uses reference ligand to define pocket.
  - 'RoG' (radius of gyration) : Uses reference ligand's radius of gyration to define pocket.  
  - 'dogsitescorer' :  Call to DogSiteScorer webserver to determine pocket coordinates, works on volume by default although this can be changed in *dogsitescorer.py*.  

`--reffile`: The path to the reference ligand to use for pocket determination. Must be provided if using 'reference' or 'RoG' pocket mode.  
`--docking_library`: The path to the docking library file (.sdf format).  
`--idcolumn`: The unique identifier column used in the docking library.  
`--protonation`: The method to use for compound protonation. Must be one of:
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
  - You can also use any of the scoring functions (see rescoring argument) and DockM8 will select the best pose for each compound according to the specified scoring function.

`--nposes`: The number of poses to generate for each docking software. Default=10  
`--exhaustiveness`: The precision used if docking with SMINA/GNINA/QVINA. Default=8  
`--ncpus`: The number of cpus to use for the workflow. Default behavior is to use half of the available cpus.  
`--clustering_method`: Which algorithm to use for clustering. Must be one of 'KMedoids', 'Aff_prop'. Must be set when using 'RMSD', 'spyRMSD', 'espsim', 'USRCAT' clustering metrics.  
`--rescoring`: Which scoring functions to use for rescoring. Must be one or more of 'GNINA_Affinity', 'CNN-Score', 'CNN-Affinity', 'AD4', 'CHEMPLP', 'RFScoreVS', 'LinF9', 'Vinardo', 'PLP', 'AAScore', 'SCORCH', 'RTMScore', 'NNScore', 'PLECScore', 'KORPL', 'ConvexPLR'.  
`--consensus`: Which consensus method to use. Must be one of 'ECR_best', 'ECR_avg', 'avg_ECR', 'RbR', 'RbV', 'Zscore_best', 'Zscore_avg'.  
`--threshold`: Threshold in % to use when using 'ensemble' mode. Will find the hits in common in the x% of top ranked compounds in all of the receptor conformations.

## Running DockM8 (via Jupyter Notebook)

1. Open dockm8.ipynb in your favorite IDE

2. Follow the instructions in the Markdown cells

## Acknowledgements

## Citation

## License (NEEDS CHECKING...)
<!-- markdown-link-check-disable-next-line -->
This project is licensed under the MIT License - see the [LICENSE.md](https://gitlab.com/Tonylac77/DockM8/-/blob/main/LICENSE) file for details.




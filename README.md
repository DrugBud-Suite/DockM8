![](./media/DockM8_white_horizontal_smaller.png)

**DockM8 is and all-in-one Structure-Based Virtual Screening workflow based on the concept of consensus docking. The workflow takes care of library and protein preparation, docking, pose selection, rescoring and ranking. We actively encourage the community to participate in the continued development of DockM8. Please see the [**contribution guide**](https://gitlab.com/Tonylac77/DockM8/-/blob/main/CONTRIBUTING.md) for details.**

DockM8 only runs on Linux systems. However, we have tested the installation on Windows Subsystem for Linux v2 and using VirtualBox virtual machines.

## Automatic installation (Python 3.10 / Ubuntu 22.04)

For automatic installation, download and run [**setup_py310.sh**](https://gitlab.com/Tonylac77/DockM8/-/blob/main/setup_py310.sh) This will create the required conda environment and download the respository if not done already. Make sure the installation script can be executed by running `chmod +x setup.sh` and then `./setup_py310.sh`.

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
    `conda install -c conda-forge rdkit ipykernel scipy spyrmsd kneed scikit-learn-extra molvs seaborn xgboost openbabel docopt -y`  
    `pip install pymesh espsim oddt biopandas redo MDAnalysis==2.0.0 prody==2.1.0 dgl Pebble tensorflow meeko chembl_structure_pipeline streamlit posebusters`  
    `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`  
    `pip install torch_scatter torch_sparse torch_spline_conv torch_cluster torch_geometric`  

6. If GNINA does not run, you may need to run the following command to point GNINA to the lib folder in the anaconda installation directory : `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:**PATH_TO**/anaconda3/lib/`  

5. **Optional** : Ensure you have permissions to run the scripts required:
On Linux, right-click the script file, and ensure 'allow executing file as program' is ticked. This applies to gnina.sh, PLANTS.sh, rf-score-vs.sh, Convex-PL.sh, KORP-PL.sh, qvina-w.sh, qvina2.1.sh, and smina.static.  Alternatively you can use the following command to give execution permissions to all files in the DockM8 folder: `chmod +x **PATH_TO**/DockM8/software`

## Running DockM8 (via streamlit GUI)

DockM8 comes with a simple form-based GUI for ease of use. To launch it, run the following command :

`streamlit run **PATH_TO**/gui.py`

You can click the `localhost` link to access the GUI.

## Running DockM8 (via command-line / dockm8.py script)

1. Ensure the required files are available:
- protein/receptor file as a .pdb file
- If using a ligand to define the binding pocket : reference ligand file as an .sdf file
- docking library as a .sdf file

2. Open a terminal and activate the dockm8 python environment (`conda activate dockm8`)

3. Run the following command:

`python **PATH_TO**/dockm8.py --args`  

`--software`: The path to the software folder. In most cases this is where the DockM8 repository was downloaded to (`path/to/DockM8/software`)  
`--mode`: Choose mode with which to run dockm8. Options are:
  - 'single' : Regular docking on one receptor.
  - 'ensemble' : Ensemble docking on multiple receptor conformations.  

`--gen_decoys`: Whether or not to generate decoys suing DeepCoy for the supplied list of acive compounds. DockM8 will then determine the optimal combination of scoring functions, pose selection and consensus method for the protein target (True/False).  
`--decoy_model`: Model for decoy generation.  
`--n_decoys`: Number of decoys to generate per active compound (default:20).  
`--actives`: The path to the .sdf file containing the active ligands.  

`--receptor`: The path to the protein file (.pdb) or multiple paths (separated by spaces) if using ensemble mode.  
`--pocket`: The method to use for pocket determination. Must be one of:
  - 'reference' : Uses reference ligand to define pocket.
  - 'RoG' (radius of gyration) : Uses reference ligand's radius of gyration to define pocket.  
  - 'dogsitescorer' :  Use the DogSiteScorer webserver to determine pocket coordinates, works on pocket volume by default although this can be changed in *dogsitescorer.py*.  

`--reffile`: The path to the reference ligand to use for pocket determination. Must be provided if using 'reference' or 'RoG' pocket mode.  
`--docking_library`: The path to the docking library file (.sdf format).  
`--idcolumn`: The unique identifier column used in the docking library.  
`--conformers`: The method to use for compound protonation. Must be one of:
  - 'GypsumDL' : Use GypsumDL library to generate 3D conformers
  - 'RDKit' : Use RDKit MMFF forcefield to generate 3D conformers  

`--protonation`: The method to use for compound protonation. Must be one of:
  - 'GypsumDL' : Use GypsumDL library to protonate library
  - 'None' : Do not protonate library  

`--docking_programs`: The method(s) to use for docking. Must be one or more of:
  - 'GNINA'
  - 'SMINA'
  - 'QVINAW'
  - 'QVINA2'
  - 'PLANTS' 

`--bust_poses`: Whether or not to use the PoseBusters library to remove bad docking poses. WARNING: takes a long time to run.

`--pose_selection`: The method(s) to use for pose clustering. Must be one or more of:
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
`--ncpus`: The number of cpus to use for the workflow. Default behavior is to use 90% of the available cpus.  
`--clustering_method`: Which algorithm to use for clustering. Must be one of 'KMedoids', 'Aff_prop'. Must be set when using 'RMSD', 'spyRMSD', 'espsim', 'USRCAT' clustering metrics.  
`--rescoring`: Which scoring functions to use for rescoring. Must be one or more of 'GNINA-Affinity', 'CNN-Score', 'CNN-Affinity', 'AD4', 'CHEMPLP', 'RFScoreVS', 'LinF9', 'Vinardo', 'PLP', 'AAScore', 'SCORCH', 'RTMScore', 'NNScore', 'PLECScore', 'KORP-PL', 'ConvexPLR'.  
`--consensus`: Which consensus method to use. Must be one of 'ECR_best', 'ECR_avg', 'avg_ECR', 'RbR', 'RbV', 'Zscore_best', 'Zscore_avg'.  
`--threshold`: Threshold in % to use when using 'ensemble' mode. Will find the hits in common in the x% of top ranked compounds in all of the receptor conformations.

## Running DockM8 (via Jupyter Notebook)

1. Open dockm8.ipynb, dockm8_ensemble.ipynb or dockm8_decoys.ipynb in your favorite IDE, depending on which DockM8 mode you want to use.

2. Follow the instructions in the Markdown cells

## Acknowledgements

## Citation

## License (NEEDS CHECKING...)

This project is licensed under the MIT License - see the [LICENSE.md](https://gitlab.com/Tonylac77/DockM8/-/blob/main/LICENSE) file for details.

## Contributing

We highly encourage contributions from the community - see the [CONTRIBUTING.md](https://gitlab.com/Tonylac77/DockM8/-/blob/main/CONTRIBUTING.md) file for details.


# Compound cleaning

- Implement Gypsum-DL - **DONE**
- Implement ChemBL Pipeline Standardisation - **DONE**

# Docking
- Implement PLANTS - **DONE**
- Implement LeDock
    - WIP: .dok to .sdf translation still giving issues
        - Solution 1 : Solve Pybel installation problem which prevents using script found here https://github.com/AngelRuizMoreno/Jupyter_Dock
            - HOW : Contact OpenBabel forum/mailing list for better installation instructions
        - Solution 2 : Make own translator
            - Current version has a translator I made, coordinates in SDF file seem to be shuffled around...needs fixing
- Implement SMINA - **DONE**
- Implement GNINA - **DONE**

# Clustering
- Implement RMSD - **DONE**
- Implement spyRMSD - **DONE**
- Implement ElectroShapeSim - **DONE**

# Rescoring
- Implement SMINA - **DONE**
- Implement VINARDO - **DONE**
- Implement GNINA - **DONE**
- Implement CHEMPLP/PLP (PLANTS) - **DONE**
- Implement RF-Score-VS - **DONE**

# Ranking
- Implement ECR
- Implement Z-Score

# Performance
- Choose targets for performance determination
    - number of targets
    - difficulty
    - number of ligands available
- Check PDBBind suitability
- Check DUD-E suitability
- Check for other performance measuring datasets
- Choose metrics for performance (RMSD, BindingAff, Enrichement Factor, )

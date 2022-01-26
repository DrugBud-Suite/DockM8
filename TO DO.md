# Compound cleaning

[x] Implement Gypsum-DL
[x] Implement ChemBL Pipeline Standardisation

# Docking
[x] Implement PLANTS
[ ] Implement LeDock
    - WIP: dok to sdf translation still giving issues
        (Pybel installation problems prevents using script found here https://github.com/AngelRuizMoreno/Jupyter_Dock)
            Solution : Contact OpenBabel forum/mailing list for better installation instructions
        Current version has a translator I made, coordinates in SDF file seem to be shuffled around...needs fixing
[x] Implement SMINA
[x] Implement GNINA

# Clustering
[x] Implement RMSD
[x] Implement spyRMSD
[x] Implement ElectroShapeSim

# Rescoring
[x] Implement SMINA
[x] Implement VINARDO
[x] Implement GNINA
[x] Implement CHEMPLP/PLP (PLANTS)
[x] Implement RF-Score-VS

# Ranking
[ ] Implement ECR
[ ] Implement Z-Score

# Performance
[ ] Choose targets for performance determination
    - number of targets
    - difficulty
    - number of ligands available
[ ] Check PDBBind suitability
[ ] Check DUD-E suitability
[ ] Check for other performance measuring datasets
[ ] Choose metrics for performance (RMSD, BindingAff, Enrichement Factor, )

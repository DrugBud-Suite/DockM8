import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
from posebusters import PoseBusters
from rdkit.Chem import PandasTools
from yaml import safe_load

# Search for 'DockM8' in parent directories
dockm8_path = next((p / 'DockM8' for p in Path(__file__).resolve().parents if (p / 'DockM8').is_dir()), None)
sys.path.append(str(dockm8_path))

from scripts.docking.gnina import fetch_gnina_poses, gnina_docking
from scripts.docking.plants import fetch_plants_poses, plants_docking
from scripts.docking.qvina2 import fetch_qvina2_poses, qvina2_docking
from scripts.docking.qvinaw import fetch_qvinaw_poses, qvinaw_docking
from scripts.docking.smina import fetch_smina_poses, smina_docking
from scripts.docking.psovina import fetch_psovina_poses, psovina_docking
from scripts.docking.vinaXB import fetch_vinaXB_poses, vinaXB_docking
from scripts.utilities.utilities import (
    parallel_executor,
    parallel_SDF_loader,
    printlog,
    split_sdf_str,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

DOCKING_PROGRAMS = ['PLANTS', 'SMINA', 'GNINA', 'QVINA2', 'QVINAW']

# Map docking program names to their respective functions
DOCKING_PROGRAMS = {
    "SMINA": [smina_docking, fetch_smina_poses],
    "GNINA": [gnina_docking, fetch_gnina_poses],
    "PLANTS": [plants_docking, fetch_plants_poses],
    "QVINAW": [qvinaw_docking, fetch_qvinaw_poses],
    "QVINA2": [qvina2_docking, fetch_qvina2_poses],
    "VINAXB": [vinaXB_docking, fetch_vinaXB_poses],
    "PSOVINA": [psovina_docking, fetch_psovina_poses],
}

def dockm8_docking(
    w_dir: Path,
    protein_file: Path,
    pocket_definition : dict,
    software: Path,
    docking_programs: list,
    exhaustiveness: int,
    n_poses: int,
    n_cpus: int,
    job_manager="concurrent_process",
):
    """Dock ligands into a protein binding site using one or more docking programs.

    Args:
        w_dir (Path): The working directory where the docking results will be saved.
        protein_file (Path): The path to the protein file.
        pocket_definition (dict): A dictionary defining the pocket for docking.
        software (Path): The path to the docking software.
        docking_programs (list): A list of docking programs to use.
        exhaustiveness (int): The exhaustiveness parameter for docking.
        n_poses (int): The number of poses to generate.
        n_cpus (int): The number of CPUs to use for parallel docking.
        job_manager (str, optional): The job manager to use for parallel docking. Defaults to "concurrent_process".
    """
    if n_cpus == 1:
        printlog("Running docking using 1 CPU...")
        for program in docking_programs:
            docking_function, fetch_function = DOCKING_PROGRAMS[program]
            if not (w_dir / program.lower()).exists():
                docking_function(None,
                    w_dir,
                    protein_file,
                    pocket_definition,
                    software,
                    exhaustiveness,
                    n_poses
                )
            if (w_dir / program.lower()).exists() and not (w_dir / program.lower() / f"{program.lower()}_poses.sdf").exists():
                fetch_function(w_dir, n_poses, software)
    else:
        printlog(f"Running docking using {n_cpus} CPUs...")
        split_final_library_path = w_dir / "split_final_library"
        if not split_final_library_path.exists():
            split_final_library_path = split_sdf_str(w_dir, w_dir / "final_library.sdf", n_cpus)
        else:
            printlog("Split final library folder already exists...")
        split_files_sdfs = [
            split_final_library_path / file
            for file in os.listdir(split_final_library_path)
            if file.endswith(".sdf")
        ]
        for program in docking_programs:
            docking_function, fetch_function = DOCKING_PROGRAMS[program]
            if not (w_dir / program.lower()).exists() or not any((w_dir / program.lower()).iterdir()):
                parallel_executor(
                    docking_function,
                    split_files_sdfs,
                    n_cpus,
                    job_manager,
                    w_dir=w_dir,
                    protein_file=protein_file,
                    pocket_definition=pocket_definition,
                    software=software,
                    exhaustiveness=exhaustiveness,
                    n_poses=n_poses,
                )
            if (w_dir / program.lower()).exists() and not (w_dir / program.lower() / f"{program.lower()}_poses.sdf").exists():
                fetch_function(w_dir, n_poses, software)
    shutil.rmtree(w_dir / "split_final_library", ignore_errors=True)

def concat_all_poses(w_dir : Path, docking_programs : list, protein_file : Path, n_cpus : int, bust_poses : bool):
    """
    Concatenates all poses from the specified docking programs and checks them for quality using PoseBusters.
    
    Args:
    w_dir (str): Working directory where the docking program output files are located.
    docking_programs (list): List of strings specifying the names of the docking programs used.
    protein_file (str): Path to the protein file used for docking.
    
    Returns:
    None
    """
    # Create an empty DataFrame to store all poses
    all_poses = pd.DataFrame()
    for program in docking_programs:
        df = parallel_SDF_loader(f"{w_dir}/{program.lower()}/{program.lower()}_poses.sdf", molColName='Molecule', idName='Pose ID', n_cpus=n_cpus)
        all_poses = pd.concat([all_poses, df])
    if bust_poses:
        try:
            tic = time.perf_counter()
            printlog("Busting poses...")
            # Initialize PoseBusters with the configuration file
            buster = PoseBusters(config=safe_load(open('./scripts/posebusters_config.yml')))
            # Add the protein file path as a column in all_poses DataFrame
            all_poses['mol_cond'] = str(protein_file)
            # Rename the 'Molecule' column to 'mol_pred'
            all_poses = all_poses.rename(columns={'Molecule':'mol_pred'})
            # Check the quality of poses using PoseBusters
            df = buster.bust_table(all_poses)
            # Remove rows where any of the specified columns is 'False'
            cols_to_check = ['all_atoms_connected', 'bond_lengths', 'bond_angles', 'internal_steric_clash', 'aromatic_ring_flatness', 'double_bond_flatness', 'protein-ligand_maximum_distance']
            df = df.loc[(df[cols_to_check] is not False).all(axis=1)]
            df.reset_index(inplace=True)
            # Filter the all_poses DataFrame based on the valid poses identified by PoseBusters
            all_poses = all_poses[all_poses['Pose ID'].isin(df['molecule'])].rename(columns={'mol_pred':'Molecule'})  
            toc = time.perf_counter()
            printlog(f"PoseBusters checking completed in {toc - tic:.2f} seconds.")
        except Exception as e:
            printlog('ERROR: Failed to check poses with PoseBusters!')     
            printlog(e)
    else:
        pass
    try:
        # Write the combined poses to an SDF file
        PandasTools.WriteSDF(all_poses,
                            f"{w_dir}/allposes.sdf",
                            molColName='Molecule',
                            idName='Pose ID',
                            properties=list(all_poses.columns))
        printlog('All poses succesfully checked and combined!')
    except Exception as e:
        printlog('ERROR: Failed to write all_poses SDF file!')        
        printlog(e)
    return

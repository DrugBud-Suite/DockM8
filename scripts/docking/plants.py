import os
import subprocess
from pathlib import Path
from typing import Dict, Union

import pandas as pd
from rdkit.Chem import PandasTools
from tqdm import tqdm

from scripts.utilities import convert_molecules, delete_files, printlog


def plants_docking(
    split_file: Path,
    w_dir: Path,
    protein_file: Path,
    pocket_definition: Dict[str, list],
    software: Path,
    exhaustiveness: int,
    n_poses: int,
):
    """
    Perform docking using the PLANTS software, optionally handling either a full library or individual split files.

    Args:
        w_dir (Path): Working directory for docking operations.
        protein_file (Path): Path to the protein file in PDB format.
        pocket_definition (dict): Dictionary containing the center and size of the docking pocket.
        software (Path): Path to the PLANTS software folder.
        n_poses (int): Number of poses to generate.
        split_file (Path, optional): Path to the split file containing the ligands to dock. If None, uses the entire library.

    Returns:
        str: Path to the results file in SDF format.
    """
    plants_folder = w_dir / "plants"
    plants_folder.mkdir(parents=True, exist_ok=True)

    if split_file:
        input_file = split_file
        result_subfolder = f"results_{split_file.stem}"
    else:
        input_file = w_dir / "final_library.sdf"
        result_subfolder = "results"

    results_dir = plants_folder / result_subfolder

    # Convert input files to mol2 format using open babel
    plants_protein_mol2 = plants_folder / "protein.mol2"
    convert_molecules(protein_file, plants_protein_mol2, "pdb", "mol2")
    plants_ligands_mol2 = plants_folder / f"{input_file.stem}.mol2"
    convert_molecules(input_file, plants_ligands_mol2, "sdf", "mol2")

    # Generate PLANTS config file
    plants_docking_config_path = plants_folder / f"{split_file.stem}.config"
    plants_config = generate_plants_config(
        plants_protein_mol2,
        plants_ligands_mol2,
        pocket_definition,
        n_poses,
        results_dir,
    )
    with plants_docking_config_path.open("w") as config_writer:
        config_writer.writelines(plants_config)

    # Run PLANTS docking
    try:
        plants_docking_command = f'{software / "PLANTS"} --mode screen ' + str(
            plants_docking_config_path
        )
        subprocess.call(
            plants_docking_command,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    except Exception as e:
        printlog("ERROR: PLANTS docking command failed...")
        printlog(e)

    # Post-process results
    results_mol2 = results_dir / "docked_ligands.mol2"
    results_sdf = results_mol2.with_suffix(".sdf")
    try:
        convert_molecules(results_mol2, results_sdf, "mol2", "sdf")
    except Exception as e:
        printlog("ERROR: Failed to convert PLANTS poses file to .sdf!")
        printlog(e)
    return


def generate_plants_config(
    protein_mol2: Path,
    ligands_mol2: Path,
    pocket_definition: dict,
    n_poses: int,
    output_dir: Path,
):
    """
    Helper function to generate PLANTS configuration.

    Args:
        protein_mol2 (Path): Path to the protein mol2 file.
        ligands_mol2 (Path): Path to the ligands mol2 file.
        pocket_definition (dict): Dictionary containing the pocket definition with keys 'center' and 'size'.
        n_poses (int): Number of poses to generate.
        output_dir (Path): Path to the output directory.

    Returns:
        list: List of strings representing the PLANTS configuration lines.
    """
    config_lines = [
        "# search algorithm\n",
        "search_speed speed1\n",
        "aco_ants 20\n",
        "flip_amide_bonds 0\n",
        "flip_planar_n 1\n",
        "force_flipped_bonds_planarity 0\n",
        "force_planar_bond_rotation 1\n",
        "rescore_mode simplex\n",
        "flip_ring_corners 0\n",
        "# scoring functions\n",
        "scoring_function chemplp\n",
        "outside_binding_site_penalty 50.0\n",
        "enable_sulphur_acceptors 1\n",
        "# Intramolecular ligand scoring\n",
        "ligand_intra_score clash2\n",
        "chemplp_clash_include_14 1\n",
        "chemplp_clash_include_HH 0\n",
        "# input\n",
        f"protein_file {protein_mol2}\n",
        f"ligand_file {ligands_mol2}\n",
        "# output\n",
        f"output_dir {output_dir}\n",
        "# write single mol2 files\n",
        "write_multi_mol2 1\n",
        "# binding site definition\n",
        f'bindingsite_center {pocket_definition["center"][0]} {pocket_definition["center"][1]} {pocket_definition["center"][2]}\n',
        f'bindingsite_radius {pocket_definition["size"][0] / 2}\n',
        "# cluster algorithm\n",
        f"cluster_structures {n_poses}\n",
        "cluster_rmsd 2.0\n",
        "# write\n",
        "write_ranking_links 0\n",
        "write_protein_bindingsite 0\n",
        "write_protein_conformations 0\n",
        "write_protein_splitted 0\n",
        "write_merged_protein 0\n",
        "####\n",
    ]
    return config_lines


def fetch_plants_poses(w_dir: Union[str, Path], n_poses: int, software: Path, *args):
    """
    Fetches PLANTS docking poses from the specified directory and converts them to an SDF file.

    Args:
        w_dir (Path): The working directory where the PLANTS docking results are located.
        software (Path): The path to the PLANTS software.
        n_poses (int): The number of poses to fetch.

    Returns:
        None
    """
    if (w_dir / "plants").is_dir() and not (
        w_dir / "plants" / "plants_poses.sdf"
    ).is_file():
        plants_dataframes = []
        results_folders = [folder for folder in os.listdir(w_dir / "plants")]
        for folder in tqdm(results_folders, desc="Fetching PLANTS docking poses"):
            if folder.startswith("results"):
                file_path = w_dir / "plants" / folder / "docked_ligands.mol2"
                if file_path.is_file():
                    try:
                        convert_molecules(
                            file_path, file_path.with_suffix(".sdf"), "mol2", "sdf"
                        )
                        plants_poses = PandasTools.LoadSDF(
                            str(file_path.with_suffix(".sdf")),
                            idName="ID",
                            molColName="Molecule",
                            includeFingerprints=False,
                            embedProps=False,
                            removeHs=False,
                            strictParsing=True,
                        )
                        plants_scores = pd.read_csv(
                            str(file_path).replace("docked_ligands.mol2", "ranking.csv")
                        ).rename(
                            columns={"LIGAND_ENTRY": "ID", "TOTAL_SCORE": "CHEMPLP"}
                        )[["ID", "CHEMPLP"]]
                        plants_df = pd.merge(plants_scores, plants_poses, on="ID")
                        plants_df["ID"] = plants_df["ID"].str.split("_").str[0]
                        list_ = [*range(1, int(n_poses) + 1, 1)]
                        ser = list_ * (len(plants_df) // len(list_))
                        plants_df["Pose ID"] = [
                            f'{row["ID"]}_PLANTS_{num}'
                            for num, (_, row) in zip(
                                ser + list_[: len(plants_df) - len(ser)],
                                plants_df.iterrows(),
                            )
                        ]
                        plants_dataframes.append(plants_df)
                    except Exception as e:
                        printlog(
                            "ERROR: Failed to convert PLANTS docking results file to .sdf!"
                        )
                        printlog(e)
            else:
                pass
        try:
            plants_df = pd.concat(plants_dataframes)
            PandasTools.WriteSDF(
                plants_df,
                str(w_dir / "plants" / "plants_poses.sdf"),
                molColName="Molecule",
                idName="Pose ID",
                properties=list(plants_df.columns),
            )
            files = Path(os.getcwd()).glob("*.pid")
            for file in files:
                file.unlink()
        except Exception as e:
            printlog("ERROR: Failed to write combined PLANTS docking poses")
            printlog(e)
        else:
            delete_files(w_dir / "plants", "plants_poses.sdf")

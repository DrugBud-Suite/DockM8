import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from yaml import safe_load

# Search for 'DockM8' in parent directories
scripts_path = next((p / 'scripts' for p in Path(__file__).resolve().parents if (p / 'scripts').is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.utilities import printlog


def process_chunk(data_chunk, config_path, protein_file):
    from posebusters import PoseBusters

    # Initialize PoseBusters with the configuration file
    buster = PoseBusters(config=safe_load(open(config_path)))
    data_chunk["mol_cond"] = str(protein_file)  # Adding a new column based on protein_file
    data_chunk = data_chunk.rename(columns={"Molecule": "mol_pred"})  # Renaming a column

    # Apply PoseBusters processing and preserve all existing columns
    busted_df = buster.bust_table(data_chunk)
    # Combine the original data_chunk with new columns from df
    combined_df = pd.concat([data_chunk.reset_index(drop=True), busted_df.reset_index(drop=True)], axis=1)
    combined_df = combined_df.rename(columns={'mol_pred': 'Molecule'})
    cols_to_check = [
        "all_atoms_connected",
        "bond_lengths",
        "bond_angles",
        "internal_steric_clash",
        "aromatic_ring_flatness",
        "double_bond_flatness",
        "protein-ligand_maximum_distance",
    ]

    # Filter rows based on conditions, but keep all columns
    valid_indices = combined_df[cols_to_check].all(axis=1)
    df_final = combined_df[valid_indices]

    return df_final if df_final.shape[0] > 0 else None


def pose_buster(
    dataframe: pd.DataFrame, protein_file: Path, n_cpus: int = os.cpu_count()
):
    tic = time.perf_counter()
    printlog("Busting poses...")
    config_path = (
        str(dockm8_path)
        + "/scripts/docking_postprocessing/posebusters/posebusters_config.yml"
    )

    # Splitting DataFrame into chunks for parallel processing
    chunks = np.array_split(dataframe, len(dataframe))
    results = []
    with ProcessPoolExecutor(max_workers=n_cpus) as executor:
        futures = [
            executor.submit(process_chunk, chunk, config_path, protein_file)
            for chunk in chunks
        ]
        for future in futures:
            results.append(future.result())
    # Concatenate all processed chunks
    concatenated_df = pd.concat(results)
    concatenated_df.reset_index(inplace=True, drop=True)
    dataframe = dataframe[dataframe["Pose ID"].isin(concatenated_df["Pose ID"])]
    toc = time.perf_counter()
    printlog(f"PoseBusters checking completed in {toc - tic:.2f} seconds.")
    return dataframe

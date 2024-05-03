import concurrent.futures
import os
import sys
from pathlib import Path

import pandas as pd
from chembl_structure_pipeline import standardizer
from rdkit import Chem
from rdkit.Chem import PandasTools
from tqdm import tqdm

cwd = Path.cwd()
dockm8_path = cwd.parents[0] / "DockM8"
sys.path.append(str(dockm8_path))

from scripts.utilities import printlog


def standardize_molecule(molecule):
    """Standardize a single molecule using ChemBL Structure Pipeline."""
    standardized_molecule = standardizer.standardize_mol(molecule)
    standardized_molecule = standardizer.get_parent_mol(standardized_molecule)
    return standardized_molecule


def standardize_library(input_sdf: Path, output_dir: Path, id_column: str, ncpus: int):
    """
    Standardizes a docking library using the ChemBL Structure Pipeline.

    Args:
        input_sdf (Path): Path to the input SDF file containing the docking library.
        output_dir (Path): Directory where the standardized SDF file will be saved.
        id_column (str): Column name containing the compound IDs.
        ncpus (int): Number of CPUs for parallel processing.

    Raises:
        Exception: If there is an error loading, processing, or writing the SDF file.
    """
    printlog("Standardizing docking library using ChemBL Structure Pipeline...")

    # Load the input SDF file and preprocess the data
    try:
        df = PandasTools.LoadSDF(
            str(input_sdf),
            idName=id_column,
            molColName="Molecule",
            includeFingerprints=False,
            embedProps=True,
            removeHs=True,
            strictParsing=True,
            smilesName="SMILES",
        )
        df.rename(columns={id_column: "ID"}, inplace=True)
        df["ID"] = ["DOCKM8-" + str(id) if str(id).isdigit() else id for id in df["ID"]]
        df["ID"] = df["ID"].str.replace("_", "-")
        n_cpds_start = len(df)
    except Exception as e:
        printlog(f"ERROR: Failed to Load library SDF file! {str(e)}")
        raise

    # Convert SMILES strings to RDKit molecules
    try:
        df.drop(columns="Molecule", inplace=True)
        df["Molecule"] = [Chem.MolFromSmiles(smiles) for smiles in df["SMILES"]]
    except Exception as e:
        printlog(f"ERROR: Failed to convert SMILES to RDKit molecules! {str(e)}")
        raise

    # Standardize the molecules using ChemBL Structure Pipeline
    if ncpus == 1:
        df["Molecule"] = [standardize_molecule(mol) for mol in df["Molecule"]]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=ncpus) as executor:
            df["Molecule"] = list(
                tqdm(
                    executor.map(standardize_molecule, df["Molecule"]),
                    total=len(df["Molecule"]),
                    desc="Standardizing molecules",
                    unit="mol",
                )
            )

    # Clean up the dataframe and calculate the number of compounds lost during standardization
    df[["Molecule", "flag"]] = pd.DataFrame(df["Molecule"].tolist(), index=df.index)
    df.drop(columns="flag", inplace=True)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    n_cpds_end = len(df)
    printlog(
        f"Standardization finished: Started with {n_cpds_start}, ended with {n_cpds_end}: {n_cpds_start - n_cpds_end} compounds lost"
    )

    # Write the standardized library to an SDF file
    output_file = output_dir / "standardized_library.sdf"
    try:
        PandasTools.WriteSDF(
            df, str(output_file), molColName="Molecule", idName="ID", allNumeric=True
        )
    except Exception as e:
        printlog(f"ERROR: Failed to write standardized library SDF file! {str(e)}")
        raise
    return output_file

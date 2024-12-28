import pandas as pd
from rdkit import Chem
from chembl_structure_pipeline import standardizer
import sys
import re
from typing import Optional, Tuple
from pathlib import Path

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.parallel_executor import parallel_executor
from scripts.utilities.logging import printlog


def standardize_molecule(molecule: Chem.Mol, remove_salts: bool = True) -> Tuple[Optional[Chem.Mol], Optional[str]]:
    """Standardize a single molecule using ChEMBL Structure Pipeline.

    Args:
        molecule (Chem.Mol): The molecule to be standardized.
        remove_salts (bool): Whether to remove salts from the molecule.

    Returns:
        Tuple[Optional[Chem.Mol], Optional[str]]: Standardized molecule and error message if any.
    """
    try:
        std_molecule = standardizer.standardize_mol(molecule)
        if remove_salts:
            std_molecule = standardizer.get_parent_mol(std_molecule)[0]  # Get parent molecule
        return std_molecule, None
    except Exception as e:
        return None, str(e)


def standardize_ids(df: pd.DataFrame, id_column: str = "ID") -> pd.DataFrame:
    """Standardizes compound IDs in the DataFrame."""
    if df[id_column].isnull().all():
        df[id_column] = [f"DOCKM8{i+1:06d}" for i in range(len(df))]
    else:
        df[id_column] = df[id_column].astype(str).apply(lambda x: f"DOCKM8{x.zfill(6)}" if x.isdigit() else x)
        df[id_column] = df[id_column].apply(lambda x: re.sub(r"[^a-zA-Z0-9]", "", x))
    return df


def standardize_library(
    df: pd.DataFrame,
    smiles_column: str = None,
    remove_salts: bool = True,
    standardize_ids_flag: bool = True,
    n_cpus: int = None,
) -> pd.DataFrame:
    """Standardizes a compound library using ChEMBL Structure Pipeline."""
    printlog("Standardizing compound library using ChEMBL Structure Pipeline...")

    # Convert SMILES to molecules if needed
    if "Molecule" not in df.columns:
        if not smiles_column:
            raise ValueError("No molecule column found and no SMILES column specified")
        df["Molecule"] = df[smiles_column].apply(Chem.MolFromSmiles)

    # Standardize IDs if requested
    if standardize_ids_flag:
        df = standardize_ids(df)

    # Standardize molecules in parallel
    results = parallel_executor(
        standardize_molecule,
        df["Molecule"].tolist(),
        n_cpus,
        "concurrent_process",
        display_name="Molecule Standardization",
        remove_salts=remove_salts,
    )

    # Process results
    molecules, errors = zip(*results)
    df["Molecule"] = molecules

    # Remove failed entries
    n_cpds_start = len(df)
    df = df.dropna(subset=["Molecule"])
    n_cpds_end = len(df)

    printlog(f"Standardization complete: {n_cpds_start - n_cpds_end} compounds removed")

    return df

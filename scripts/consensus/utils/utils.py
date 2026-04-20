"""This module provides utility functions for loading and processing data.

Including functions to load data from CSV or SDF files and to weigh dataframes.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools


def load_data(file_path: str | Path) -> pd.DataFrame:
    """Load data from CSV or SDF file.

    Args:
        file_path (Union[str, Path]): Path to the input file.

    Returns:
        pd.DataFrame: Loaded data.

    Raises:
        ValueError: If the file format is not supported.
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    if file_path.suffix.lower() == ".sdf":
        return load_sdf(file_path)
    msg = f"Unsupported file format: {file_path.suffix}"
    raise ValueError(msg)


def load_sdf(file_path: Path) -> pd.DataFrame:
    """Load data from an SDF file.

    Args:
        file_path (Path): Path to the SDF file.

    Returns:
        pd.DataFrame: Loaded data with RDKit molecules.
    """
    df = PandasTools.LoadSDF(str(file_path), molColName="ROMol", includeFingerprints=False)

    # Convert RDKit molecules to SMILES strings
    df["SMILES"] = df["ROMol"].apply(lambda x: Chem.MolToSmiles(x) if x is not None else None)

    # Drop the ROMol column as it's not easily serializable
    return df.drop("ROMol", axis=1)

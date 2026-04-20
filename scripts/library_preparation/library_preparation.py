from pathlib import Path
from typing import Optional, Union
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
import sys

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.library_preparation.standardization.standardize import standardize_library
from scripts.library_preparation.conformer_generation.confgen_rdkit import generate_conformers_RDKit
from scripts.library_preparation.conformer_generation.confgen_GypsumDL import generate_conformers_GypsumDL
from scripts.library_preparation.protonation.protgen_GypsumDL import protonate_GypsumDL

PROTONATION_OPTIONS = ["GypsumDL", "None"]
CONFORMER_OPTIONS = ["UFF", "MMFF", "GypsumDL"]


def prepare_library(
    input_data: Union[pd.DataFrame, Path],
    protonation: str,
    conformers: str,
    software: Path,
    n_cpus: int,
    standardize_ids: bool = True,
    remove_salts: bool = True,
    min_ph: float = 6.4,
    max_ph: float = 8.4,
    pka_precision: float = 1.0,
    output_sdf: Optional[Path] = None,
) -> pd.DataFrame:
    """Main function for preparing compound library."""

    # Load input data
    if isinstance(input_data, pd.DataFrame):
        input_df = input_data
        if "Molecule" not in input_df.columns:
            try:
                input_df["Molecule"] = input_df["SMILES"].apply(Chem.MolFromSmiles)
            except Exception as e:
                raise ValueError(f"Failed to process molecule data: {str(e)}")
    else:
        input_df = PandasTools.LoadSDF(str(input_data), molColName="Molecule", idName="ID")

    # Standardization
    standardized_df = standardize_library(
        input_df, remove_salts=remove_salts, standardize_ids_flag=standardize_ids, n_cpus=n_cpus
    )

    # Protonation
    if protonation == "GypsumDL":
        protonated_df = protonate_GypsumDL(standardized_df, software, n_cpus, min_ph, max_ph, pka_precision)
    else:
        protonated_df = standardized_df

    # Conformer generation
    if conformers in ["MMFF", "UFF"]:
        final_df = generate_conformers_RDKit(protonated_df, n_cpus, forcefield=conformers)
    elif conformers == "GypsumDL":
        final_df = generate_conformers_GypsumDL(protonated_df, software, n_cpus)
    else:
        raise ValueError(f"Invalid conformer method: {conformers}")

    # Finalize output
    final_df = final_df[["Molecule", "ID"]]

    if output_sdf:
        PandasTools.WriteSDF(
            final_df, str(output_sdf), molColName="Molecule", idName="ID", properties=list(final_df.columns)
        )
        return output_sdf

    return final_df

import sys
import warnings
from pathlib import Path

from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
scripts_path = next((p / 'scripts' for p in Path(__file__).resolve().parents if (p / 'scripts').is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.library_preparation.conformer_generation.confgen_GypsumDL import (
    generate_conformers_GypsumDL,
)
from scripts.library_preparation.conformer_generation.confgen_RDKit import (
    generate_conformers_RDKit,
)
from scripts.library_preparation.protonation.protgen_GypsumDL import protonate_GypsumDL
from scripts.library_preparation.standardisation.standardise import standardize_library
from scripts.utilities.utilities import printlog

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

PROTONATION_OPTIONS = ["GypsumDL", "None"]
CONFORMER_OPTIONS = ["RDKit", "MMFF", "GypsumDL"]

def prepare_library(
    input_sdf: str,
    output_dir: Path,
    id_column: str,
    protonation: str,
    conformers: str,
    software: Path,
    n_cpus: int,
    n_conformers: int = 1
):
    """
    Prepares a docking library for further analysis.

    Args:
        input_sdf (str): The path to the input SDF file containing the docking library.
        id_column (str): The name of the column in the SDF file that contains the compound IDs.
        protonation (str): The method to use for protonation. Can be 'GypsumDL', or 'None' for no protonation.
        n_cpus (int): The number of CPUs to use for parallelization.
    """
    standardized_sdf = output_dir / "standardized_library.sdf"

    if not standardized_sdf.is_file():
        standardize_library(input_sdf, output_dir, id_column, n_cpus)

    if protonation == "GypsumDL":
        protonated_sdf = output_dir / "protonated_library.sdf"
        if not protonated_sdf.is_file():
            protonate_GypsumDL(standardized_sdf, output_dir, software, n_cpus)
    elif protonation == "None":
        protonated_sdf = standardized_sdf
    else:
        raise ValueError(
            f'Invalid protonation method specified : {protonation}. Must be either "None" or "GypsumDL".'
        )

    if conformers == "RDKit" or conformers == "MMFF":
        generate_conformers_RDKit(protonated_sdf, output_dir, n_cpus)
    elif conformers == "GypsumDL":
        generate_conformers_GypsumDL(protonated_sdf, output_dir, software, n_cpus, n_conformers)
    else:
        raise ValueError(
            f'Invalid conformer method specified : {conformers}. Must be either "RDKit", "MMFF" or "GypsumDL".'
        )

    printlog("Cleaning up files...")
    
    final_library_df = PandasTools.LoadSDF(str(output_dir / "generated_conformers.sdf"), molColName="Molecule", idName="ID")
    final_library_df[["Molecule", "ID"]]
    PandasTools.WriteSDF(final_library_df, str(output_dir / "final_library.sdf"), molColName="Molecule", idName="ID")

    # Delete the temporary files generated during the library preparation process
    (output_dir / "standardized_library.sdf").unlink(missing_ok=True)
    (output_dir / "protonated_library.sdf").unlink(missing_ok=True)
    (output_dir / "generated_conformers.sdf").unlink(missing_ok=True)

    printlog("Preparation of compound library finished.")
    return output_dir / "final_library.sdf"

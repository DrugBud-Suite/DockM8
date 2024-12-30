import sys
from pathlib import Path
import os
from typing import Union, Optional

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.protein_preparation.fetching.fetch_alphafold import fetch_alphafold_structure
from scripts.protein_preparation.fetching.fetch_pdb import fetch_pdb_structure
from scripts.protein_preparation.fixing.pdb_fixer import fix_pdb_file
from scripts.protein_preparation.protonation.protonate_protoss import protonate_protein_protoss
from scripts.utilities.logging import printlog
import requests


def prepare_protein(
    protein_file_or_code: Union[str, Path],
    output_dir: Optional[Path] = None,
    fix_nonstandard_residues: bool = True,
    fix_missing_residues: bool = True,
    protonation_method: Union[str, float] = "protoss",
    remove_hetero: bool = True,
    remove_water: bool = True,
) -> Path:
    """
    Prepare a protein structure by performing various modifications.

    Args:
        protein_file_or_code (str or Path): The protein_file_or_code value. Can be a PDB code, Uniprot code, or file path.
        output_dir (Path, optional): The directory where the prepared protein structure will be saved.
        fix_nonstandard_residues (bool, optional): Whether to fix nonstandard residues. Default is True.
        fix_missing_residues (bool, optional): Whether to fix missing residues. Default is True.
        protonation_method (str or float): Method for protein protonation. Can be:
            - "pdbfixer": Use PDBFixer with pH 7.0
            - "protoss": Use Protoss
            - float value: Use PDBFixer with the specified pH value
        remove_hetero (bool, optional): Whether to remove heteroatoms. Default is True.
        remove_water (bool, optional): Whether to remove water molecules. Default is True.

    Returns:
        Path: The path to the prepared protein structure.

    Raises:
        ValueError: If the input parameters are invalid or if the protein structure cannot be prepared.
    """
    prepared_receptor_path = output_dir / "prepared_receptor.pdb"
    printlog(f"Checking validity of receptor input: {protein_file_or_code}")

    if not prepared_receptor_path.exists():
        # Validate and identify input type
        protein_type = None
        original_input = Path(protein_file_or_code) if isinstance(protein_file_or_code, (str, Path)) else None
        protein_file_or_code = str(protein_file_or_code)

        if len(protein_file_or_code) == 4 and protein_file_or_code.isalnum():
            url = f"https://www.rcsb.org/structure/{protein_file_or_code}"
            response = requests.head(url)
            if response.status_code == 200:
                protein_type = "PDB"
            else:
                raise ValueError(f"Invalid PDB code: {protein_file_or_code}")

        elif len(protein_file_or_code) == 6 and protein_file_or_code.isalnum():
            url = f"https://www.uniprot.org/uniprotkb/{protein_file_or_code}/entry"
            response = requests.head(url)
            if response.status_code == 200:
                protein_type = "Uniprot"
            else:
                raise ValueError(f"Invalid Uniprot code: {protein_file_or_code}")

        elif Path(protein_file_or_code).is_file():
            protein_type = "File"
            original_input = Path(protein_file_or_code).resolve()
        else:
            raise ValueError(
                f"Invalid input: {protein_file_or_code}. Must be a valid PDB code, Uniprot code, or file path."
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        # Validate protonation method
        if isinstance(protonation_method, str):
            protonation_method = protonation_method.lower()
            if protonation_method not in ["pdbfixer", "protoss"]:
                raise ValueError("Protonation method must be 'pdbfixer', 'protoss', or a pH value")
        elif isinstance(protonation_method, (int, float)):
            if not 0 <= protonation_method <= 14:
                raise ValueError("pH value must be between 0 and 14")
        else:
            raise ValueError("Protonation method must be 'pdbfixer', 'protoss', or a pH value")

        try:
            # Fetch the protein structure
            if protein_type == "PDB":
                fetched_pdb = fetch_pdb_structure(protein_file_or_code, output_dir)
            elif protein_type == "Uniprot":
                fetched_pdb = fetch_alphafold_structure(protein_file_or_code, output_dir)
                if not fetched_pdb or not fetched_pdb.exists():
                    raise ValueError(f"Failed to fetch AlphaFold structure for Uniprot code: {protein_file_or_code}")
            else:
                fetched_pdb = original_input

            # Only proceed if we have a valid PDB file
            if not fetched_pdb or not fetched_pdb.exists():
                raise ValueError(f"Failed to obtain valid structure for: {protein_file_or_code}")

            # Create a working copy in the output directory if input is a file
            if protein_type == "File":
                working_copy = output_dir / fetched_pdb.name
                if working_copy != fetched_pdb:
                    from shutil import copy2

                    copy2(fetched_pdb, working_copy)
                fetched_pdb = working_copy

            # Determine protonation settings
            use_protoss = False
            ph_value = None

            if isinstance(protonation_method, (int, float)):
                ph_value = protonation_method
            elif protonation_method == "pdbfixer":
                ph_value = 7.0
            elif protonation_method == "protoss":
                use_protoss = True

            # Fix the protein structure
            if (
                fix_nonstandard_residues
                or fix_missing_residues
                or ph_value is not None
                or remove_hetero
                or remove_water
            ):
                fixed_pdb = fix_pdb_file(
                    fetched_pdb,
                    output_dir,
                    fix_nonstandard_residues,
                    fix_missing_residues,
                    ph_value,
                    remove_hetero,
                    remove_water,
                )
            else:
                fixed_pdb = fetched_pdb

            # Protonate with Protoss if specified
            if use_protoss:
                final_pdb = protonate_protein_protoss(fixed_pdb, output_dir)
            else:
                final_pdb = fixed_pdb

            # Rename the prepared receptor
            final_pdb.rename(prepared_receptor_path)

            # Clean up intermediate files
            if fetched_pdb.parent == output_dir and fetched_pdb != original_input:
                fetched_pdb.unlink(missing_ok=True)
            if fixed_pdb.parent == output_dir and fixed_pdb != original_input:
                fixed_pdb.unlink(missing_ok=True)

        except Exception as e:
            # If it's already a ValueError, propagate it directly
            if isinstance(e, ValueError):
                raise
            # Otherwise wrap it in a ValueError with context
            raise ValueError(f"Failed to prepare protein: {str(e)}")

    return prepared_receptor_path

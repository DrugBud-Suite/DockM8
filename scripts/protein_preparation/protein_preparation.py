import sys
from pathlib import Path

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.protein_preparation.fetching.fetch_alphafold import fetch_alphafold_structure
from scripts.protein_preparation.fetching.fetch_pdb import fetch_pdb_structure
from scripts.protein_preparation.fixing.pdb_fixer import fix_pdb_file
from scripts.protein_preparation.protonation.protonate_protoss import protonate_protein_protoss
from scripts.protein_preparation.structure_assessment.edia import get_best_chain_edia
from scripts.utilities.utilities import printlog
import requests


def prepare_protein(protein_file_or_code: str or Path,
					output_dir: Path = None,
					select_best_chain: bool = True,
					fix_protein: bool = True,
					fix_nonstandard_residues: bool = True,
					fix_missing_residues: bool = True,
					add_missing_hydrogens_pH: float = 7.0,
					remove_hetero: bool = True,
					remove_water: bool = True,
					protonate: bool = True,
					) -> Path:
	"""
    Prepare a protein structure by performing various modifications.

    Args:
        protein_file_or_code (str or Path): The protein_file_or_code value. It can be a PDB code, Uniprot code, or file path.
        output_dir (str or Path, optional): The directory where the prepared protein structure will be saved. If not provided, the same directory as the protein_file_or_code file will be used.
        select_best_chain (bool, optional): Whether to select the best chain from the protein_file_or_code structure. Only applicable for PDB protein_file_or_code. Default is True.
        fix_nonstandard_residues (bool, optional): Whether to fix nonstandard residues in the protein structure. Default is True.
        fix_missing_residues (bool, optional): Whether to fix missing residues in the protein structure. Default is True.
        add_missing_hydrogens_pH (float, optional): The pH value for adding missing hydrogens. Default is 7.0.
        remove_hetero (bool, optional): Whether to remove heteroatoms from the protein structure. Default is True.
        remove_water (bool, optional): Whether to remove water molecules from the protein structure. Default is True.
        protonate (bool, optional): Whether to protonate the protein structure. Default is True.

    Returns:
        Path: The path to the prepared protein structure.
    """
	prepared_receptor_path = output_dir / "prepared_receptor.pdb"
	printlog(f"Checking validity of receptor input: {protein_file_or_code}")
	if not (prepared_receptor_path).exists():
		if len(str(protein_file_or_code)) == 4 and protein_file_or_code.isalnum():
			url = f"https://www.rcsb.org/structure/{protein_file_or_code}"
			response = requests.head(url)
			if response.status_code == 200:
				type = "PDB"
			else:
				raise ValueError(f"The provided PDB code {protein_file_or_code} is invalid.")
		elif len(str(protein_file_or_code)) == 6 and protein_file_or_code.isalnum():
			url = f"https://www.uniprot.org/uniprotkb/{protein_file_or_code}/entry"
			response = requests.head(url)
			if response.status_code == 200:
				type = "Uniprot"
			else:
				raise ValueError(f"The provided Uniprot code {protein_file_or_code} is invalid.")
		else:
			# Check if the protein_file_or_code is a valid path
			if not Path(protein_file_or_code).is_file():
				raise ValueError(f"{protein_file_or_code} is an invalid file path.")
			else:
				type = "File"

		output_dir.mkdir(parents=True, exist_ok=True)

		# Check if the protein_file_or_code type is valid
		if select_best_chain and type.upper() != "PDB":
			printlog(
				"Selecting the best chain is only supported for PDB protein_file_or_code. Turning off the best chain selection ..."
			)
			select_best_chain = False
		# Check if protonation is required
		if add_missing_hydrogens_pH is None and not protonate:
			printlog(
				"Protonating with Protoss or PDBFixer is required for reliable results. Setting protonate to True.")
			protonate = True

		# Fetch the protein structure
		if type.upper() == "PDB":
			# Ensure the pdb code is in the right format (4 letters or digits)
			pdb_code = protein_file_or_code.strip().upper()
			if len(pdb_code) != 4 or not pdb_code.isalnum():
				raise ValueError("Invalid pdb code format. It should be 4 letters or digits.")
			if select_best_chain:
				# Get the best chain using EDIA
				step1_pdb = get_best_chain_edia(pdb_code, output_dir)
			else:
				# Get PDB structure
				step1_pdb = fetch_pdb_structure(protein_file_or_code, output_dir)
		elif type.upper() == "UNIPROT":
			# Fetch the Uniprot structure
			uniprot_code = protein_file_or_code
			step1_pdb = fetch_alphafold_structure(uniprot_code, output_dir)
		else:
			# Assume protein_file_or_code is a file path
			step1_pdb = Path(protein_file_or_code)

		# Fix the protein structure
		if (fix_nonstandard_residues or fix_missing_residues or add_missing_hydrogens_pH is not None or remove_hetero or
			remove_water):
			# Fix the PDB file
			step2_pdb = fix_pdb_file(step1_pdb,
										output_dir,
										fix_nonstandard_residues,
										fix_missing_residues,
										add_missing_hydrogens_pH,
										remove_hetero,
										remove_water)

		else:
			step2_pdb = step1_pdb
		# Protonate the protein
		if protonate:
			step3_pdb = protonate_protein_protoss(step2_pdb, output_dir)
		else:
			step3_pdb = step2_pdb

		if step1_pdb != step3_pdb and step1_pdb != protein_file_or_code:
			step1_pdb.unlink()
		if step2_pdb != step3_pdb and step2_pdb != protein_file_or_code:
			step2_pdb.unlink()

		step3_pdb.rename(prepared_receptor_path)
	return prepared_receptor_path

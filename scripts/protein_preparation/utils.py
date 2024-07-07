import sys
import warnings
from pathlib import Path

from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning

warnings.filterwarnings("ignore", category=PDBConstructionWarning)

scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.logging import printlog


def extract_chain(pdb_file: Path, chain_id: str):
	try:
		parser = PDBParser()
		structure = parser.get_structure("structure", pdb_file)

		# Create a PDBIO object for writing the PDB file
		pdbio = PDBIO()

		# Select the specified chain
		for model in structure:
			for chain in model:
				if chain.get_id() == chain_id:
					# Set the structure for the output to the selected chain
					pdbio.set_structure(chain)
					output_file = pdb_file.parent / f"{pdb_file.stem}_{chain_id}.pdb"
					pdbio.save(str(output_file))
					return output_file
	except FileNotFoundError:
		printlog(f"Error: PDB file '{pdb_file}' not found.")
	except Exception as e:
		printlog(f"Error in extracting chains from {pdb_file}: {e}")

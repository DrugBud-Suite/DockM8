import warnings
from pathlib import Path

from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning

warnings.filterwarnings("ignore", category=PDBConstructionWarning)


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
		print(f"Error: PDB file '{pdb_file}' not found.")
	except Exception as e:
		print(f"Error in extracting chains from {pdb_file}: {e}")

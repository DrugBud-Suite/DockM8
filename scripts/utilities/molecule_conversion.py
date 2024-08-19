import os
import subprocess
import sys
import warnings
from pathlib import Path
from rdkit import Chem
from meeko import MoleculePreparation, PDBQTWriterLegacy
import traceback
from openbabel import pybel

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.logging import printlog
from scripts.setup.software_manager import ensure_software_installed

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

pybel.ob.obErrorLog.StopLogging()



def convert_molecules(input_file: Path, output_file_or_path: Path, input_format: str, output_format: str):
	"""
	Converts molecules from one format to another using various conversion tools.

	Args:
		input_file (Path): The path to the input file containing the molecules.
		output_file_or_path (Path): The path to the output file or directory where the converted molecules will be saved.
		input_format (str): The format of the input molecules.
		output_format (str): The desired format of the output molecules.

	Returns:
		The path(s) to the converted molecule file(s).

	Raises:
		FileNotFoundError: If the input file for molecule conversion is not found.
		subprocess.CalledProcessError: If an error occurs during conversion using MGLTools.
		Exception: If an error occurs during conversion using Meeko or Pybel.
	"""
	if not Path(input_file).exists():
		raise FileNotFoundError(f"Input file for molecule conversion not found: {input_file}")

	# For protein conversion to pdbqt file format using MGLTools
	if input_format == "pdb" and output_format == "pdbqt":
		ensure_software_installed("MGLTOOLS", Path("/"))
		try:
			cmd = f"conda run -n mgltools prepare_receptor4.py -r {input_file} -o {output_file_or_path} -A bond_hydrogens"
			subprocess.run(cmd, shell=True)
			return output_file_or_path
		except Exception as e:
			printlog(f"Error occurred during conversion using MGLTools prepare_receptor4.py: {str(e)}")
			printlog(traceback.format_exc())
			raise

	# For compound conversion to pdbqt file format using MGLTools
	if input_format == "sdf" and output_format == "pdbqt":
		try:
			pdbqt_files = []
			mols = list(Chem.SDMolSupplier(str(input_file), removeHs=False))

			if len(mols) == 1 and output_file_or_path.suffix == '.pdbqt':
				# Single compound, output is a file
				mol = mols[0]
				preparator = MoleculePreparation(min_ring_size=10)
				mol = Chem.AddHs(mol)
				setup_list = preparator.prepare(mol)
				pdbqt_string = PDBQTWriterLegacy.write_string(setup_list[0])
				with open(output_file_or_path, "w") as f:
					f.write(pdbqt_string[0])
				pdbqt_files.append(output_file_or_path)
			else:
				# Multiple compounds or output is a directory
				output_dir = output_file_or_path if output_file_or_path.is_dir() else output_file_or_path.parent
				output_dir.mkdir(parents=True, exist_ok=True)

				for i, mol in enumerate(mols):
					if mol is None:
						continue
					preparator = MoleculePreparation(min_ring_size=10)
					mol = Chem.AddHs(mol)
					setup_list = preparator.prepare(mol)
					pdbqt_string = PDBQTWriterLegacy.write_string(setup_list[0])
					mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"compound_{i+1}"
					output_path = Path(output_dir) / f"{mol_name}.pdbqt"
					with open(output_path, "w") as f:
						f.write(pdbqt_string[0])
					pdbqt_files.append(output_path)

			return pdbqt_files
		except Exception as e:
			printlog(f"Error occurred during conversion using Meeko: {str(e)}")
			raise

	# For general conversion using Pybel
	else:
		pybel.ob.obErrorLog.StopLogging()
		try:
			output = pybel.Outputfile(output_format, str(output_file_or_path), overwrite=True)
			for mol in pybel.readfile(input_format, str(input_file)):
				output.write(mol)
			output.close()
			return output_file_or_path
		except Exception as e:
			printlog(f"Error occurred during conversion using Pybel: {str(e)}")
			raise

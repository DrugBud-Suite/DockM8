import os
import subprocess
import sys
import warnings
from pathlib import Path
from rdkit import Chem
from meeko import MoleculePreparation, PDBQTWriterLegacy

from openbabel import pybel

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.logging import printlog

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

pybel.ob.obErrorLog.StopLogging()


def find_mgltools_installation(software_path):
	"""
	Find the MGLTools installation directory.

	Parameters:
	- software_path (str): The path to the software installation directory.

	Returns:
	- str: The path to the MGLTools installation directory.

	Raises:
	- FileNotFoundError: If the MGL_Tools directory or MGLToolsPckgs directory is not found.
	"""
	mgl_tools_dir = software_path / 'MGL_Tools'
	if not mgl_tools_dir.exists():
		raise FileNotFoundError(f"MGL_Tools directory not found at {mgl_tools_dir}")

	# Look for MGLToolsPckgs directory
	mgltools_pckgs = list(mgl_tools_dir.glob('**/MGLToolsPckgs'))
	if not mgltools_pckgs:
		raise FileNotFoundError(f"MGLToolsPckgs directory not found in {mgl_tools_dir}")

	return mgltools_pckgs[0].parent


def get_mgltools_env(software_path):
	"""
	Get environment variables for MGLTools.

	Args:
		software_path (str): The path to the MGLTools software installation.

	Returns:
		dict: A dictionary containing the environment variables for MGLTools.

	Raises:
		FileNotFoundError: If the required directories or files are not found.
		RuntimeError: If the pythonsh executable is not runnable.

	"""
	mgltools_path = find_mgltools_installation(software_path)

	bin_dir = mgltools_path / 'bin'
	if not bin_dir.exists():
		raise FileNotFoundError(f"bin directory not found at {bin_dir}")

	pythonsh = bin_dir / 'pythonsh'
	if not pythonsh.exists():
		raise FileNotFoundError(f"pythonsh not found at {pythonsh}")

	utilities_dir = mgltools_path / 'MGLToolsPckgs' / 'AutoDockTools' / 'Utilities24'
	if not utilities_dir.exists():
		raise FileNotFoundError(f"Utilities24 directory not found at {utilities_dir}")

	env = os.environ.copy()
	env['PATH'] = f"{bin_dir}:{env['PATH']}"
	env['MGL'] = str(mgltools_path)
	env['MGLPY'] = str(pythonsh)
	env['MGLUTIL'] = str(utilities_dir)

	# Verify that pythonsh is runnable
	try:
		subprocess.run([pythonsh, '--version'], check=True, capture_output=True)
	except subprocess.CalledProcessError:
		raise RuntimeError(f"pythonsh at {pythonsh} is not runnable")

	return env


def convert_molecules(input_file: Path,
						output_file_or_path: Path,
						input_format: str,
						output_format: str,
						software: Path):
	"""
	Converts molecules from one format to another using various conversion tools.

	Args:
		input_file (Path): The path to the input file containing the molecules.
		output_file_or_path (Path): The path to the output file or directory where the converted molecules will be saved.
		input_format (str): The format of the input molecules.
		output_format (str): The desired format of the output molecules.
		software (Path): The path to the software used for conversion.

	Returns:
		The path(s) to the converted molecule file(s).

	Raises:
		FileNotFoundError: If the input file for molecule conversion is not found.
		subprocess.CalledProcessError: If an error occurs during conversion using MGLTools.
		Exception: If an error occurs during conversion using Meeko or Pybel.
	"""
	try:
		mgl_env = get_mgltools_env(software)
	except (FileNotFoundError, RuntimeError) as e:
		printlog(f"Error setting up MGLTools environment: {str(e)}")
		raise

	if not input_file.exists():
		raise FileNotFoundError(f"Input file for molecule conversion not found: {input_file}")

	# For protein conversion to pdbqt file format using MGLTools
	if input_format == "pdb" and output_format == "pdbqt":
		try:
			prepare_receptor_script = Path(mgl_env['MGLUTIL']) / "prepare_receptor4.py"
			cmd = f"{mgl_env['MGLPY']} {prepare_receptor_script} -r {input_file} -o {output_file_or_path} -A checkhydrogens"
			subprocess.run(cmd, shell=True, check=True, env=mgl_env)
			return output_file_or_path
		except subprocess.CalledProcessError as e:
			printlog(f"Error occurred during conversion using MGLTools prepare_receptor4.py: {str(e)}")
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
					output_path = output_dir / f"{mol_name}.pdbqt"
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

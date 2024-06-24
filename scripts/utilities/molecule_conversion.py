import os
from pathlib import Path
import sys
import warnings
import subprocess
from meeko import MoleculePreparation
from meeko import PDBQTWriterLegacy
import openbabel
from openbabel import pybel
from rdkit import Chem

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.utilities import printlog

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def convert_molecules(input_file: Path, output_file: Path, input_format: str, output_format: str):
	"""
    Convert molecules from one file format to another.

    Args:
        input_file (Path): The path to the input file.
        output_file (Path): The path to the output file.
        input_format (str): The format of the input file.
        output_format (str): The format of the output file.

	Returns:
        Path: The path to the converted output file.
    """
	# For protein conversion to pdbqt file format using OpenBabel
	if input_format == "pdb" and output_format == "pdbqt":
		try:
			mol = Chem.MolFromPDBFile(str(input_file), sanitize=False, removeHs=False)
			lines = [x.strip() for x in open(input_file).readlines()]
			out_lines = []
			for line in lines:
				if "ROOT" in line or "ENDROOT" in line or "TORSDOF" in line:
					out_lines.append("%s\n" % line)
					continue
				if not line.startswith("ATOM"):
					continue
				line = line[:66]
				atom_index = int(line[6:11])
				atom = mol.GetAtoms()[atom_index - 1]
				line = "%s    +0.000 %s\n" % (line, atom.GetSymbol().ljust(2))
				out_lines.append(line)
			with open(output_file, 'w') as fout:
				for line in out_lines:
					fout.write(line)
			return output_file
		except Exception as e:
			printlog(f"Error occurred during conversion using RDkit: {str(e)}. Trying with OpenBabel...")
			try:
				# Run the pdb2pqr command
				subprocess.call(
					f"pdb2pqr --ff=AMBER --ffout=AMBER --keep-chain --nodebump {input_file} {input_file.with_suffix('.pqr')} -q",
					shell=True,
					stdout=subprocess.DEVNULL,
					stderr=subprocess.DEVNULL)
				# Run the mk_prepare_receptor.py script
				subprocess.call(
					f"{dockm8_path}/scripts/utilities/mk_prepare_receptor.py --pdb {input_file.with_suffix('.pqr')} -o {output_file} --skip_gpf",
					shell=True,
					stdout=subprocess.DEVNULL,
					stderr=subprocess.DEVNULL)
				os.remove(input_file.with_suffix(".pqr")) if os.path.exists(input_file.with_suffix(".pqr")) else None
				return output_file
			except Exception as e:
				printlog(
					f"Error occurred during conversion using PDB2PQR and Meeko: {str(e)}. Trying with OpenBabel...")
				try:
					obConversion = openbabel.OBConversion()
					mol = openbabel.OBMol()
					obConversion.ReadFile(mol, str(input_file))
					obConversion.SetInAndOutFormats("pdb", "pdbqt")
					# Calculate Gasteiger charges
					charge_model = openbabel.OBChargeModel.FindType("gasteiger")
					charge_model.ComputeCharges(mol)
					obConversion.WriteFile(mol, str(output_file))
					# Remove all torsions from pdbqt output
					with open(output_file, "r") as file:
						lines = file.readlines()
						lines = [
							line for line in lines if all(keyword not in line for keyword in [
								"between atoms:", "BRANCH", "ENDBRANCH", "torsions", "Active", "ENDROOT", "ROOT", ])]
						lines = [line.replace(line, "TER\n") if line.startswith("TORSDOF") else line for line in lines]
						with open(output_file, "w") as file:
							file.writelines(lines)
				except Exception as e:
					printlog(f"Error occurred during conversion using OpenBabel: {str(e)}")
				return output_file
	# For compound conversion to pdbqt file format using RDKit and Meeko
	if input_format == "sdf" and output_format == "pdbqt":
		try:
			for mol in Chem.SDMolSupplier(str(input_file), removeHs=False):
				preparator = MoleculePreparation(min_ring_size=10)
				mol = Chem.AddHs(mol)
				setup_list = preparator.prepare(mol)
				pdbqt_string = PDBQTWriterLegacy.write_string(setup_list[0])
				mol_name = mol.GetProp("_Name")
				output_path = Path(output_file) / f"{mol_name}.pdbqt"
				# Write the pdbqt string to the file
				with open(output_path, "w") as f:
					f.write(pdbqt_string[0])
		except Exception as e:
			printlog(f"Error occurred during conversion using Meeko: {str(e)}")
		return output_file
	# For general conversion using Pybel
	else:
		try:
			output = pybel.Outputfile(output_format, str(output_file), overwrite=True)
			for mol in pybel.readfile(input_format, str(input_file)):
				output.write(mol)
			output.close()
		except Exception as e:
			printlog(f"Error occurred during conversion using Pybel: {str(e)}")
		return output_file

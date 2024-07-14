import sys
from pathlib import Path
from typing import Dict, List, Union
import traceback
import math
import MDAnalysis as mda
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, PandasTools

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.logging import printlog
from scripts.utilities.parallel_executor import parallel_executor


def extract_pocket_mol(pdb_filepath: str,
						ligand: Chem.Mol,
						distance_from_ligand: float,
						ligand_resname: str = 'UNL') -> Union[Chem.Mol, None]:
	try:
		universe = mda.Universe(pdb_filepath)
		ligand_universe = mda.Universe(ligand)
		ligand_universe.add_TopologyAttr('resname', [ligand_resname])

		complx = mda.Merge(universe.atoms, ligand_universe.atoms)

		selections = ['protein', f'around {distance_from_ligand} resname {ligand_resname}', 'not type H']
		selection = '(' + ') and ('.join(selections) + ')'
		atom_group: mda.AtomGroup = complx.select_atoms(selection)

		if len(atom_group) > 20:
			segids = {}
			for residue in atom_group.residues:
				segid = residue.segid
				resid = residue.resid
				if segid in segids:
					segids[segid].append(resid)
				else:
					segids[segid] = [resid]
			selections = []
			for segid, resids in segids.items():
				resids_str = ' '.join([str(resid) for resid in set(resids)])
				selections.append(f'((resid {resids_str}) and (segid {segid}))')
			pocket_selection = ' or '.join(selections)
			protein_pocket: mda.AtomGroup = universe.select_atoms(pocket_selection)
			return protein_pocket.atoms.convert_to("RDKIT")
		else:
			printlog('Warning: Pocket quite small')
			return None
	except Exception as e:
		printlog(f"Error in extract_pocket_mol: {str(e)}")
		printlog(f"Traceback: {traceback.format_exc()}")
		return None


def correct_valence(mol):
	"""
	Attempt to correct valence errors in an RDKit molecule.
	"""
	try:
		Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
	except ValueError:
		pass

	for atom in mol.GetAtoms():
		if atom.GetSymbol() == 'N' and atom.GetFormalCharge() == 0 and len(atom.GetBonds()) == 4:
			atom.SetFormalCharge(1)
		elif atom.GetSymbol() == 'S' and atom.GetFormalCharge() == 0 and len(atom.GetBonds()) > 4:
			atom.SetIsAromatic(False)

	mol = Chem.AddHs(mol)
	AllChem.EmbedMolecule(mol, randomSeed=42)
	return mol


def extract_pocket_from_coords(pdb_filepath: str, pocket_coords: Dict[str, List[float]]) -> Union[Chem.Mol, None]:
	try:
		universe = mda.Universe(pdb_filepath.replace('.pdb', '_pocket.pdb'))
		center = pocket_coords['center']
		size = pocket_coords['size']
		atom_group = universe.select_atoms('protein')
		# # Initial selection based on coordinates
		# selection = f"protein and point {center[0]} {center[1]} {center[2]} {max(size)/2}"
		# initial_selection: mda.AtomGroup = universe.select_atoms(selection)

		# if len(initial_selection) > 20:
		# 	# Get unique residues from the initial selection
		# 	selected_residues = initial_selection.residues

		# 	# Create a new selection including all atoms of the selected residues
		# 	segids = {}
		# 	for residue in selected_residues:
		# 		segid = residue.segid
		# 		resid = residue.resid
		# 		if segid in segids:
		# 			segids[segid].append(resid)
		# 		else:
		# 			segids[segid] = [resid]

		# 	# Construct the selection string for whole residues
		# 	selections = []
		# 	for segid, resids in segids.items():
		# 		resids_str = ' '.join([str(resid) for resid in set(resids)])
		# 		selections.append(f'((resid {resids_str}) and (segid {segid}))')
		# 	pocket_selection = ' or '.join(selections)

		# 	# Select the atoms based on the constructed selection string
		# 	pocket_atoms: mda.AtomGroup = universe.select_atoms(pocket_selection)

		# Convert to RDKit molecule
		mol = atom_group.atoms.convert_to("RDKIT")

		# Attempt to correct valence errors
		mol = correct_valence(mol)

		return mol
	# 	else:
	# 		printlog('Warning: Initial pocket selection quite small')
	# 		return None
	except KeyError as e:
		printlog(f"Error in pocket coordinates: {str(e)}")
		printlog(f"Traceback: {traceback.format_exc()}")
		return None
	except Exception as e:
		printlog(f"Error in extract_pocket_from_coords: {str(e)}")
		printlog(f"Traceback: {traceback.format_exc()}")
		return None


def prepare_complex(pose: Chem.Mol, protein_file: str, config: Dict) -> Union[Chem.Mol, None]:
	try:
		pose = Chem.AddHs(pose, addCoords=True)

		if config['docking_type'] == 'blind':
			pocket_mol = extract_pocket_mol(protein_file, pose, config['distance_from_ligand'])
			if pocket_mol is None:
				raise ValueError("Failed to extract pocket for blind docking")
		elif config['docking_type'] == 'regular':
			if config.get('pocket') is None:
				raise ValueError("Pocket coordinates must be provided for regular docking")
			pocket_mol = extract_pocket_from_coords(protein_file, config['pocket'])
			if pocket_mol is None:
				raise ValueError("Failed to extract pocket for regular docking")
		else:
			raise ValueError(f"Invalid docking type: {config['docking_type']}")

		complex_mol = Chem.CombineMols(pocket_mol, pose)
		complex_mol = Chem.AddHs(complex_mol, addCoords=True)
		#complex_mol = Chem.SanitizeMol(complex_mol)
		if complex_mol is None:
			raise ValueError("Failed to combine pocket and pose molecules")

		return complex_mol
	except Exception as e:
		printlog(f"Error in prepare_complex: {str(e)}")
		printlog(f"Traceback: {traceback.format_exc()}")
		return None


def minimize_pose(pose: Chem.Mol, protein_file: str, config: Dict):
	try:
		complex_mol = prepare_complex(pose, protein_file, config)
		if complex_mol is None:
			raise ValueError("Failed to prepare complex for minimization")

		if complex_mol is None:
			raise ValueError("Failed to sanitize the complex molecule")

		if config['forcefield'] in ['MMFF94', 'MMFF94s']:
			mol_properties = AllChem.MMFFGetMoleculeProperties(complex_mol, mmffVariant=config['forcefield'])
			if mol_properties is None:
				raise ValueError(f"Failed to get molecule properties for {config['forcefield']}")
			ff = AllChem.MMFFGetMoleculeForceField(complex_mol,
													mol_properties,
													confId=0,
													nonBondedThresh=10.0,
													ignoreInterfragInteractions=False)
		elif config['forcefield'] == 'UFF':
			ff = AllChem.UFFGetMoleculeForceField(complex_mol, confId=0)
		else:
			raise ValueError(f"Unsupported forcefield: {config['forcefield']}")

		if ff is None:
			raise ValueError(f"Failed to initialize force field: {config['forcefield']}")

		ff.Initialize()

		for idx in range(complex_mol.GetNumAtoms()):
			atom = complex_mol.GetAtomWithIdx(idx)
			if atom.GetSymbol() != 'H':
				if config['forcefield'] in ['MMFF94', 'MMFF94s']:
					ff.MMFFAddPositionConstraint(idx, config['distance_constraint'], 999.0)
				elif config['forcefield'] == 'UFF':
					ff.UFFAddPositionConstraint(idx, config['distance_constraint'], 999.0)

		E_init = ff.CalcEnergy()
		results = ff.Minimize(maxIts=config['n_steps'])
		print(results)
		E_final = ff.CalcEnergy()

		minimized_frags = Chem.GetMolFrags(complex_mol, asMols=True)
		minimized_pose = minimized_frags[-1]

		return minimized_pose, E_init, E_final

	except Exception as e:
		printlog(f"Error in minimize_pose: {str(e)}")
		printlog(f"Traceback: {traceback.format_exc()}")
		return pose, None, None


def minimize_batch(batch: pd.DataFrame, protein_file: str, config: Dict) -> pd.DataFrame:
	results = []

	for _, row in batch.iterrows():
		try:
			pose = row['Molecule']
			minimized_pose, energy, min_energy = minimize_pose(pose, protein_file, config)

			new_row = row.copy()      # Create a copy of the original row to preserve all columns
			new_row['Molecule'] = minimized_pose
			if energy is not None and min_energy is not None:
				new_row['pose_energy'] = energy
				new_row['minimized_pose_energy'] = min_energy
			results.append(new_row)
		except Exception as e:
			printlog(f"Error processing row: {str(e)}")
			printlog(f"Traceback: {traceback.format_exc()}")
			results.append(row)       # Append original row if processing fails

	return pd.DataFrame(results)


def minimize_poses(input_data: Union[str, pd.DataFrame],
					protein_file: str,
					output_file: str,
					config: Dict,
					n_cpus: int = 1):
	"""
	Main function to minimize poses in batches using parallel processing.
	
	Args:
		input_data (Union[str, pd.DataFrame]): Input SDF file path or DataFrame containing poses
		protein_file (str): Path to the protein PDB file
		output_file (str): Path to save the output SDF file
		config (Dict): Configuration dictionary for minimization
		batch_size (int): Number of poses to process in each batch
		n_cpus (int): Number of CPUs to use for parallel processing
	"""
	try:
		if isinstance(input_data, str):
			df = PandasTools.LoadSDF(input_data, idName='Pose ID', molColName='Molecule')
		elif isinstance(input_data, pd.DataFrame):
			df = input_data
		else:
			raise ValueError("Input must be either a path to an SDF file or a pandas DataFrame")

		if 'Molecule' not in df.columns:
			raise ValueError("Input DataFrame must contain a 'Molecule' column")

		total_poses = len(df)

		if total_poses > n_cpus:
			compounds_per_batch = math.ceil(total_poses / n_cpus)
		else:
			compounds_per_batch = total_poses

		batches = [df[i:i + compounds_per_batch] for i in range(0, len(df), compounds_per_batch)]

		minimized_batches = parallel_executor(minimize_batch,
												batches,
												n_cpus=n_cpus,
												job_manager="concurrent_process",
												display_name="Pose Minimization",
												protein_file=protein_file,
												config=config)

		minimized_df = pd.concat(minimized_batches, ignore_index=True)

		PandasTools.WriteSDF(minimized_df,
								output_file,
								molColName='Molecule',
								idName='Pose ID',
								properties=list(minimized_df.columns))

		return minimized_df

	except Exception as e:
		printlog(f"Error in minimize_poses: {str(e)}")
		printlog(f"Traceback: {traceback.format_exc()}")
		return None

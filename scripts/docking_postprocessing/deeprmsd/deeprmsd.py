import os
import sys
import time
import shutil
from pathlib import Path
import subprocess
import pandas as pd
from typing import Union, List
from rdkit.Chem import PandasTools
from rdkit import Chem

from scripts.utilities.logging import printlog
from scripts.utilities.parallel_executor import parallel_executor


def prepare_receptor(protein_file: Path, output_file: Path) -> None:
	"""
    Prepare the receptor using prepare_receptor4.py.

    Args:
        protein_file (Path): Path to the input protein PDB file.
        output_file (Path): Path to the output PDBQT file.
        mgltools_path (Path): Path to the MGLTools directory.
    """
	cmd = f"conda run -n mgltools prepare_receptor4.py -r {protein_file} -o {output_file} -A bond_hydrogens -U lps"
	subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def prepare_input_file(protein_file: Path, poses_dir: Path, input_file: Path) -> None:
	"""
    Prepare the input file for DeepRMSD+Vina optimization.

    Args:
        protein_file (Path): Path to the prepared protein PDBQT file.
        poses_dir (Path): Directory containing ligand pose PDBQT files.
        input_file (Path): Path to the input file to be created.
    """
	with open(input_file, 'w') as f:
		for pose_file in poses_dir.glob('*.pdbqt'):
			f.write(f"{pose_file.stem} {protein_file} {poses_dir}\n")


def run_optimization(input_file: Path, output_dir: Path, software_path: Path) -> None:
	"""
    Run the DeepRMSD+Vina optimization using the bash script.

    Args:
        input_file (Path): Path to the prepared input file.
        output_dir (Path): Directory to store optimization results.
        software_path (Path): Path to the DeepRMSD+Vina software directory.
    """
	cmd = f"bash ./run_pose_optimization.sh {str(input_file)}"
	try:
		subprocess.run(cmd, shell=True, cwd=software_path / "DeepRMSD-Vina_Optimization")
	except subprocess.CalledProcessError as e:
		printlog(f"Error running optimization: {e}")
		raise


def convert_sdf_to_mol2(compound_data: tuple) -> List[Path]:
	sdf_file, output_dir, compound_id, pose_ids = compound_data
	output_dir.mkdir(parents=True, exist_ok=True)
	mol2_files = []
	for pose_id in pose_ids:
		cmd = f"obabel {sdf_file} -O {output_dir}/{pose_id}.mol2 -f {pose_ids.index(pose_id)+1} -l {pose_ids.index(pose_id)+1}"
		subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		mol2_files.append(output_dir / f"{pose_id}.mol2")
	return mol2_files


def prepare_ligand(ligand_data: tuple) -> None:
	mol2_file, pdbqt_file = ligand_data
	cmd = f"conda run -n mgltools prepare_ligand4.py -l {mol2_file.name} -o {pdbqt_file} -A bond_hydrogens -U lps"
	subprocess.run(cmd,
					shell=True,
					cwd=mol2_file.parent,
					stdout=subprocess.DEVNULL,
					stderr=subprocess.DEVNULL)


def process_compound(compound_data: tuple) -> pd.DataFrame:
	compound_id, compound_poses, compound_sdf, mol2_dir, poses_pdbqt_dir, protein_pdbqt, input_file, output_dir, software_path = compound_data

	# Convert SDF to MOL2
	pose_ids = compound_poses['Pose ID'].tolist()
	mol2_files = convert_sdf_to_mol2((compound_sdf, mol2_dir, compound_id, pose_ids))

	# Convert molecules to PDBQT
	poses_pdbqt_dir.mkdir(exist_ok=True)
	ligand_data = [(mol2_file, poses_pdbqt_dir / f"{mol2_file.stem}.pdbqt") for mol2_file in mol2_files]
	parallel_executor(prepare_ligand, ligand_data, n_cpus=1, job_manager="concurrent_process")

	# Create input file
	prepare_input_file(protein_pdbqt, poses_pdbqt_dir, input_file)

	# Run optimization
	run_optimization(input_file, output_dir, software_path)

	# Process results
	result_df = compound_poses.copy()
	for pose_file in poses_pdbqt_dir.glob('*.pdbqt'):
		pose_id = pose_file.stem
		optimized_pose_file = output_dir / f"{compound_id}_pdbqts/{pose_id}/final_optimized_cnfr.pdb"
		optimized_data_file = output_dir / f"{compound_id}_pdbqts/{pose_id}/opt_data.csv"

		if optimized_pose_file.exists() and optimized_data_file.exists():
			opt_data = pd.read_csv(optimized_data_file)
			final_score = opt_data['total_score'].iloc[-1]

			result_df.loc[result_df['Pose ID'] == pose_id, 'Optimized Score'] = final_score
			result_df.loc[result_df['Pose ID'] == pose_id, 'Optimized Pose Path'] = str(optimized_pose_file)
		else:
			printlog(f"Warning: Optimization results not found for pose {pose_id}")

	return result_df


def optimize_poses(poses: Union[pd.DataFrame, Path],
					protein_file: Path,
					output_dir: Path,
					software_path: Path,
					ncpus: int = max(1, int(os.cpu_count() * 0.9))) -> pd.DataFrame:
	tic = time.perf_counter()
	printlog("Optimizing poses with DeepRMSD+Vina...")

	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	# Step 1: Read the dataframe or sdf file
	if isinstance(poses, pd.DataFrame):
		sdf_file = output_dir / "poses.sdf"
		PandasTools.WriteSDF(poses,
								str(sdf_file),
								molColName='Molecule',
								idName='Pose ID',
								properties=list(poses.columns))
	else:
		sdf_file = poses
		poses = PandasTools.LoadSDF(str(sdf_file), idName='Pose ID', molColName='Molecule', includeFingerprints=False)
		if 'ID' not in poses.columns:
			poses['ID'] = poses['Pose ID'].str.split("_").str[0]
		if 'Pose ID' not in poses.columns:
			raise ValueError("Pose ID not found in SDF file")

	# Step 2: Prepare protein (only needs to be done once)
	protein_pdbqt = output_dir / "protein_prepared.pdbqt"
	if not protein_pdbqt.exists():
		prepare_receptor(protein_file, protein_pdbqt)

	# Step 3: Process compounds in parallel
	compounds = poses['ID'].unique()
	compound_data = []
	for compound_id in compounds:
		compound_poses = poses[poses['ID'] == compound_id]
		compound_sdf = output_dir / f"{compound_id}_poses.sdf"
		PandasTools.WriteSDF(compound_poses,
								str(compound_sdf),
								molColName='Molecule',
								idName='Pose ID',
								properties=list(compound_poses.columns))

		mol2_dir = output_dir / f"{compound_id}_mol2s"
		poses_pdbqt_dir = output_dir / f"{compound_id}_pdbqts"
		input_file = output_dir / f"{compound_id}_input.dat"

		compound_data.append((compound_id,
								compound_poses,
								compound_sdf,
								mol2_dir,
								poses_pdbqt_dir,
								protein_pdbqt,
								input_file,
								output_dir,
								software_path))

	all_results = parallel_executor(process_compound,
									compound_data,
									n_cpus=ncpus,
									job_manager="concurrent_process",
									display_name="Processing compounds")

	# Combine results from all compounds
	final_result_df = pd.concat(all_results, ignore_index=True)

	toc = time.perf_counter()
	printlog(f"DeepRMSD+Vina optimization completed in {toc - tic:.2f} seconds.")
	return final_result_df

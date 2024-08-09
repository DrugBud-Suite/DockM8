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

# Add DockM8 path to sys.path (assuming similar structure to posebusters.py)
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.logging import printlog
from scripts.utilities.molecule_conversion import convert_molecules


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
			f.write(f"{pose_file.stem} {protein_file} {pose_file}\n")


def run_optimization(input_file: Path, output_dir: Path, software_path: Path) -> None:
	"""
    Run the DeepRMSD+Vina optimization using the bash script.

    Args:
        input_file (Path): Path to the prepared input file.
        output_dir (Path): Directory to store optimization results.
    """
	cmd = f"cd {software_path}/DeepRMSD-Vina_Optimization && bash run_pose_optimization.sh {str(input_file)}"
	try:
		subprocess.run(cmd, cwd=output_dir)
	except subprocess.CalledProcessError as e:
		printlog(f"Error running optimization: {e}")
		raise


def optimize_poses(poses: Union[pd.DataFrame, Path],
	protein_file: Path,
	output_dir: Path,
	software_path: Path,
	ncpus: int = max(1, int(os.cpu_count() * 0.9)),
	) -> pd.DataFrame:
	"""
    Optimize docking poses using DeepRMSD+Vina.

    Args:
        poses (Union[pd.DataFrame, Path]): Input DataFrame containing poses OR path to an .sdf file.
        protein_file (Path): Path to a protein file.
        output_dir (Path): Path to an output directory.
        ncpus (int): Number of CPUs to use.

    Returns:
        pd.DataFrame: DataFrame with optimized pose information.
    """
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
		sdf_file = output_dir / "poses.sdf"
		shutil.copy(poses, sdf_file)
		poses = pd.DataFrame({'Pose ID': range(1, len(list(Chem.SDMolSupplier(str(sdf_file)))) + 1)})

	# Step 2: Convert the molecules to .pdbqt files
	poses_pdbqt_dir = output_dir / "poses_pdbqts"
	poses_pdbqt_dir.mkdir(exist_ok=True)
	convert_molecules(sdf_file, poses_pdbqt_dir, "sdf", "pdbqt")

	# Step 3: Convert the protein to .pdbqt file
	protein_pdbqt = output_dir / "protein_prepared.pdbqt"
	convert_molecules(protein_file, protein_pdbqt, "pdb", "pdbqt")

	# Step 4: Create the input.dat file
	input_file = output_dir / "input.dat"
	prepare_input_file(protein_pdbqt, poses_pdbqt_dir, input_file)

	# Step 5: Run the optimization
	run_optimization(input_file, output_dir, software_path)

	# Process results
	result_df = poses.copy()
	for pose_file in poses_pdbqt_dir.glob('*.pdbqt'):
		pose_id = pose_file.stem
		optimized_pose_file = output_dir / f"poses_pdbqts/{pose_id}/final_optimized_cnfr.pdb"
		optimized_data_file = output_dir / f"poses_pdbqts/{pose_id}/opt_data.csv"

		if optimized_pose_file.exists() and optimized_data_file.exists():
			opt_data = pd.read_csv(optimized_data_file)
			final_score = opt_data['total_score'].iloc[-1]

			result_df.loc[result_df['Pose ID'] == pose_id, 'Optimized Score'] = final_score
			result_df.loc[result_df['Pose ID'] == pose_id, 'Optimized Pose Path'] = str(optimized_pose_file)
		else:
			printlog(f"Warning: Optimization results not found for pose {pose_id}")

	toc = time.perf_counter()
	printlog(f"DeepRMSD+Vina optimization completed in {toc - tic:.2f} seconds.")
	return result_df

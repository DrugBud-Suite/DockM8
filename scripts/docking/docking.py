import sys
import tempfile
import warnings
from pathlib import Path
from typing import Union

import pandas as pd
from rdkit import RDLogger
from rdkit.Chem import PandasTools

RDLogger.DisableLog("rdApp.warning")

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.docking.fabind_docking import FABindDocking
from scripts.docking.gnina_docking import GninaDocking
from scripts.docking.panther_docking import PantherDocking
from scripts.docking.plantain_docking import PlantainDocking
from scripts.docking.plants_docking import PlantsDocking
from scripts.docking.psovina_docking import PsovinaDocking
from scripts.docking.qvina2_docking import Qvina2Docking
from scripts.docking.qvinaw_docking import QvinawDocking
from scripts.docking.smina_docking import SminaDocking
from scripts.utilities.logging import printlog
from scripts.utilities.utilities import parallel_SDF_loader

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

DOCKING_PROGRAMS = {
	"SMINA": SminaDocking,
	"GNINA": GninaDocking,
	"PLANTS": PlantsDocking,
	"QVINAW": QvinawDocking,
	"QVINA2": Qvina2Docking,
	"PSOVINA": PsovinaDocking,
	"FABIND+": FABindDocking,
	"PANTHER": PantherDocking,
	"PLANTAIN": PlantainDocking}


def dockm8_docking(library: Union[pd.DataFrame, Path],
					w_dir: Path,
					protein_file: Path,
					pocket_definition: dict,
					software: Path,
					docking_programs: list,
					exhaustiveness: int,
					n_poses: int,
					n_cpus: int,
					job_manager="concurrent_process"):
	"""
	Dock ligands into a protein binding site using one or more docking programs and concatenate all poses.

	Args:
		library (pd.DataFrame or Path): The prepared library as a DataFrame or path to an SDF file.
		w_dir (Path): The working directory where the docking results will be saved.
		protein_file (Path): The path to the protein file.
		pocket_definition (dict): A dictionary defining the pocket for docking.
		software (Path): The path to the docking software.
		docking_programs (list): A list of docking programs to use.
		exhaustiveness (int): The exhaustiveness parameter for docking.
		n_poses (int): The number of poses to generate.
		n_cpus (int): The number of CPUs to use for parallel docking.
		job_manager (str, optional): The job manager to use for parallel docking. Defaults to "concurrent_process".

	Returns:
		all_poses_path (Path): The path to the concatenated poses SDF file.

	Raises:
		Exception: If any error occurs during the docking process.

	"""
	try:
		# Create a temporary file if the input is a DataFrame
		if isinstance(library, pd.DataFrame):
			with tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as temp_file:
				PandasTools.WriteSDF(library, temp_file.name, molColName="Molecule", idName="ID")
				library_path = Path(temp_file.name)
		else:
			library_path = library

		output_paths = []

		for program in docking_programs:
			printlog(f"Running {program} docking...")
			docking_function = DOCKING_PROGRAMS[program](software)
			(w_dir / program.lower()).mkdir(exist_ok=True, parents=True)
			output_sdf = w_dir / program.lower() / f"{program.lower()}_poses.sdf"
			docking_results = docking_function.dock(library_path,
													protein_file,
													pocket_definition,
													exhaustiveness,
													n_poses,
													n_cpus,
													job_manager,
													output_sdf=output_sdf)

			if output_sdf.exists() and output_sdf.stat().st_size > 0:
				printlog(f"{program} docking completed. Results saved to {output_sdf}")
				output_paths.append(output_sdf)
			else:
				printlog(f"ERROR: No results obtained from {program} docking")

		# Concatenate all poses
		if output_paths:
			all_poses_path = w_dir / "all_poses.sdf"
			all_poses = pd.DataFrame()
			for path in output_paths:
				df = parallel_SDF_loader(str(path), molColName="Molecule", idName="Pose ID", n_cpus=n_cpus)
				all_poses = pd.concat([all_poses, df], ignore_index=True)
			try:
				PandasTools.WriteSDF(all_poses,
										str(all_poses_path),
										molColName="Molecule",
										idName="Pose ID",
										properties=list(all_poses.columns))
				printlog(f"All poses successfully combined and saved to {all_poses_path}")
				return all_poses_path
			except Exception as e:
				printlog(f"ERROR: Failed to write all_poses SDF file: {str(e)}")
		else:
			printlog("ERROR: No poses were generated from any docking program")
	except Exception as e:
		printlog(f"ERROR: Docking failed: {str(e)}")
	finally:
		# Clean up temporary file if it was created
		if isinstance(library, pd.DataFrame):
			library_path.unlink()

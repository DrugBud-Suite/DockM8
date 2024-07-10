import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

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
from scripts.docking.plants_docking import PlantsDocking
from scripts.docking.psovina_docking import PsovinaDocking
from scripts.docking.qvina2_docking import Qvina2Docking
from scripts.docking.qvinaw_docking import QvinawDocking
from scripts.docking.smina_docking import SminaDocking
from scripts.docking.plantain_docking import PlantainDocking
from scripts.utilities.logging import printlog
from scripts.utilities.parallel_executor import parallel_executor
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


def dockm8_docking(library: pd.DataFrame or Path,
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
    Dock ligands into a protein binding site using one or more docking programs.

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
    """
	try:
		# Create a temporary file if the input is a DataFrame
		if isinstance(library, pd.DataFrame):
			with tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as temp_file:
				PandasTools.WriteSDF(library, temp_file.name, molColName="Molecule", idName="ID")
				library_path = Path(temp_file.name)
		else:
			library_path = library

		for program in docking_programs:
			printlog(f"Running {program} docking...")
			docking_function = DOCKING_PROGRAMS[program](software)

			output_sdf = w_dir / program.lower() / f"{program.lower()}_poses.sdf"
			docking_results = docking_function.dock(library_path,
													protein_file,
													pocket_definition,
													exhaustiveness,
													n_poses,
													n_cpus,
													job_manager,
													output_sdf=output_sdf)

			if isinstance(docking_results, pd.DataFrame) and not docking_results.empty:
				printlog(f"{program} docking completed. Results saved to {output_sdf}")
			else:
				printlog(f"ERROR: No results obtained from {program} docking")

	except Exception as e:
		printlog(f"ERROR: Docking failed: {str(e)}")
	finally:
		# Clean up temporary file if it was created
		if isinstance(library, pd.DataFrame):
			library_path.unlink()


def concat_all_poses(w_dir: Path, docking_programs: list, protein_file: Path, n_cpus: int):
	"""
    Concatenates all poses from the specified docking programs.

    Args:
    w_dir (Path): Working directory where the docking program output files are located.
    docking_programs (list): List of strings specifying the names of the docking programs used.
    protein_file (Path): Path to the protein file used for docking.
    n_cpus (int): Number of CPUs to use for parallel processing.

    Returns:
    None
    """
	# Create an empty DataFrame to store all poses
	all_poses = pd.DataFrame()
	for program in docking_programs:
		df = parallel_SDF_loader(f"{w_dir}/{program.lower()}/{program.lower()}_poses.sdf",
									molColName="Molecule",
									idName="Pose ID",
									n_cpus=n_cpus)

		all_poses = pd.concat([all_poses, df])
	try:
		# Write the combined poses to an SDF file
		PandasTools.WriteSDF(all_poses,
								f"{w_dir}/allposes.sdf",
								molColName="Molecule",
								idName="Pose ID",
								properties=list(all_poses.columns))

		printlog("All poses successfully checked and combined!")
	except Exception as e:
		printlog("ERROR: Failed to write all_poses SDF file!")
		printlog(e)
	return

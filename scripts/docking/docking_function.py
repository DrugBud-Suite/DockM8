"""Base class for docking functions with essential error recovery."""

import json
import os
import shutil
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.file_splitting import split_sdf
from scripts.utilities.logging import printlog
from scripts.utilities.parallel_executor import parallel_executor


class DockingFunction(ABC):
    """Abstract base class for docking functions with error handling and recovery.

    This class provides a framework for running docking software, managing temporary
    files, handling errors, and supporting the resumption of failed runs.
    """

    def __init__(self, name: str, software_path: Path):
        """Initializes the DockingFunction.

        Args:
            name (str): The name of the docking software.
            software_path (Path): The base path to the docking software installation.
        """
        self.name = name
        self.software_path = software_path
        self._temp_dir = None
        self._run_id = f"{int(time.time())}_{os.getpid()}"
        self._setup_directories()

    def _setup_directories(self):
        """Sets up the directory structure for the current docking run.

        Creates a main temporary directory and subdirectories for raw output,
        processed results, and input splits. Also saves initial run information.
        """
        base_temp = Path.home() / "dockm8_temp_files"
        os.makedirs(base_temp, exist_ok=True)

        self._temp_dir = base_temp / f"dockm8_{self.name.lower()}_{self._run_id}"
        for subdir in ["raw", "processed", "splits"]:
            (self._temp_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Save initial run info
        self._save_run_info("created")  # Initial status

    def _save_run_info(self, status: str, error_msg: str | None = None):
        """Saves or updates run information atomically.

        Writes the run information to a temporary file and then atomically
        replaces the existing run info file to prevent data corruption.

        Args:
            status (str): The current status of the docking run (e.g., "running", "failed", "completed").
            error_msg (str, optional): An error message to save, if applicable. Defaults to None.
        """
        run_info = {
            "run_id": self._run_id,
            "program": self.name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
            "software_path": str(self.software_path),
        }
        if error_msg:
            run_info["error"] = error_msg

        # Write to a temporary file then rename for atomicity
        temp_filepath = self._temp_dir / "run_info.json.tmp"
        with open(temp_filepath, "w") as f:
            json.dump(run_info, f, indent=2)
        os.replace(str(temp_filepath), str(self._temp_dir / "run_info.json"))

    def cleanup(self, force: bool = False):
        """Handles cleanup of temporary files and directories.

        If `force` is True or the run status is "completed", the temporary
        directory is deleted. Otherwise, failed runs are moved to a recovery
        directory.

        Args:
            force (bool, optional): If True, forces cleanup regardless of run status. Defaults to False.
        """
        if not self._temp_dir or not self._temp_dir.exists():
            return

        info_file = self._temp_dir / "run_info.json"
        if not info_file.exists():
            # If run_info is missing, assume it's a failed setup and force cleanup
            shutil.rmtree(str(self._temp_dir), ignore_errors=True)
            return

        try:
            with open(info_file) as f:
                run_info = json.load(f)
        except json.JSONDecodeError:
            # Handle potentially corrupted run_info file
            printlog(
                f"Warning: Could not decode run_info.json for {self._run_id}. Forcing cleanup."
            )
            shutil.rmtree(str(self._temp_dir), ignore_errors=True)
            return

        if force or run_info["status"] == "completed":
            shutil.rmtree(str(self._temp_dir), ignore_errors=True)
        else:
            # Move failed runs to recovery directory
            recovery_dir = Path.home() / "dockm8_recovery" / f"failed_{self.name}_{self._run_id}"
            recovery_dir.parent.mkdir(exist_ok=True)
            try:
                shutil.move(str(self._temp_dir), str(recovery_dir))
                printlog(f"Run failed - files preserved in: {recovery_dir}")
            except OSError as e:
                printlog(
                    f"Error moving temporary directory for failed run {self._run_id}: {e}"
                )

    @abstractmethod
    def dock_batch(
        self,
        batch_file: Path,
        protein_file: Path,
        pocket_definition: dict,
        exhaustiveness: int,
        n_poses: int,
    ) -> Path | None:
        """Docks a batch of compounds. Must be implemented by subclasses.

        Args:
            batch_file (Path): Path to the SDF file containing the batch of ligands.
            protein_file (Path): Path to the receptor file.
            pocket_definition (dict): Dictionary defining the docking pocket.
            exhaustiveness (int): Exhaustiveness parameter for docking.
            n_poses (int): Number of poses to generate per ligand.

        Returns:
            Path | None: Path to the processed SDF file containing docking results,
                         or None if the batch failed.
        """
        pass

    @classmethod
    def resume_from_recovery(cls, recovery_dir: Path) -> Optional["DockingFunction"]:
        """Creates a new GninaDocking instance from a failed run.

        Args:
            recovery_dir (Path): Path to the recovery directory containing the failed run's files.

        Returns:
            GninaDocking | None: A new instance of GninaDocking, or None if recovery fails.
        """
        info_file = recovery_dir / "run_info.json"
        if not info_file.exists():
            printlog(f"Error: run_info.json not found in recovery directory: {recovery_dir}")
            return None
            
        try:
            with open(info_file) as f:
                run_info = json.load(f)
                
            # Create instance with just the software path
            instance = cls(Path(run_info["software_path"]))
            
            # Override the temp directory and run ID to match the recovery
            instance._temp_dir = recovery_dir
            instance._run_id = run_info["run_id"]
            
            printlog(f"Resuming GNINA docking run from {recovery_dir}")
            return instance
            
        except json.JSONDecodeError:
            printlog(f"Error: Could not decode run_info.json in recovery directory: {recovery_dir}")
            return None
        except KeyError as e:
            printlog(f"Error: Missing key in run_info.json: {e}")
            return None
        except Exception as e:
            printlog(f"Error resuming docking run: {str(e)}")
            return None

    def resume_dock(self, n_cpus: int) -> Path | None:
        """Resumes a failed docking run.

        Args:
            n_cpus (int): The number of CPUs to use for parallel processing.

        Returns:
            Path | None: Path to the combined SDF file of the resumed docking run,
                        or None if resuming failed.
        """
        try:
            self._save_run_info("resuming")

            # Get remaining unprocessed batches
            processed_files = {
                f.stem.replace("_processed", "")
                for f in (self._temp_dir / "processed").glob("*.sdf")
            }
            print(len(processed_files))
            all_batches = sorted((self._temp_dir / "splits").glob("split_*.sdf"))
            print(len(all_batches))
            batches_to_process = [b for b in all_batches if b.stem not in processed_files]
            print(len(batches_to_process))

            if not batches_to_process:
                printlog("No remaining batches to process.")
                self._save_run_info("completed")
                output_sdf = self._temp_dir / "combined_results.sdf"
                return output_sdf if output_sdf.exists() else None

            # Load original parameters
            params_file = self._temp_dir / "run_parameters.json"
            if not params_file.exists():
                raise FileNotFoundError(f"run_parameters.json not found in {self._temp_dir}")
            
            with open(params_file) as f:
                params = json.load(f)

            # Convert string paths back to Path objects
            params = {
                k: Path(v) if isinstance(v, str) and k.endswith('_file') else v
                for k, v in params.items()
            }

            # Process remaining batches
            results = parallel_executor(
                self.dock_batch,
                batches_to_process,
                n_cpus=n_cpus,
                job_manager="concurrent_process",
                display_name=f"{self.name} docking (resumed)",
                protein_file=params["protein_file"],
                pocket_definition=params["pocket_definition"],
                exhaustiveness=params["exhaustiveness"],
                n_poses=params["n_poses"],
            )

            # Combine all processed results
            output_sdf = params.get("output_sdf", self._temp_dir / "combined_results.sdf")
            if isinstance(output_sdf, str):
                output_sdf = Path(output_sdf)
                
            self._combine_results_atomic(
                [f for f in (self._temp_dir / "processed").glob("*.sdf")],
                output_sdf
            )
            
            self._save_run_info("completed")
            return output_sdf

        except Exception as e:
            error_msg = f"ERROR in resumed docking: {str(e)}"
            self._save_run_info("failed", error_msg)
            printlog(error_msg)
            return None
        finally:
            self.cleanup()

    def _combine_results_atomic(self, input_files: list[Path], output_file: Path) -> None:
        """Combines multiple SDF files atomically.

        Writes the combined output to a temporary file and then atomically
        replaces the existing output file to prevent data corruption.

        Args:
            input_files (list[Path]): A list of paths to the SDF files to combine.
            output_file (Path): The path to the output SDF file.
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)
        temp_output = output_file.with_suffix(".sdf.tmp")
        try:
            with open(temp_output, "wb") as outfile:
                for file_path in sorted(input_files):
                    if file_path.exists() and file_path.stat().st_size > 0:
                        with open(file_path, "rb") as infile:
                            shutil.copyfileobj(infile, outfile)
            os.replace(str(temp_output), str(output_file))
        except Exception as e:
            printlog(f"Error combining results: {e}")
            if temp_output.exists():
                temp_output.unlink()
            raise

    def dock(
        self,
        library: pd.DataFrame | Path,
        protein_file: Path,
        pocket_definition: dict[str, Any],
        exhaustiveness: int,
        n_poses: int,
        n_cpus: int,
        output_sdf: Path | None = None,
        split_mode: str = "single",
    ) -> Path | None:
        """Main docking method.

        Performs the docking of a library of compounds against a protein target.

        Args:
            library (pd.DataFrame | Path): Pandas DataFrame or path to the SDF file
                                            containing the ligands.
            protein_file (Path): Path to the receptor file.
            pocket_definition (dict[str, Any]): Dictionary defining the docking pocket.
            exhaustiveness (int): Exhaustiveness parameter for docking.
            n_poses (int): Number of poses to generate per ligand.
            n_cpus (int): The number of CPUs to use for parallel processing.
            output_sdf (Path | None, optional): Path to save the final combined
                                                SDF file. Defaults to None.
            split_mode (str, optional): Mode for splitting the input library
                                         ('single' or 'cpu'). Defaults to "single".

        Returns:
            Path | None: Path to the final combined SDF file, or None if docking failed.
        """
        try:
            self._save_run_info("preparing")  # Initial status of the dock run

            # Save parameters for potential recovery - do this early
            params_to_save = {
                k: str(v) if isinstance(v, Path) else v
                for k, v in locals().items()
                if k not in ["self", "library"]
            }
            with open(self._temp_dir / "run_parameters.json", "w") as f:
                json.dump(params_to_save, f, indent=2)

            # Handle input library
            if isinstance(library, pd.DataFrame):
                library_path = self._temp_dir / "input_library.sdf"
                PandasTools.WriteSDF(
                    library, str(library_path), molColName="Molecule", idName="ID"
                )
            else:
                library_path = library

            # Split input
            batch_files = split_sdf(
                library_path,
                self._temp_dir / "splits",
                mode=split_mode,
                splits=n_cpus if split_mode == "cpu" else None,
            )

            if not batch_files:
                raise ValueError("No valid batches created")

            self._save_run_info("processing")  # Update status before starting heavy work

            # Run docking
            results = parallel_executor(
                self.dock_batch,
                sorted(batch_files.glob("split_*.sdf")),
                n_cpus=n_cpus,
                job_manager="concurrent_process",
                display_name=f"{self.name} docking",
                protein_file=protein_file,
                pocket_definition=pocket_definition,
                exhaustiveness=exhaustiveness,
                n_poses=n_poses,
            )

            valid_results = [r for r in results if r is not None]

            # Combine results
            final_output = output_sdf or (self._temp_dir / "combined_results.sdf")
            self._combine_results_atomic(
                [f for f in (self._temp_dir / "processed").glob("*.sdf")], final_output
            )
            self._save_run_info("completed")
            return final_output

        except Exception as e:
            error_msg = f"ERROR in docking: {str(e)}"
            self._save_run_info("failed", error_msg)
            printlog(error_msg)
            return None

        finally:
            if isinstance(library, pd.DataFrame) and 'library_path' in locals() and Path(library_path).exists():
                Path(library_path).unlink(missing_ok=True)
            self.cleanup()

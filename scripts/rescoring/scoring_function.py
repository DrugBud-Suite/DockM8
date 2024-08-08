import os
import shutil
import sys
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.setup.software_manager import ensure_software_installed
from scripts.utilities.logging import printlog


class ScoringFunction(ABC):

	"""
    Abstract base class for scoring functions used in DockM8.
    """

	def __init__(self,
					name: str,
					column_name: str,
					best_value: str,
					score_range: Tuple[float, float],
					software_path: Path):
		"""
        Initialize a ScoringFunction object.

        Args:
            name (str): The name of the scoring function.
            column_name (str): The name of the column where the scores will be stored.
            best_value (str): The best value for the scoring function ('min' or 'max').
            score_range (tuple): The range of scores for the scoring function.
            software_path (Path): The path to the software installation directory.
        """
		self.name = name
		self.column_name = column_name
		self.best_value = best_value
		self.score_range = score_range
		self.software_path = Path(software_path)

	@abstractmethod
	def rescore(self, sdf_file: str, n_cpus: int, protein_file: str, **kwargs) -> pd.DataFrame:
		"""
        Rescore the molecules in the given SDF file using the scoring function.

        Args:
            sdf_file (str): The path to the SDF file.
            n_cpus (int): The number of CPUs to use for parallel processing.
            protein_file (str): The path to the protein file.
            **kwargs: Additional keyword arguments specific to the scoring function.

        Returns:
            pd.DataFrame: A DataFrame containing the rescored molecules.
        """
		pass

	def get_info(self) -> Dict[str, Any]:
		"""
        Get information about the scoring function.

        Returns:
            dict: A dictionary containing the name, column name, best value, and score range of the scoring function.
        """
		return {
			"name": self.name,
			"column_name": self.column_name,
			"best_value": self.best_value,
			"range": self.score_range}

	def create_temp_dir(self) -> Path:
		"""
        Create a temporary directory for the scoring function.

        Returns:
            Path: The path to the temporary directory.
        """
		os.makedirs(Path.home() / "dockm8_temp_files", exist_ok=True)
		return Path(tempfile.mkdtemp(dir=Path.home() / "dockm8_temp_files", prefix=f"dockm8_{self.name}_"))

	@staticmethod
	def remove_temp_dir(temp_dir: Path):
		"""
        Remove the temporary directory.

        Args:
            temp_dir (Path): The path to the temporary directory.
        """
		shutil.rmtree(str(temp_dir), ignore_errors=True)

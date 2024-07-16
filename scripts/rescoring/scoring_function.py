import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import pandas as pd


class ScoringFunction(ABC):

	"""
	Abstract base class for scoring functions used in DockM8.
	"""

	def __init__(self, name: str, column_name: str, best_value: str, score_range: tuple):
		"""
		Initializes a ScoringFunction object.

		Args:
			name (str): The name of the scoring function.
			column_name (str): The name of the column where the scores will be stored.
			best_value (str): The best value for the scoring function.
			score_range (tuple): The range of scores for the scoring function.
		"""
		self.name = name
		self.column_name = column_name
		self.best_value = best_value
		self.score_range = score_range

	@abstractmethod
	def rescore(self, sdf: str, n_cpus: int, protein_file: str, **kwargs) -> pd.DataFrame:
		"""
		Rescores the molecules in the given SDF file using the scoring function.

		Args:
			sdf (str): The path to the SDF file.
			n_cpus (int): The number of CPUs to use for parallel processing.
			protein_file (str): The path to the protein file.
			**kwargs: Additional keyword arguments specific to the scoring function.

		Returns:
			pd.DataFrame: A DataFrame containing the rescored molecules.
		"""
		pass

	def get_info(self) -> Dict[str, Any]:
		"""
		Returns information about the scoring function.

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
		Creates a temporary directory for the scoring function.

		Returns:
			Path: The path to the temporary directory.
		"""
		os.makedirs(Path.home() / "dockm8_temp_files", exist_ok=True)
		return Path(tempfile.mkdtemp(dir=Path.home() / "dockm8_temp_files", prefix=f"dockm8_{self.name}_"))

	@staticmethod
	def remove_temp_dir(temp_dir: Path):
		"""
		Removes the temporary directory.

		Args:
			temp_dir (str): The path to the temporary directory.
		"""
		shutil.rmtree(str(temp_dir), ignore_errors=True)

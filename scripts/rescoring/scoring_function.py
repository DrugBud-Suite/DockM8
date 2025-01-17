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


class ScoringFunction(ABC):
    def __init__(
        self, name: str, column_name: str, best_value: str, score_range: Tuple[float, float], software_path: Path
    ):
        self.name = name
        self.column_name = column_name
        self.best_value = best_value
        self.score_range = score_range
        self.software_path = Path(software_path)
        self._temp_dir = None
        self.create_temp_dir()

    def create_temp_dir(self) -> Path:
        """
        Creates and returns the temporary directory for operations.
        Creates it if it doesn't exist.
        """
        if self._temp_dir is None:
            base_temp = Path.home() / "dockm8_temp_files"
            os.makedirs(base_temp, exist_ok=True)
            self._temp_dir = base_temp / f"dockm8_{self.name.lower()}_{os.getpid()}"
            self._temp_dir.mkdir(parents=True, exist_ok=True)
        return self._temp_dir

    def cleanup(self):
        """
        Cleans up the temporary directory and resets the reference.
        """
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(str(self._temp_dir), ignore_errors=True)

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
            "range": self.score_range,
        }

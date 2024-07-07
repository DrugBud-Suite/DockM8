from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional


class ScoringFunction(ABC):

	def __init__(self, name: str, column_name: str, best_value: str, score_range: tuple):
		self.name = name
		self.column_name = column_name
		self.best_value = best_value
		self.score_range = score_range

	@abstractmethod
	def rescore(self, sdf: str, n_cpus: int, **kwargs) -> pd.DataFrame:
		pass

	def get_info(self) -> Dict[str, Any]:
		return {
			"name": self.name,
			"column_name": self.column_name,
			"best_value": self.best_value,
			"range": self.score_range}

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))


@dataclass
class DockM8Results:
	docked_poses: Dict[str, Path]                                       # Key is the docking program
	all_poses: Path
	processed_poses: Path
	selected_poses: Dict[str, Path]                                     # Key is the selection method
	rescored_poses: Dict[str, Path]                                     # Key is the selection method
	consensus_results: Optional[Dict[str, Path]] = None                 # Key is the selection method
	docking_programs: List[str] = field(default_factory=list)
	selection_methods: List[str] = field(default_factory=list)

	def to_dict(self):
		return {
			"docked_poses": {
				k: str(v) for k, v in self.docked_poses.items()},
			"all_poses": str(self.all_poses),
			"processed_poses": str(self.processed_poses),
			"selected_poses": {
				k: str(v) for k, v in self.selected_poses.items()},
			"rescored_poses": {
				k: str(v) for k, v in self.rescored_poses.items()},
			"consensus_results": {
				k: str(v) for k, v in self.consensus_results.items()} if self.consensus_results else None,
			"docking_programs": self.docking_programs,
			"selection_methods": self.selection_methods, }

import os
from functools import wraps
from pathlib import Path
import sys
import subprocess
from typing import Callable, Dict, Optional

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.setup.software_install import (
	install_aa_score, install_censible, install_convex_pl, install_dligand2,
	install_fabind, install_gnina, install_gypsum_dl, install_itscoreAff,
	install_korp_pl, install_lin_f9, install_posecheck, install_psovina,
	install_qvina2, install_qvina_w, install_rf_score_vs, install_rtmscore,
	install_scorch, install_plants, install_panther, install_plantain,
	install_genscore, install_mgltools
)
from scripts.utilities.logging import printlog

def check_mgltools() -> bool:
	env_name = 'mgltools'
	try:
		result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True, check=True)
		return env_name in result.stdout
	except subprocess.CalledProcessError:
		printlog(f"Error checking for {env_name} environment")
		return False

# Define a dictionary mapping docking programs to their installation functions and check methods
SOFTWARE_INFO: Dict[str, Dict[str, Callable]] = {
	"GNINA": {
		"install": install_gnina, "check": lambda path: (path / "gnina").is_file()},
	"QVINAW": {
		"install": install_qvina_w, "check": lambda path: (path / "qvina-w").is_file()},
	"QVINA2": {
		"install": install_qvina2, "check": lambda path: (path / "qvina2.1").is_file()},
	"PSOVINA": {
		"install": install_psovina, "check": lambda path: (path / "psovina").is_file()},
	"PLANTS": {
		"install": install_plants, "check": lambda path: (path / "PLANTS").is_file()},
	"FABind": {
		"install": install_fabind, "check": lambda path: (path / "FABind").is_dir()},
	"PANTHER": {
		"install": install_panther, "check": lambda path: (path / "panther").is_dir()},
	"PLANTAIN": {
		"install": install_plantain, "check": lambda path: (path / "plantain").is_dir()},
	"AA_SCORE": {
		"install": install_aa_score, "check": lambda path: (path / "AA-Score-Tool-main").is_dir()},
	"CONVEX_PLR": {
		"install": install_convex_pl, "check": lambda path: (path / "Convex_PL").is_file()},
	"CENSIBLE": {
		"install": install_censible, "check": lambda path: (path / "censible").is_file()},
	"DLIGAND2": {
		"install": install_dligand2, "check": lambda path: (path / "DLIGAND2").is_dir()},
	"LIN_F9": {
		"install": install_lin_f9, "check": lambda path: (path / "LinF9").is_file()},
	"GYPSUM_DL": {
		"install": install_gypsum_dl, "check": lambda path: (path / "gypsum_dl-1.2.1").is_dir()},
	"SCORCH": {
		"install": install_scorch, "check": lambda path: (path / "SCORCH-1.0.0").is_dir()},
	"RF_SCORE_VS": {
		"install": install_rf_score_vs, "check": lambda path: (path / "rf-score-vs").is_file()},
	"RTMSCORE": {
		"install": install_rtmscore, "check": lambda path: (path / "RTMScore-main").is_dir()},
	"IT_SCORE_AFF": {
		"install": install_itscoreAff, "check": lambda path: (path / "ITScoreAff_v1.0").is_dir()},
	"KORP_PL": {
		"install": install_korp_pl, "check": lambda path: (path / "KORP-PL").is_file()},
	"POSECHECK": {
		"install": install_posecheck, "check": lambda path: (path / "posecheck-main").is_file()},
	"GENSCORE": {
		"install": install_genscore, "check": lambda path: (path / "GenScore").is_dir()},
	"POSECHECK": {
		"install": install_posecheck, "check": lambda path: (path / "posecheck-main").is_dir()},
	"MGLTOOLS": {
		"install": install_mgltools, "check": check_mgltools}}

def ensure_software_installed(program_name: str):
	"""
    Decorator function that ensures the specified software is installed before executing the decorated function.

    Args:
        program_name (str): The name of the software to be installed.

    Returns:
        function: The decorated function.

    Raises:
        ValueError: If the software path is not provided or if the program is not in SOFTWARE_INFO.
    """
	def decorator(func):
		@wraps(func)
		def wrapper(*args, **kwargs):
			if program_name not in SOFTWARE_INFO:
				raise ValueError(f"Unknown program: {program_name}")

			# Assuming the software path is always the second argument
			software_path = args[1] if len(args) > 1 else kwargs.get('software_path')

			if software_path is None:
				raise ValueError("Software path not provided")

			software_path = Path(software_path)
			if not is_software_installed(program_name, software_path):
				install_software(program_name, software_path)

			return func(*args, **kwargs)
		return wrapper
	return decorator

def is_software_installed(program_name: str, software_path: Path) -> bool:
	"""
    Check if the specified software is installed.

    Args:
        program_name (str): The name of the software program.
        software_path (Path): The path to the software installation.

    Returns:
        bool: True if the software is installed, False otherwise.
    """
	check_func = SOFTWARE_INFO.get(program_name, {}).get("check")
	if check_func is None:
		printlog(f"No check function found for {program_name}")
		return False
	return check_func(software_path)

def install_software(program_name: str, software_path: Path) -> None:
	"""
    Install the specified software using the provided installation function.

    Args:
        program_name (str): The name of the software to install.
        software_path (Path): The path to the software.

    Raises:
        Exception: If an error occurs during installation.
    """
	install_func = SOFTWARE_INFO.get(program_name, {}).get("install")
	if install_func is None:
		printlog(f"No installation function found for {program_name}")
		return

	printlog(f"Downloading and installing {program_name}...")
	try:
		install_func(software_path)
		printlog(f"{program_name} has been successfully installed.")
	except Exception as e:
		printlog(f"Error installing {program_name}: {str(e)}")
		raise

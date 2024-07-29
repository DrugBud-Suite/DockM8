import os
import pytest
from pathlib import Path
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

# Import all installation functions
from scripts.setup.software_install import (install_gnina,
											install_qvina_w,
											install_qvina2,
											install_psovina,
											install_plants,
											install_panther,
											install_plantain,
											install_korp_pl,
											install_convex_pl,
											install_lin_f9,
											install_aa_score,
											install_gypsum_dl,
											install_scorch,
											install_rf_score_vs,
											install_rtmscore,
											install_posecheck,
											install_mgl_tools,
											install_fabind,
											install_censible,
											install_dligand2,
											install_itscoreAff,
											install_genscore,
											install_deepcoy_models)


@pytest.fixture(scope="module")
def software_path():
	# Create a directory under the specified path
	software_dir = dockm8_path / "tests" / "test_files" / "setup" / "test_software"
	software_dir.mkdir(parents=True, exist_ok=True)
	logging.info(f"Created software directory at: {software_dir}")
	return software_dir


def log_and_assert(software_path, name, file_or_dir):
	full_path = software_path / file_or_dir
	if full_path.exists():
		logging.info(f"{name} was successfully installed at: {full_path}")
		assert True
	else:
		logging.error(f"{name} installation failed. Expected path does not exist: {full_path}")
		assert False


def test_install_gnina(software_path):
	logging.info("Starting GNINA installation")
	install_gnina(software_path)
	log_and_assert(software_path, "GNINA", 'gnina')


def test_install_qvina_w(software_path):
	logging.info("Starting QVINA-W installation")
	install_qvina_w(software_path)
	log_and_assert(software_path, "QVINA-W", 'qvina-w')


def test_install_qvina2(software_path):
	logging.info("Starting QVINA2 installation")
	install_qvina2(software_path)
	log_and_assert(software_path, "QVINA2", 'qvina2.1')


def test_install_psovina(software_path):
	logging.info("Starting PSOVINA installation")
	install_psovina(software_path)
	log_and_assert(software_path, "PSOVINA", 'psovina')


def test_install_plants(software_path, capsys):
	logging.info("Starting PLANTS installation")
	install_plants(software_path)
	captured = capsys.readouterr()
	logging.info(f"PLANTS installation output: {captured.out}")
	assert "PLANTS automatic installation is not supported" in captured.out


def test_install_panther(software_path, capsys):
	logging.info("Starting PANTHER installation")
	install_panther(software_path)
	captured = capsys.readouterr()
	logging.info(f"PANTHER installation output: {captured.out}")
	assert "PANTHER automatic installation is not supported" in captured.out


def test_install_plantain(software_path):
	logging.info("Starting PLANTAIN installation")
	install_plantain(software_path)
	log_and_assert(software_path, "PLANTAIN", 'plantain')


def test_install_korp_pl(software_path):
	logging.info("Starting KORP-PL installation")
	install_korp_pl(software_path)
	log_and_assert(software_path, "KORP-PL", 'KORP-PL')


def test_install_convex_pl(software_path):
	logging.info("Starting Convex-PL installation")
	install_convex_pl(software_path)
	log_and_assert(software_path, "Convex-PL", 'Convex-PL')


def test_install_lin_f9(software_path):
	logging.info("Starting LinF9 installation")
	install_lin_f9(software_path)
	log_and_assert(software_path, "LinF9", 'LinF9')


def test_install_aa_score(software_path):
	logging.info("Starting AA-Score installation")
	install_aa_score(software_path)
	log_and_assert(software_path, "AA-Score", 'AA-Score-Tool-main')


def test_install_gypsum_dl(software_path):
	logging.info("Starting Gypsum-DL installation")
	install_gypsum_dl(software_path)
	log_and_assert(software_path, "Gypsum-DL", 'gypsum_dl-1.2.1')


def test_install_scorch(software_path):
	logging.info("Starting SCORCH installation")
	install_scorch(software_path)
	log_and_assert(software_path, "SCORCH", 'SCORCH-1.0.0')


def test_install_rf_score_vs(software_path):
	logging.info("Starting RF-Score-VS installation")
	install_rf_score_vs(software_path)
	log_and_assert(software_path, "RF-Score-VS", 'rf-score-vs')


def test_install_rtmscore(software_path):
	logging.info("Starting RTMScore installation")
	install_rtmscore(software_path)
	log_and_assert(software_path, "RTMScore", 'RTMScore-main')


def test_install_posecheck(software_path):
	logging.info("Starting PoseCheck installation")
	install_posecheck(software_path)
	log_and_assert(software_path, "PoseCheck", 'posecheck-main')


def test_install_fabind(software_path):
	logging.info("Starting FABind installation")
	install_fabind(software_path)
	log_and_assert(software_path, "FABind", 'FABind')


def test_install_censible(software_path):
	logging.info("Starting CENsible installation")
	install_censible(software_path)
	log_and_assert(software_path, "CENsible", 'censible')


def test_install_dligand2(software_path):
	logging.info("Starting DLIGAND2 installation")
	install_dligand2(software_path)
	log_and_assert(software_path, "DLIGAND2", 'DLIGAND2')


def test_install_itscoreAff(software_path):
	logging.info("Starting ITScoreAff installation")
	install_itscoreAff(software_path)
	log_and_assert(software_path, "ITScoreAff", 'ITScoreAff_v1.0')


def test_install_genscore(software_path):
	logging.info("Starting GenScore installation")
	install_genscore(software_path)
	log_and_assert(software_path, "GenScore", 'GenScore')


def test_install_deepcoy_models(software_path):
	logging.info("Starting DeepCoys models installation")
	install_deepcoy_models(software_path)
	log_and_assert(software_path, "DeepCoys models", 'models')

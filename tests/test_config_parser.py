import os
import sys
from pathlib import Path

import pytest

# Search for 'DockM8' in parent directories
dockm8_path = next(
    (p / "DockM8" for p in Path(__file__).resolve().parents if (p / "DockM8").is_dir()),
    None,
)
sys.path.append(str(dockm8_path))

from scripts.utilities.config_parser import check_config


class DockM8Error(Exception):
    """Custom Error for DockM8 specific issues."""

    pass


class DockM8Warning(Warning):
    """Custom warning for DockM8 specific issues."""

    pass


@pytest.fixture
def common_test_data():
    """Set up common test data."""
    config_file = dockm8_path / "config.yml"
    return config_file


def test_check_config_valid(common_test_data):
    """Test check_config with a valid configuration file."""
    config_file = common_test_data
    config = check_config(config_file)
    assert isinstance(config, dict)
    assert "general" in config
    assert "software" in config["general"]
    assert config["general"]["software"] == Path(str(dockm8_path) + "/software")
    assert "mode" in config["general"]
    assert config["general"]["mode"] == "single"
    assert "ncpus" in config["general"]
    assert config["general"]["ncpus"] == int(os.cpu_count() * 0.9)
    assert "decoy_generation" in config
    assert "gen_decoys" in config["decoy_generation"]
    assert not config["decoy_generation"]["gen_decoys"]
    assert "receptor(s)" in config
    assert isinstance(config["receptor(s)"], list)
    assert len(config["receptor(s)"]) == 1
    assert config["receptor(s)"][0] == Path(
        str(dockm8_path) + "/tests/test_files/1fvv_p.pdb"
    )
    assert "protein_preparation" in config
    assert "select_best_chain" in config["protein_preparation"]
    if config["protein_preparation"]["select_best_chain"]:
        assert config["protein_preparation"]["select_best_chain"]
    assert "fix_nonstandard_residues" in config["protein_preparation"]
    if config["protein_preparation"]["fix_nonstandard_residues"]:
        assert config["protein_preparation"]["fix_nonstandard_residues"]
    assert "fix_missing_residues" in config["protein_preparation"]
    if config["protein_preparation"]["fix_missing_residues"]:
        assert config["protein_preparation"]["fix_missing_residues"]
    assert "remove_heteroatoms" in config["protein_preparation"]
    if config["protein_preparation"]["remove_heteroatoms"]:
        assert config["protein_preparation"]["remove_heteroatoms"]
    assert "remove_water" in config["protein_preparation"]
    if config["protein_preparation"]["remove_water"]:
        assert config["protein_preparation"]["remove_water"]
    assert "protonation" in config["protein_preparation"]
    if config["protein_preparation"]["protonation"]:
        assert config["protein_preparation"]["protonation"]
    assert "add_hydrogens" in config["protein_preparation"]
    assert config["protein_preparation"]["add_hydrogens"] == 7.0
    assert "ligand_preparation" in config
    assert "protonation" in config["ligand_preparation"]
    if config["ligand_preparation"]["protonation"]:
        assert config["ligand_preparation"]["protonation"]
    assert "conformers" in config["ligand_preparation"]
    assert config["ligand_preparation"]["conformers"] == "RDKit"
    assert "n_conformers" in config["ligand_preparation"]
    assert config["ligand_preparation"]["n_conformers"] == 1
    assert "pocket_detection" in config
    assert "method" in config["pocket_detection"]
    assert config["pocket_detection"]["method"] == "Reference"
    assert "reference_ligand(s)" in config["pocket_detection"]
    assert isinstance(config["pocket_detection"]["reference_ligand(s)"], list)
    assert len(config["pocket_detection"]["reference_ligand(s)"]) == 1
    assert config["pocket_detection"]["reference_ligand(s)"][0] == Path(
        str(dockm8_path) + "/tests/test_files/1fvv_l.sdf"
    )
    assert "radius" in config["pocket_detection"]
    assert config["pocket_detection"]["radius"] == 10
    assert "docking" in config
    assert "docking_programs" in config["docking"]
    assert isinstance(config["docking"]["docking_programs"], list)
    assert len(config["docking"]["docking_programs"]) == 2
    assert config["docking"]["docking_programs"][0] == "SMINA"
    assert config["docking"]["docking_programs"][1] == "GNINA"
    assert "bust_poses" in config["docking"]
    if config["docking"]["bust_poses"]:
        assert config["docking"]["bust_poses"]
    assert "nposes" in config["docking"]
    assert config["docking"]["nposes"] == 10
    assert "exhaustiveness" in config["docking"]
    assert config["docking"]["exhaustiveness"] == 8
    assert "pose_selection" in config
    assert "pose_selection_method" in config["pose_selection"]
    assert isinstance(config["pose_selection"]["pose_selection_method"], list)
    assert len(config["pose_selection"]["pose_selection_method"]) == 2
    assert config["pose_selection"]["pose_selection_method"][0] == "bestpose"
    assert config["pose_selection"]["pose_selection_method"][1] == "bestpose_GNINA"
    assert "clustering_method" in config["pose_selection"]
    assert config["pose_selection"]["clustering_method"] == "KMedoids"
    assert "rescoring" in config
    assert isinstance(config["rescoring"], list)
    assert len(config["rescoring"]) == 2
    assert config["rescoring"][0] == "CNN-Score"
    assert config["rescoring"][1] == "KORP-PL"
    assert "consensus" in config
    assert config["consensus"] == "ECR_best"
    assert "threshold" in config
    assert config["threshold"] == 0.01

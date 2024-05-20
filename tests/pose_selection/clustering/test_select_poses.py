from pathlib import Path
import sys
import os
import shutil

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools


# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.pose_selection.pose_selection import select_poses


def test_select_poses():
    allposes = PandasTools.LoadSDF(str(dockm8_path / "tests" / "test_files" / "pose_selection" / "allposes.sdf"), molColName="Molecule", idName="Pose ID")

    shutil.rmtree(dockm8_path / "tests" / "test_files" / "pose_selection" / "clustering", ignore_errors=True)

    expected_RMSD_output = PandasTools.LoadSDF(str(dockm8_path / "tests" / "test_files" / "pose_selection" / "RMSD_clustered.sdf"), molColName="Molecule", idName="Pose ID")

    select_poses(
        selection_method="RMSD",
        clustering_method="KMedoids",
        w_dir=dockm8_path / "tests" / "test_files" / "pose_selection",
        protein_file=dockm8_path / "tests" / "test_files" / "pose_selection" / "4kd1_p.pdb",
        software=dockm8_path / "software",
        all_poses=allposes,
        n_cpus=int(os.cpu_count() * 0.9))


    RMSD_results = PandasTools.LoadSDF(str(dockm8_path / "tests" / "test_files" / "pose_selection" / "clustering" / "RMSD_clustered.sdf"), molColName="Molecule", idName="Pose ID")

    assert len(RMSD_results) == len(expected_RMSD_output)
    assert set(RMSD_results["Pose ID"]) == set(expected_RMSD_output["Pose ID"])

    expected_GNINA_output = PandasTools.LoadSDF(str(dockm8_path / "tests" / "test_files" / "pose_selection" / "bestpose_GNINA_clustered.sdf"), molColName="Molecule", idName="Pose ID")

    select_poses(
        selection_method="bestpose_GNINA",
        clustering_method="KMedoids",
        w_dir=dockm8_path / "tests" / "test_files" / "pose_selection",
        protein_file=dockm8_path / "tests" / "test_files" / "pose_selection" / "4kd1_p.pdb",
        software=dockm8_path / "software",
        all_poses=allposes,
        n_cpus=int(os.cpu_count() * 0.9))


    bestpose_GNINA_results = PandasTools.LoadSDF(str(dockm8_path / "tests" / "test_files" / "pose_selection" / "clustering" / "bestpose_GNINA_clustered.sdf"), molColName="Molecule", idName="Pose ID")

    assert len(bestpose_GNINA_results) == len(expected_GNINA_output)
    assert set(bestpose_GNINA_results["Pose ID"]) == set(expected_GNINA_output["Pose ID"])

    expected_KORPL_output = PandasTools.LoadSDF(str(dockm8_path / "tests" / "test_files" / "pose_selection" / "KORP-PL_clustered.sdf"), molColName="Molecule", idName="Pose ID")

    select_poses(
        selection_method="KORP-PL",
        clustering_method="KMedoids",
        w_dir=dockm8_path / "tests" / "test_files" / "pose_selection",
        protein_file=dockm8_path / "tests" / "test_files" / "pose_selection" / "4kd1_p.pdb",
        software=dockm8_path / "software",
        all_poses=allposes,
        n_cpus=int(os.cpu_count() * 0.9))


    KORP_PL_results = PandasTools.LoadSDF(str(dockm8_path / "tests" / "test_files" / "pose_selection" / "clustering" / "KORP-PL_clustered.sdf"), molColName="Molecule", idName="Pose ID")

    assert len(KORP_PL_results) == len(expected_KORPL_output)
    assert set(KORP_PL_results["Pose ID"]) == set(expected_KORPL_output["Pose ID"])
    
    shutil.rmtree(dockm8_path / "tests" / "test_files" / "pose_selection" / "clustering", ignore_errors=True)

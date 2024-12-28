import os
import sys
from pathlib import Path

import pytest
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.docking.plants_docking import PlantsDocking


@pytest.fixture
def test_data():
    dockm8_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None).parent
    protein_file = dockm8_path / "test_data/docking/prepared_protein.pdb"
    library_file = dockm8_path / "test_data/docking/prepared_library.sdf"
    ligand_file = dockm8_path / "test_data/docking/prepared_ligand.sdf"
    software = dockm8_path / "software"
    n_cpus = int(os.cpu_count() * 0.9)
    output_dir = dockm8_path / "test_data/docking/output"
    output_dir.mkdir(exist_ok=True)
    pocket_definition = {"center": [-9.67, 207.73, 113.41], "size": [20.0, 20.0, 20.0]}
    return protein_file, library_file, ligand_file, software, n_cpus, output_dir, pocket_definition


def test_PlantsDocking(test_data):
    protein_file, library_file, ligand_file, software, n_cpus, output_dir, pocket_definition = test_data

    plants = PlantsDocking(software)
    temp_dir = plants._temp_dir
    print(temp_dir)
    # Test dock_batch method with a single ligand file
    result_file = plants.dock_batch(
        batch_file=ligand_file,
        protein_file=protein_file,
        pocket_definition=pocket_definition,
        exhaustiveness=8,  # Not used by PLANTS but kept for interface consistency
        n_poses=10,
    )

    assert result_file is not None
    assert result_file.is_file()

    # Verify the processed output directly
    output_df = PandasTools.LoadSDF(str(result_file), molColName="Molecule", idName="Pose ID")

    # Check the processed output structure
    assert "Pose ID" in output_df.columns
    assert "ID" in output_df.columns
    assert "Molecule" in output_df.columns
    assert "CHEMPLP" in output_df.columns
    assert len(output_df) > 0

    # Check if the number of poses matches the input ligands
    suppl = Chem.SDMolSupplier(str(ligand_file))
    num_input_ligands = len([mol for mol in suppl if mol is not None])
    assert len(output_df) <= num_input_ligands * 10

    # Verify pose naming convention
    assert all(pid.startswith(id_ + "_PLANTS_") for pid, id_ in zip(output_df["Pose ID"], output_df["ID"]))

    # Test the full docking process with the library
    output_sdf = output_dir / "plants_docking_results.sdf"
    if output_sdf.exists():
        output_sdf.unlink()
    final_output = plants.dock(
        library=library_file,
        protein_file=protein_file,
        pocket_definition=pocket_definition,
        exhaustiveness=8,
        n_poses=10,
        n_cpus=n_cpus,
        output_sdf=output_sdf,
    )

    # Verify the output path was returned
    assert isinstance(final_output, Path)
    assert final_output.is_file()

    # Load and verify the final combined output
    final_df = PandasTools.LoadSDF(str(final_output), molColName="Molecule", idName="Pose ID")

    assert "Pose ID" in final_df.columns
    assert "CHEMPLP" in final_df.columns
    assert "ID" in final_df.columns
    assert "Molecule" in final_df.columns
    assert len(final_df) > 0

    # Check if the number of poses matches the input ligands in the library
    library_suppl = Chem.SDMolSupplier(str(library_file))
    num_library_ligands = len([mol for mol in library_suppl if mol is not None])
    assert len(final_df) <= num_library_ligands * 10

    # Verify each compound has correct number of poses and proper ranking
    pose_counts = final_df.groupby("ID").size()
    assert all(count <= 10 for count in pose_counts)  # No more than 10 poses per compound

    # Verify scoring and ranking (PLANTS: lower CHEMPLP scores are better)
    final_df["CHEMPLP"] = pd.to_numeric(final_df["CHEMPLP"], errors="coerce")
    for id_ in final_df["ID"].unique():
        compound_poses = final_df[final_df["ID"] == id_]
        # Scores should be sorted in ascending order (lower is better)
        assert all(compound_poses["CHEMPLP"].diff().fillna(0) >= 0)

    # Verify statistics were tracked
    assert hasattr(plants, "stats")
    assert isinstance(plants.stats, dict)
    assert "processed_batches" in plants.stats
    assert "failed_batches" in plants.stats
    assert "empty_results" in plants.stats

    # Verify all batches were processed successfully
    assert plants.stats["failed_batches"] == 0
    assert plants.stats["empty_results"] == 0
    assert plants.stats["processed_batches"] > 0

    # Additional PLANTS-specific checks
    # Verify that temporary directories are cleaned u
    assert not plants._temp_dir.exists()

    # Verify score ranges are reasonable for CHEMPLP
    assert all(final_df["CHEMPLP"] < 100)  # CHEMPLP scores should be reasonably small
    assert all(final_df["CHEMPLP"] > -200)  # CHEMPLP scores shouldn't be too negative

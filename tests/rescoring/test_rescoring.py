import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.rescoring import RESCORING_FUNCTIONS, rescore_poses


@pytest.fixture
def test_data():
    dockm8_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None).parent
    protein_file = dockm8_path / "test_data/rescoring/example_prepared_receptor_1fvv.pdb"
    software = dockm8_path / "software"
    sdf = dockm8_path / "test_data/rescoring/example_poses_1fvv.sdf"
    pocket_definition = {"center": [-9.67, 207.73, 113.41], "size": [20.0, 20.0, 20.0]}
    functions = [key for key in RESCORING_FUNCTIONS.keys() if key not in ["AAScore", "PLECScore", "RTMScore", "SCORCH"]]
    n_cpus = int(os.cpu_count() * 0.9)
    return protein_file, pocket_definition, software, sdf, functions, n_cpus


def test_rescore_poses(test_data):
    protein_file, pocket_definition, software, sdf, functions, n_cpus = test_data
    output_file = sdf.parent / "rescoring_output.csv"
    if output_file.exists():
        output_file.unlink()
    result = rescore_poses(
        protein_file=protein_file,
        pocket_definition=pocket_definition,
        software=software,
        poses=sdf,
        functions=functions,
        n_cpus=n_cpus,
        output_file=output_file,
    )

    assert output_file.exists()
    assert output_file.with_suffix(".sdf").exists()

    scores_df = pd.read_csv(output_file)
    assert not scores_df.empty
    required_columns = ["Pose ID", "ID", "SMILES"] + [RESCORING_FUNCTIONS[func]["column_name"] for func in functions]
    assert all(col in scores_df.columns for col in required_columns)

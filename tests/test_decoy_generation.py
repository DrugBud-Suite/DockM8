import json
import sys
from pathlib import Path

import pytest

tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))


@pytest.fixture
def deepcoy_paths():
    """Provide paths to DeepCoy test data and software."""
    return {
        "test_data": dockm8_path / "test_data" / "DeepCoy",
        "software": dockm8_path / "software",
        "deepcoy_dir": dockm8_path / "software" / "DeepCoy",
        "config_json": dockm8_path / "software" / "DeepCoy" / "config.json",
    }


class TestDeepCoyConfig:
    """Tests for DeepCoy configuration file integrity."""

    def test_config_json_exists(self, deepcoy_paths):
        assert deepcoy_paths["config_json"].exists(), "DeepCoy config.json not found"

    def test_config_json_valid(self, deepcoy_paths):
        with open(deepcoy_paths["config_json"]) as f:
            config = json.load(f)
        assert "generation" in config
        assert "batch_size" in config

    def test_config_json_no_personal_paths(self, deepcoy_paths):
        with open(deepcoy_paths["config_json"]) as f:
            content = f.read()
        assert "/home/tony" not in content, "Personal paths found in config.json"
        assert "/home/user" not in content, "Hardcoded user paths found in config.json"


class TestDeepCoyTestData:
    """Tests for DeepCoy test data fixture availability."""

    def test_test_data_directory_exists(self, deepcoy_paths):
        assert deepcoy_paths["test_data"].exists(), "test_data/DeepCoy/ directory not found"

    def test_actives_smi_exists(self, deepcoy_paths):
        actives = deepcoy_paths["test_data"] / "actives.smi"
        assert actives.exists(), "actives.smi not found in test data"

    def test_decoys_smi_exists(self, deepcoy_paths):
        decoys = deepcoy_paths["test_data"] / "decoys.smi"
        assert decoys.exists(), "decoys.smi not found in test data"

    def test_test_set_sdf_exists(self, deepcoy_paths):
        test_set = deepcoy_paths["test_data"] / "test_set.sdf"
        assert test_set.exists(), "test_set.sdf not found in test data"

    def test_molecules_json_exists(self, deepcoy_paths):
        molecules = deepcoy_paths["test_data"] / "molecules_actives.json"
        assert molecules.exists(), "molecules_actives.json not found in test data"


class TestDeepCoyImports:
    """Tests for DeepCoy module importability."""

    def test_generate_decoys_import(self):
        from software.DeepCoy.generate_decoys import generate_decoys
        assert callable(generate_decoys)

    def test_select_and_evaluate_import(self):
        from software.DeepCoy.evaluation.select_and_evaluate_decoys import select_and_evaluate_decoys
        assert callable(select_and_evaluate_decoys)

    def test_preprocess_import(self):
        from software.DeepCoy.data.prepare_data import preprocess, read_file
        assert callable(preprocess)
        assert callable(read_file)

    def test_generate_decoys_signature(self):
        import inspect
        from software.DeepCoy.generate_decoys import generate_decoys
        sig = inspect.signature(generate_decoys)
        params = list(sig.parameters.keys())
        assert "input_sdf" in params
        assert "n_decoys" in params
        assert "model" in params
        assert "software" in params


class TestDeepCoyModels:
    """Tests for DeepCoy model file availability."""

    def test_models_directory_exists(self, deepcoy_paths):
        models_dir = deepcoy_paths["software"] / "models"
        assert models_dir.exists(), "software/models/ directory not found"

    @pytest.mark.parametrize("model_pattern", [
        "DeepCoy_DUDE_model_*.pickle",
        "DeepCoy_DEKOIS_model_*.pickle",
        "DeepCoy_DUDE_phosphorus_model_*.pickle",
    ])
    def test_model_files_exist(self, deepcoy_paths, model_pattern):
        models_dir = deepcoy_paths["software"] / "models"
        matches = list(models_dir.glob(model_pattern))
        assert len(matches) > 0, f"No model files matching {model_pattern} found"


class TestDeepCoyPreprocess:
    """Tests for DeepCoy data preprocessing."""

    def test_read_file(self, deepcoy_paths):
        from software.DeepCoy.data.prepare_data import read_file
        actives_smi = deepcoy_paths["test_data"] / "actives.smi"
        data = read_file(actives_smi)
        assert data is not None
        assert len(data) > 0, "read_file returned empty data"

    def test_molecules_json_loadable(self, deepcoy_paths):
        molecules_json = deepcoy_paths["test_data"] / "molecules_actives.json"
        with open(molecules_json) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0, "molecules_actives.json is empty"

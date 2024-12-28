from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from pathlib import Path
import sys

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.parallel_executor import parallel_executor
from scripts.utilities.logging import printlog

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))


def conf_gen_RDKit(molecule: Chem.Mol, forcefield: str = "MMFF"):
    """Generates 3D conformer using RDKit."""
    try:
        if not molecule.GetConformer().Is3D():
            molecule = Chem.AddHs(molecule)
            AllChem.EmbedMolecule(molecule, AllChem.ETKDGv3())
            if forcefield == "MMFF":
                AllChem.MMFFOptimizeMolecule(molecule)
            elif forcefield == "UFF":
                AllChem.UFFOptimizeMolecule(molecule)
            AllChem.SanitizeMol(molecule)
        return molecule
    except Exception as e:
        printlog(f"Error generating conformer: {str(e)}")
        return None


def generate_conformers_RDKit(df: pd.DataFrame, n_cpus: int, forcefield: str) -> pd.DataFrame:
    """Generates 3D conformers for molecules in DataFrame using RDKit."""
    printlog(f"Generating 3D conformers using RDKit with {forcefield}...")

    results = parallel_executor(
        conf_gen_RDKit,
        df["Molecule"].tolist(),
        n_cpus,
        "concurrent_process",
        display_name="Conformer Generation",
        forcefield=forcefield,
    )

    df["Molecule"] = results
    df = df.dropna(subset=["Molecule"])
    df = df[["Molecule", "ID"]]

    return df

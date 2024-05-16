import concurrent.futures
import sys
import warnings
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, PandasTools
from tqdm import tqdm

# Search for 'DockM8' in parent directories
scripts_path = next((p / 'scripts'
                     for p in Path(__file__).resolve().parents
                     if (p / 'scripts').is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.utilities import printlog

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def conf_gen_RDKit(molecule):
    """
    Generates 3D conformers using RDKit.

    Args:
        molecule (RDKit molecule): The input molecule.

    Returns:
        molecule (RDKit molecule): The molecule with 3D conformers.
    """
    if not molecule.GetConformer().Is3D():
        molecule = Chem.AddHs(molecule)  # Add hydrogens to the molecule
        AllChem.EmbedMolecule(molecule, AllChem.ETKDGv3(
        ))  # Generate initial 3D coordinates for the molecule
        AllChem.MMFFOptimizeMolecule(
            molecule)  # Optimize the 3D coordinates using the MMFF force field
        AllChem.SanitizeMol(
            molecule)  # Sanitize the molecule to ensure it is chemically valid
    return molecule


def generate_conformers_RDKit(input_sdf: str, output_dir: str,
                              n_cpus: int) -> Path:
    """
    Generates 3D conformers using RDKit.

    Args:
        input_sdf (str): Path to the input SDF file.
        output_dir (str): Path to the output directory.
        n_cpus (int): Number of CPUs to use for parallel processing.

    Returns:
        output_file (Path): Path to the output SDF file containing the generated conformers.
    """
    printlog('Generating 3D conformers using RDKit...')

    try:
        # Load the input SDF file into a Pandas DataFrame
        df = PandasTools.LoadSDF(str(input_sdf),
                                 idName='ID',
                                 molColName='Molecule',
                                 includeFingerprints=False,
                                 smilesName='SMILES')
        # Generate conformers for each molecule in parallel using the conf_gen_RDKit function
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=n_cpus) as executor:
            df['Molecule'] = list(
                tqdm(executor.map(conf_gen_RDKit, df['Molecule']),
                     total=len(df['Molecule']),
                     desc='Minimizing molecules',
                     unit='mol'))

        # Remove molecules where conformer generation failed
        df = df[df['Molecule'].notna()]

        # Check if the number of compounds matches the input
        input_mols = [
            mol for mol in Chem.SDMolSupplier(str(input_sdf)) if mol is not None
        ]
        if len(input_mols) != len(df):
            printlog(
                "Conformer generation for some compounds failed. Removing compounds from library."
            )

            input_ids = {
                mol.GetProp("_Name")
                for mol in input_mols
                if mol.HasProp("_Name")
            }
            generated_ids = set(df["ID"])
            missing_ids = input_ids - generated_ids

            df = df[df["ID"].isin(input_ids - missing_ids)]

        # Write the conformers to the output SDF file using PandasTools.WriteSDF()
        output_file = output_dir / 'generated_conformers.sdf'
        PandasTools.WriteSDF(df,
                             str(output_file),
                             molColName='Molecule',
                             idName='ID')
    except Exception as e:
        printlog('ERROR: Failed to generate conformers using RDKit!' + e)

    return output_file

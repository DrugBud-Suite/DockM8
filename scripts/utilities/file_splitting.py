import math
import os
import sys
import warnings
from pathlib import Path


# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


from pathlib import Path
from typing import Literal


def count_compounds(sdf_content: str) -> int:
    """Count the number of compounds in an SDF file by counting '$$$$' delimiters."""
    return sdf_content.count('$$$$')

def get_compounds(sdf_content: str) -> list[str]:
    """Split SDF content into individual compounds."""
    # Split on '$$$$' and filter out empty strings
    compounds = [comp + '$$$$\n' for comp in sdf_content.split('$$$$\n') if comp.strip()]
    return compounds

def create_split_dir(sdf_path: Path) -> Path:
    """Create and clean the output directory for split files."""
    sdf_file_name = sdf_path.stem
    split_files_folder = sdf_path.parent / f"split_{sdf_file_name}"
    split_files_folder.mkdir(parents=True, exist_ok=True)
    
    # Clean existing files
    for file in split_files_folder.iterdir():
        file.unlink()
    
    return split_files_folder

def write_compounds(compounds: list[str], output_dir: Path, start_idx: int = 1) -> None:
    """Write compounds to individual files."""
    for idx, compound in enumerate(compounds, start=start_idx):
        output_file = output_dir / f"split_{idx}.sdf"
        output_file.write_text(compound)

def split_compounds_by_size(compounds: list[str], n_splits: int) -> list[list[str]]:
    """Split compounds into approximately equal-sized chunks."""
    compounds_per_split = math.ceil(len(compounds) / n_splits)
    return [compounds[i:i + compounds_per_split]
            for i in range(0, len(compounds), compounds_per_split)]

def split_sdf(
    sdf_path: str | Path,
    output_dir: str | Path,
    mode: Literal["cpu", "count", "single"],
    splits: int | None = None
) -> Path:
    """
    Split an SDF file using string operations according to specified mode.
    
    Args:
        sdf_path: Path to the input SDF file (can be string or Path object)
        output_dir: Directory where split files will be written
        mode: Splitting mode:
              - "cpu": Split based on number of CPUs (splits = number of CPUs)
              - "count": Split into specific number of files
              - "single": Split into individual compounds
        splits: Number of splits (required for "cpu" and "count" modes)
    
    Returns:
        Path to the directory containing split files
    """
    # Convert string paths to Path objects if necessary
    sdf_path = Path(sdf_path)
    output_dir = Path(output_dir)
    
    # Input validation
    if not sdf_path.exists():
        raise FileNotFoundError(f"Input file not found: {sdf_path}")
        
    if mode in ["cpu", "count"] and splits is None:
        raise ValueError(f"Number of splits must be provided for mode '{mode}'")
    
    # Read entire file as text
    sdf_content = sdf_path.read_text()
    
    # Create output directory
    split_files_folder = output_dir / f"split_{sdf_path.stem}"
    split_files_folder.mkdir(parents=True, exist_ok=True)
    
    # Clean existing files
    for file in split_files_folder.iterdir():
        file.unlink()
    
    # Get individual compounds
    compounds = get_compounds(sdf_content)
    total_compounds = len(compounds)
    
    if mode == "single":
        write_compounds(compounds, split_files_folder)
    else:
        # For CPU or count mode, calculate optimal split size
        n_splits = splits
        if mode == "cpu":
            if total_compounds > 100000:
                # Use smaller chunks for large files
                n_splits = splits * 8
            else:
                n_splits = splits
            
        # Split compounds into chunks
        compound_chunks = split_compounds_by_size(compounds, n_splits)
        
        # Write each chunk to a file
        for idx, chunk in enumerate(compound_chunks, start=1):
            chunk_content = ''.join(chunk)
            output_file = split_files_folder / f"split_{idx}.sdf"
            output_file.write_text(chunk_content)
    
    return split_files_folder

def split_pdbqt_str(file):
    """
    Splits a PDBQT file into separate models and saves each model as a separate file.

    Args:
            file (str): The path to the PDBQT file.

    Returns:
            None
    """
    models = []
    current_model = []
    with open(file) as f:
        lines = f.readlines()
    for line in lines:
        current_model.append(line)
        if line.startswith("ENDMDL"):
            models.append(current_model)
            current_model = []
    for i, model in enumerate(models):
        for line in model:
            if line.startswith("MODEL"):
                model_number = int(line.split()[-1])
                break
        output_filename = file.with_name(f"{file.stem}_{model_number}.pdbqt")
        with open(output_filename, "w") as output_file:
            output_file.writelines(model)
    os.remove(file)

def split_sdf_str(dir, sdf_file, n_cpus):
	"""
	Split an SDF file into multiple smaller SDF files based on the number of compounds.

	Args:
		dir (str): The directory where the split SDF files will be saved.
		sdf_file (str): The path to the input SDF file.
		n_cpus (int): The number of CPUs to use for splitting.

	Returns:
		Path: The path to the folder containing the split SDF files.
	"""
	sdf_file_name = Path(sdf_file).name.replace(".sdf", "")
	split_files_folder = Path(dir) / f"split_{sdf_file_name}"
	split_files_folder.mkdir(parents=True, exist_ok=True)

	with open(sdf_file) as infile:
		sdf_lines = infile.readlines()

	total_compounds = sdf_lines.count("$$$$\n")

	if total_compounds > 100000:
		n = max(1, math.ceil(total_compounds // n_cpus // 8))
	else:
		n = max(1, math.ceil(total_compounds // n_cpus // 2))

	compound_count = 0
	file_index = 1
	current_compound_lines = []

	for line in sdf_lines:
		current_compound_lines.append(line)

		if line.startswith("$$$$"):
			compound_count += 1

			if compound_count % n == 0:
				output_file = split_files_folder / f"split_{file_index}.sdf"
				with open(output_file, "w") as outfile:
					outfile.writelines(current_compound_lines)
				current_compound_lines = []
				file_index += 1

	# Write the remaining compounds to the last file
	if current_compound_lines:
		output_file = split_files_folder / f"split_{file_index}.sdf"
		with open(output_file, "w") as outfile:
			outfile.writelines(current_compound_lines)

	return split_files_folder


def split_sdf_single_str(dir, sdf_file):
	"""
	Split an SDF file into individual compounds and save them as separate files.

	Args:
		dir (str): The directory where the split files will be saved.
		sdf_file (str): The path to the input SDF file.

	Returns:
		pathlib.Path: The path to the folder containing the split files.
	"""
	sdf_file_name = Path(sdf_file).name.replace(".sdf", "")
	split_files_folder = Path(dir) / f"split_{sdf_file_name}"
	split_files_folder.mkdir(parents=True, exist_ok=True)

	with open(sdf_file) as infile:
		sdf_lines = infile.readlines()

	compound_count = 0
	current_compound_lines = []

	for line in sdf_lines:
		current_compound_lines.append(line)

		if line.startswith("$$$$"):
			compound_count += 1

			output_file = split_files_folder / f"split_{compound_count}.sdf"
			with open(output_file, "w") as outfile:
				outfile.writelines(current_compound_lines)
			current_compound_lines = []

	# Write the remaining compounds to the last file
	if current_compound_lines:
		compound_count += 1
		output_file = split_files_folder / f"split_{compound_count}.sdf"
		with open(output_file, "w") as outfile:
			outfile.writelines(current_compound_lines)

	return split_files_folder

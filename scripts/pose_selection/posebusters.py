"""
PoseBusters integration for DockM8 - Quality assessment for docking poses.
"""

import sys
import time
from pathlib import Path

from yaml import safe_load

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.fast_sdf_loader import fast_load_sdf
from scripts.utilities.fast_sdf_writer import fast_write_sdf
from scripts.utilities.logging import printlog

# Try to import PoseBusters, with helpful error message if not available
try:
    from posebusters import PoseBusters
except ImportError:
    printlog("ERROR: PoseBusters is not installed. Please install it with pip install posebusters")
    printlog("See https://github.com/maabuu/posebusters for more information")

def bust_poses(
    poses_path: Path,
    protein_file: Path,
    config_path: Path = None,
    output_path: Path = None,
    n_cpus: int = 1,
) -> Path:
    """
    Apply PoseBusters to filter out low-quality poses based on various criteria.
    
    Args:
        poses_path (Path): Path to SDF file containing docking poses
        protein_file (Path): Path to PDB file of the protein structure
        config_path (Path, optional): Path to PoseBusters configuration file.
            If None, uses default config in scripts/posebusters/posebusters_config.yml
        output_path (Path, optional): Path to save filtered poses.
            If None, creates a new file with '_busted' suffix
        n_cpus (int, optional): Number of CPU cores to use
        
    Returns:
        Path: Path to the filtered poses SDF file
        
    Raises:
        ImportError: If PoseBusters is not installed
        FileNotFoundError: If input files are not found
        RuntimeError: If pose busting fails
    """
    try:
        tic = time.perf_counter()
        printlog(f"Busting poses from {poses_path}...")
        
        # Check if input files exist
        if not poses_path.exists():
            raise FileNotFoundError(f"Poses file not found: {poses_path}")
        if not protein_file.exists():
            raise FileNotFoundError(f"Protein file not found: {protein_file}")
        
        # Set default config path if not provided
        if config_path is None:
            config_path = scripts_path / "posebusters" / "posebusters_config.yml"
            if not config_path.exists():
                config_path = Path(__file__).parent / "posebusters_config.yml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"PoseBusters config file not found: {config_path}")
        
        # Set default output path if not provided
        if output_path is None:
            output_path = poses_path.parent / f"{poses_path.stem}_busted{poses_path.suffix}"
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load poses using fast_sdf_loader
        printlog("Loading poses...")
        poses_df = fast_load_sdf(
            poses_path,
            molColName="Molecule",
            idName="Pose ID",
            batch_size=100,
            n_cpus=n_cpus
        )
        
        initial_pose_count = len(poses_df)
        printlog(f"Loaded {initial_pose_count} poses")
        
        if initial_pose_count == 0:
            printlog("No poses to bust, returning original file")
            return poses_path
        
        # Initialize PoseBusters with configuration file
        printlog("Initializing PoseBusters...")
        config = safe_load(open(config_path))
        buster = PoseBusters(config=config)
        
        # Prepare DataFrame for PoseBusters (requires specific column names)
        printlog("Preparing poses for PoseBusters...")
        poses_for_busting = poses_df.copy()
        poses_for_busting['mol_cond'] = str(protein_file)
        poses_for_busting = poses_for_busting.rename(columns={'Molecule': 'mol_pred'})
        
        # Apply PoseBusters checks to the poses
        printlog("Applying PoseBusters quality checks...")
        busted_df = buster.bust_table(poses_for_busting)
        
        # Columns to check for validity (can be customized based on config)
        cols_to_check = [
            'all_atoms_connected',
            'bond_lengths',
            'bond_angles',
            'internal_steric_clash',
            'aromatic_ring_flatness',
            'double_bond_flatness',
            'protein-ligand_maximum_distance',
        ]
        
        # Filter out invalid poses using positional correspondence
        printlog("Filtering out invalid poses...")
        valid_mask = (busted_df[cols_to_check] != False).all(axis=1)
        filtered_poses_df = poses_df[valid_mask.values].reset_index(drop=True)

        final_pose_count = len(filtered_poses_df)
        printlog(f"Removed {initial_pose_count - final_pose_count} poses, {final_pose_count} remain")

        if final_pose_count > 0:
            printlog(f"Saving filtered poses to {output_path}...")
            fast_write_sdf(
                df=filtered_poses_df,
                output_path=output_path,
                molColName="Molecule",
                idName="Pose ID",
                properties=list(filtered_poses_df.columns),
                batch_size=100,
                n_cpus=n_cpus
            )
        else:
            printlog("No valid poses remain after filtering")
            output_path.touch()
        
        toc = time.perf_counter()
        printlog(f"PoseBusters completed in {toc - tic:.2f} seconds")
        
        return output_path
        
    except FileNotFoundError as e:
        printlog(f"ERROR: {e}")
        # Re-raise the caught exception to be handled by the caller.
        raise
    except Exception as e:
        printlog(f"ERROR: An unexpected error occurred during pose busting: {e}")
        # Raise a new RuntimeError that wraps the original exception.
        raise RuntimeError("Pose busting failed.") from e

import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import pandas as pd
import requests

# Search for 'DockM8' in parent directories
scripts_path = next((p / 'scripts' for p in Path(__file__).resolve().parents if (p / 'scripts').is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.utilities import printlog


def find_pocket_p2rank(software: Path, receptor: Path, radius: int):
    """
    Finds the pocket coordinates using p2rank software.

    Args:
        software (Path): The path to the p2rank software directory.
        receptor (Path): The path to the receptor file.
        radius (int): The radius of the pocket.

    Returns:
        dict: A dictionary containing the pocket coordinates with keys 'center' and 'size'.
    """
    p2rank_path = software / "p2rank" / "prank"

    # Check if p2rank executable is available
    if not os.path.exists(p2rank_path):
        print("p2rank executable not found. Downloading...")

        # Use GitHub API to get latest release info
        repo_url = "https://api.github.com/repos/rdk/p2rank/releases/latest"
        response = requests.get(repo_url)
        data = response.json()

        # Find the tarball URL in the assets
        tarball_url = None
        for asset in data["assets"]:
            if asset["name"].endswith(".tar.gz"):
                tarball_url = asset["browser_download_url"]
                break

        if tarball_url is None:
            print("No tarball found in the latest release.")
            return

        # Download p2rank tarball
        tarball_path = software / "p2rank.tar.gz"  # Adjust path as needed
        urllib.request.urlretrieve(tarball_url, tarball_path)

        # Extract p2rank tarball
        subprocess.run(["tar", "-xzf", tarball_path, "-C", software])
        os.unlink(tarball_path)
        # Find the folder in the software directory that starts with "p2rank"
        p2rank_folder = next(
            (
                software / folder
                for folder in os.listdir(software)
                if folder.startswith("p2rank")
            ),
            None,
        )
        os.rename(p2rank_folder, software / "p2rank")

        printlog("p2rank executable downloaded and installed successfully.")
    else:
        pass
    # Create a directory to store output
    output_dir = receptor.parent / "p2rank_output"
    os.makedirs(output_dir, exist_ok=True)

    # Run p2rank with the receptor and output directory
    p2rank_command = (
        f'{software / "p2rank" / "prank"} predict -f {receptor} -o {output_dir}'
    )
    subprocess.run(
        p2rank_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    # Load the predictions file
    predictions_file = output_dir / f"{receptor.name}_predictions.csv"
    df = pd.read_csv(predictions_file)
    # Rename columns to remove spaces
    df.columns = df.columns.str.replace(' ', '')

    pocket_coordinates = {
        "center": (df["center_x"][0], df["center_y"][0], df["center_z"][0]),
        "size": [float(radius) * 2, float(radius) * 2, float(radius) * 2]
    }
    # Remove the output directory
    shutil.rmtree(output_dir)
    return pocket_coordinates

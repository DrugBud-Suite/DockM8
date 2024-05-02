from pathlib import Path
from scripts.pocket_finding.default import find_pocket_default
from scripts.pocket_finding.radius_of_gyration import find_pocket_RoG
from scripts.pocket_finding.dogsitescorer import find_pocket_dogsitescorer
from scripts.pocket_finding.manual import parse_pocket_coordinates

def pocket_finder(mode: str, w_dir: Path = None, receptor: Path = None, ligand: Path = None, radius: int = 10):
    # Determine the docking pocket
    if mode == 'Reference':
        pocket_definition = find_pocket_default(ligand, receptor, radius)
    elif mode == 'RoG':
        pocket_definition = find_pocket_RoG(ligand, receptor)
    elif mode == 'Dogsitescorer':
        pocket_definition = find_pocket_dogsitescorer(receptor, w_dir, method='volume')
    else:
        pocket_definition = parse_pocket_coordinates(mode)
    return pocket_definition
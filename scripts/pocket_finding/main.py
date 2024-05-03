from pathlib import Path
from scripts.utilities import printlog
from scripts.pocket_finding.default import find_pocket_default
from scripts.pocket_finding.radius_of_gyration import find_pocket_RoG
from scripts.pocket_finding.dogsitescorer import find_pocket_dogsitescorer
from scripts.pocket_finding.manual import parse_pocket_coordinates
from scripts.pocket_finding.p2rank import find_pocket_p2rank

def pocket_finder(mode: str, software: Path = None, receptor: Path = None, ligand: Path = None, radius: int = 10):
    # Determine the docking pocket
    if mode == 'Reference':
        pocket_definition = find_pocket_default(ligand, receptor, radius)
    elif mode == 'RoG':
        pocket_definition = find_pocket_RoG(ligand, receptor)
    elif mode == 'Dogsitescorer':
        pocket_definition = find_pocket_dogsitescorer(receptor, method='volume')
    elif mode == 'p2rank':
        pocket_definition = find_pocket_p2rank(software, receptor, radius)
    else:
        pocket_definition = parse_pocket_coordinates(mode)
    printlog(f'Pocket definition: {pocket_definition}')
    return pocket_definition
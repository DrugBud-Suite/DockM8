# DockM8 External Software

This directory contains external tools used by DockM8 for molecular docking and scoring.

## Bundled (tracked in git)

| Tool | Description | Why bundled |
|------|-------------|-------------|
| `DeepCoy/` | Decoy molecule generator | Modified for Python 3 / TensorFlow 2 compatibility |

## Downloaded by setup script (`setup_py310.sh`)

| Tool | Description | Source |
|------|-------------|--------|
| `gnina` | GPU-accelerated docking | <https://github.com/gnina/gnina> |
| `qvina2.1` | Fast Vina variant | <https://github.com/QVina/qvina> |
| `qvina-w` | Wide-search Vina variant | <https://github.com/QVina/qvina> |
| `LinF9` | Linear scoring function | <https://github.com/cyangNYU/Lin_F9_test> |
| `KORP-PL` | Knowledge-based scoring | <https://files.inria.fr/NanoDFiles/Website/Software/KORP-PL/> |
| `Convex-PL` | Convex scoring function | <https://files.inria.fr/NanoDFiles/Website/Software/Convex-PL/> |
| `rf-score-vs` | Random forest scoring | <https://github.com/oddt/rfscorevs_binary> |
| `AA-Score-Tool-main/` | Atom-atom scoring | <https://github.com/Xundrug/AA-Score-Tool> |
| `gypsum_dl-1.2.1/` | Ligand preparation | <https://github.com/durrantlab/gypsum_dl> |
| `SCORCH-1.0.0/` | ML scoring function | <https://github.com/SMVDGroup/SCORCH> |
| `RTMScore-main/` | Graph neural network scoring | <https://github.com/sc8668/RTMScore> |
| `GenScore/` | Generalized scoring | <https://github.com/sc8668/GenScore> |
| `models/` | Pre-trained model weights | Downloaded by setup script |

## Requires separate license

| Tool | Description | How to obtain |
|------|-------------|---------------|
| `PLANTS` | Protein-Ligand ANT System docking | Register at <http://www.tcd.uni-konstanz.de/research/plants.php> |

PLANTS is optional. All other docking engines work without it.

## Conda environments

Some scoring functions require separate conda environments (created by `setup_py310.sh`):

| Environment | Purpose |
|------------|---------|
| `dockm8` | Main DockM8 environment |
| `mgltools` | MGLTools for PDBQT conversion (Python 2.7) |
| `genscore` | GenScore scoring function (Python 3.8) |
| `rtmscore` | RTMScore scoring function (Python 3.8) |

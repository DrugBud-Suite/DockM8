# Regression: select_poses must not SILENTLY drop a compound that has valid docked
# poses. Root cause (scripts/pose_selection/pose_selection.py): groupby("ID")[sf].idxmin()
# returns NaN for a compound whose poses are all-NaN for the selection score, and
# .loc[...] then removes it with no trace -- shrinking the scored pool and inflating
# enrichment (the "compound-pool aggregation bug"). The fix surfaces the exclusion with
# a WARNING and drops the NaN indices explicitly. With complete scores it is a no-op.
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.pose_selection import pose_selection

SEL = "AD4"  # RESCORING_FUNCTIONS[SEL] -> column_name "AD4", best_value "min"


def _poses(scores_by_compound):
    rows = []
    for cid, scores in scores_by_compound.items():
        for i, s in enumerate(scores, start=1):
            rows.append({"Pose ID": f"{cid}_gnina_{i}", SEL: s})
    return pd.DataFrame(rows)


def test_all_nan_compound_is_reported_not_silently_dropped(capsys):
    # C002's poses are all-NaN for the selection score. Input has NaN, so select_poses
    # routes through rescore_poses -- mock it to return the poses unchanged so we
    # exercise the selection logic on NaN data.
    df = _poses({"C000": [8.5, 7.2], "C001": [6.0, 9.1], "C002": [np.nan, np.nan]})
    with patch.object(pose_selection, "rescore_poses", side_effect=lambda **kw: kw["poses"]):
        selected = pose_selection.select_poses(
            selection_method=SEL,
            poses_input=df.copy(),
            protein_file=Path("/nonexistent/protein.pdb"),
            pocket_definition={},
            software=Path("/nonexistent"),
            n_cpus=1,
        )
    ids = set(selected["ID"])
    assert {"C000", "C001"} <= ids, "compounds with valid scores must be retained"
    assert "C002" not in ids, "all-NaN-score compound cannot be selected"
    assert len(selected) == 2, "one representative pose per retained compound"
    out = capsys.readouterr().out
    assert "WARNING" in out and "C002" in out, "the exclusion must be explicit, not silent"


def test_complete_scores_retains_every_compound_without_warning(capsys):
    # No NaN -> no rescore, no drop, no warning: proves the guard is a no-op on
    # complete data (no divergence from prior behavior).
    df = _poses({"C000": [8.5, 7.2], "C001": [6.0, 9.1], "C002": [5.5, 4.4]})
    selected = pose_selection.select_poses(
        selection_method=SEL,
        poses_input=df.copy(),
        protein_file=Path("/nonexistent/protein.pdb"),
        pocket_definition={},
        software=Path("/nonexistent"),
        n_cpus=1,
    )
    assert set(selected["ID"]) == {"C000", "C001", "C002"}
    assert len(selected) == 3
    # best_value=min -> lowest score per compound: C000->7.2, C001->6.0, C002->4.4
    best = selected.set_index("ID")[SEL].to_dict()
    assert best == {"C000": 7.2, "C001": 6.0, "C002": 4.4}
    assert "WARNING" not in capsys.readouterr().out

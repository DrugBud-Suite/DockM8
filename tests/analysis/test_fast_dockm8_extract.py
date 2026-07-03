"""Parity tests: fast DockM8-best/median vs the canonical originals.

The originals (``analyze_dockm8_best`` / ``analyze_dockm8_median``) are instant
on tiny hand-crafted data, so we assert the fast vectorized versions return
dicts that are frame-for-frame, value-for-value, component-pick-for-pick equal.

Edge cases covered by the synthetic fixture:
  * nulls in criteria columns (must sort last / worst),
  * a primary-criterion tie broken by a later criterion,
  * a target column whose winning row's value is null (row must be skipped),
  * a workflow row that is all-null for a target (must never win the median).
"""

import sys
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.data_extraction import (
    analyze_dockm8_best,
    analyze_dockm8_median,
)
from analysis.fast_dockm8_extract import (
    analyze_dockm8_best_fast,
    analyze_dockm8_median_fast,
)


def _canonicalize(df: pl.DataFrame) -> pl.DataFrame:
    """Sort rows/cols for order-independent comparison.

    Two equal result frames must compare equal regardless of construction order
    (the originals iterate targets via set/dict order).
    """
    sort_cols = [c for c in ['Target', 'Model', 'Percentile'] if c in df.columns]
    return df.select(sorted(df.columns)).sort(sort_cols)


def _assert_results_equal(fast: dict, ref: dict):
    assert set(fast.keys()) == set(ref.keys()), f"metric keys differ: {fast.keys()} vs {ref.keys()}"
    for metric in ref:
        f_thr = fast[metric]
        r_thr = ref[metric]
        assert set(f_thr.keys()) == set(r_thr.keys()), f"{metric}: threshold keys differ"
        for thr in r_thr:
            assert_frame_equal(
                _canonicalize(f_thr[thr]),
                _canonicalize(r_thr[thr]),
                check_dtypes=False,
            )


def _make_data() -> dict:
    """6 workflow rows, 3 targets, with the edge cases described in the module docstring.

    Workflow rows (docking, scoring, consensus_method, selection_method):
      r0 ..  r5  -- unique tuples.

    Targets: t1, t2, t3.
    """
    comp = {
        'docking':          ['gnina', 'gnina', 'plants', 'smina', 'smina', 'qvina2'],
        'scoring':          ['CNN',   'KORP',  'CHEMPLP', 'AD4',  'Vinardo', 'LinF9'],
        'consensus_method': ['zscore', 'rbr',  'ecr',    'zscore', 'rbr',   'ecr'],
        'selection_method': ['CNN',   'KORP',  'CHEMPLP', 'AD4',  'Vinardo', 'LinF9'],
    }

    def frame(t1, t2, t3):
        return pl.DataFrame({**comp, 't1': t1, 't2': t2, 't3': t3})

    # ref@1.0 : primary criterion for "best", and the median source.
    #   t1: r2 highest (0.9). r0 has a null (sorts last).
    #   t2: r0 and r1 TIE at 0.8 (primary tie) -> broken by ref@0.1 below.
    #   t3: every row null -> all-null target column (best must skip t3 entirely;
    #        for median t3 contributes nothing).
    ref_1 = frame(
        t1=[None, 0.2, 0.9, 0.5, 0.1, 0.3],
        t2=[0.8, 0.8, 0.4, 0.2, 0.6, 0.1],
        t3=[None, None, None, None, None, None],
    )
    # ref@0.1 : secondary criterion. Breaks the t2 primary tie: r1 > r0.
    ref_01 = frame(
        t1=[0.4, 0.2, 0.7, 0.5, 0.1, 0.3],
        t2=[0.3, 0.9, 0.4, 0.2, 0.6, 0.1],
        t3=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    )
    pm_1 = frame(
        t1=[0.4, 0.2, 0.7, 0.5, 0.1, 0.3],
        t2=[0.3, 0.4, 0.4, 0.2, 0.6, 0.1],
        t3=[0.5, 0.6, 0.5, 0.5, 0.5, 0.5],
    )
    bedroc = frame(
        t1=[0.4, 0.2, 0.7, 0.5, 0.1, 0.3],
        t2=[0.3, 0.4, 0.4, 0.2, 0.6, 0.1],
        # winning row of t1 (r2) has null bedroc -> best bedroc row for t1 skipped
        t3=[0.5, 0.6, 0.5, 0.5, 0.5, 0.5],
    )
    bedroc = bedroc.with_columns(
        pl.when(pl.arange(0, bedroc.height) == 2).then(None).otherwise(pl.col('t1')).alias('t1')
    )
    auc = frame(
        t1=[0.4, 0.2, 0.7, 0.5, 0.1, 0.3],
        t2=[0.3, 0.4, 0.4, 0.2, 0.6, 0.1],
        t3=[0.5, 0.6, 0.5, 0.5, 0.5, 0.5],
    )

    # Other ef/ref/pm thresholds present so the extraction loop has frames too.
    ef_1 = frame(
        t1=[0.4, 0.2, 0.7, 0.5, 0.1, 0.3],
        t2=[0.3, 0.4, 0.4, 0.2, 0.6, 0.1],
        t3=[None, None, None, None, None, None],
    )

    return {
        'ef': {0.1: ef_1.clone(), 1.0: ef_1.clone()},
        'ref': {0.1: ref_01, 1.0: ref_1, 5.0: ref_01.clone()},
        'pm': {0.1: pm_1.clone(), 1.0: pm_1},
        'bedroc': {None: bedroc},
        'auc_roc': {None: auc},
    }


def test_best_fast_matches_original():
    data = _make_data()
    ref = analyze_dockm8_best(data, None)
    fast = analyze_dockm8_best_fast(data, None)
    _assert_results_equal(fast, ref)


def test_median_fast_matches_original():
    data = _make_data()
    ref = analyze_dockm8_median(data)
    fast = analyze_dockm8_median_fast(data)
    _assert_results_equal(fast, ref)


def test_frontier_narrowing_matches_lexsort():
    """Frontier-narrowing must equal np.lexsort on randomized tie/null cases.

    This is the exactness proof for the memory-bounded DEKOIS selection path.
    """
    import numpy as np

    rng = np.random.default_rng(12345)

    def lexsort_winner(keys):
        return int(np.lexsort(keys[::-1])[0])

    def frontier_winner(keys):
        cand = np.arange(len(keys[0]))
        for ci, key in enumerate(keys):
            if ci == 0:
                cand = np.flatnonzero(key == key.min())
            else:
                sub = key[cand]
                cand = cand[sub == sub.min()]
        return int(cand.min())

    for _ in range(5000):
        nrows = int(rng.integers(1, 12))
        ncrit = int(rng.integers(1, 6))
        keys = []
        for _ in range(ncrit):
            v = rng.integers(0, 3, size=nrows).astype(float)
            v[rng.random(nrows) < 0.3] = np.nan
            keys.append(np.where(np.isnan(v), np.inf, -v))
        assert lexsort_winner(keys) == frontier_winner(keys)


def test_best_picks_expected_workflows():
    """Lock the specific edge-case picks so a regression in tie/null handling fails loudly."""
    data = _make_data()
    fast = analyze_dockm8_best_fast(data, None)
    # ref@1.0 best per target:
    best_ref1 = fast['ref'][1.0].sort('Target')
    rows = {r['Target']: r for r in best_ref1.to_dicts()}
    # t1 winner = r2 (ref 0.9), but only present where the value is non-null.
    assert rows['t1']['REF'] == 0.9
    assert rows['t1']['scoring'] == 'CHEMPLP'
    # t2 primary tie r0/r1 at 0.8 broken by ref@0.1 (r1=0.9 > r0=0.3) -> r1 wins.
    assert rows['t2']['REF'] == 0.8
    assert rows['t2']['scoring'] == 'KORP'
    # t3 all-null in ref@1.0 -> no t3 row in ref@1.0 best frame.
    assert 't3' not in rows
    # bedroc: t1 winner r2 has null bedroc -> t1 absent from bedroc best frame.
    bedroc_targets = set(fast['bedroc'][None]['Target'].to_list())
    assert 't1' not in bedroc_targets


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))

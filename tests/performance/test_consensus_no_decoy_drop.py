# ROOT CAUSE (Task 1 finding): compounds are silently dropped at
# scripts/performance/analyzer.py:249-251. This is CASE (a) -- a consensus METHOD
# emits NaN scores (NOT case b: consensus_result keeps every input ID, so id_to_idx
# maps them all). The methods zscore, rbv and softrbv all divide by
# `values.std(axis=0)` (scripts/consensus/methods/zscore.py:11,
# rbv_tiebreaker.py:7, soft_rbv.py:41-44). Whenever a scoring-function column has
# ZERO variance across the (scoring INTERSECT activity) universe -- e.g. a
# constant/degenerate normalized column -- std==0 makes that whole column NaN, so
# every compound's consensus score is NaN. At analyzer.py:250
# `valid_mask = ~np.isnan(ordered_scores)` then removes ALL of them:
# `final_scores`/`final_labels` become empty, calculate_metrics raises "No active
# compounds", the (combination, method) row is swallowed by the `except` at :270 and
# never reaches the output.
#
# PRODUCTION FIX (Task 2): dockm8.py now imports run_consensus_analysis from
# scripts.performance.fast_analyzer instead of scripts.performance.analyzer.
# fast_analyzer is the engine that produced `corrected_final` (the published,
# corrected benchmark data) and is strictly better on this degenerate input: it
# keeps 223/285 (combination, method) pairs, vs analyzer.py's 192/285. It also
# normalizes the consensus_method label "soft_rbv" -> "softrbv" on output
# (fast_analyzer.py, in analyze()), matching the corrected_final / _METHODS
# convention. scripts/performance/analyzer.py is UNCHANGED and retains the
# original, more severe decoy-drop bug (its own tests stay green, untouched).
#
# NOTE: fast_analyzer.py still silently drops the "rbv" and "softrbv" rows (not
# ecr/rbr/zscore) for combinations containing the degenerate column: rbv/soft_rbv
# compute a z-score-based tiebreaker that divides by values.std(axis=0), producing
# NaN; roc_auc_score then raises "Input contains NaN", which
# _apply_consensus_method (fast_analyzer.py:609) catches and papers over by
# returning np.zeros(...), and _process_combination (fast_analyzer.py:641) then
# `continue`s past the all-zero result -- silently dropping that
# (combination, method) row. This is a real residual gap versus the "no compound
# ever dropped" ideal described for the new engines (analysis/cupy_consensus.py);
# it is NOT fixed by this task (routing to the already-corrected/published engine
# is the only change in scope). This test therefore asserts against fast_analyzer's
# actual, verified output shape (223/285 pairs on this fixture), not against a
# hypothetical full fix.

import sys
from pathlib import Path

import numpy as np
import pandas as pd

tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

import scripts.performance.fast_analyzer as analyzer_mod
from scripts.performance.fast_analyzer import run_consensus_analysis

# 6 scoring functions -> C(6,2)+...+C(6,6) = 57 combinations; with batch_size=25 the
# analyzer splits into 3 batches, so n_jobs=2 forces the real ProcessPoolExecutor
# multi-batch path (not a single in-process batch).
SFS = ["AD4", "CHEMPLP", "ConvexPLR", "GNINA-Affinity", "CNN-Score", "KORP-PL"]
# The genuine-NaN trigger: KORP-PL is constant -> zero variance -> zscore/rbv/softrbv
# emit all-NaN for every combination containing it.
DEGENERATE_SF = "KORP-PL"
N_COMPOUNDS = 40
N_ACTIVES = 8


def _write_inputs(tmp_path):
    rng = np.random.default_rng(20260703)
    ids = [f"C{i:03d}" for i in range(N_COMPOUNDS)]

    scoring = pd.DataFrame({"ID": ids})
    for sf in SFS:
        scoring[sf] = rng.random(N_COMPOUNDS)
    # Genuine-NaN case: a constant (zero-variance) scoring column.
    scoring[DEGENERATE_SF] = 7.0

    activity = pd.DataFrame(
        {"ID": ids, "Activity": [1] * N_ACTIVES + [0] * (N_COMPOUNDS - N_ACTIVES)}
    )

    s_path = tmp_path / "scoring.csv"
    a_path = tmp_path / "activity.csv"
    scoring.to_csv(s_path, index=False)
    activity.to_csv(a_path, index=False)
    return s_path, a_path, N_COMPOUNDS


def _install_metric_probe(record_path: Path):
    """Wrap calculate_metrics to record how many scores actually reach it.

    Start method is 'fork', so this wrapper -- installed in the parent BEFORE the
    ProcessPoolExecutor spawns workers -- is inherited by every worker. Each call
    appends its received score count to `record_path`, letting the parent recover the
    true non-NaN count that reached the metric across all processes.
    """
    real_calculate_metrics = analyzer_mod.calculate_metrics

    def probed(scores, labels, percentile, *args, **kwargs):
        try:
            with open(record_path, "a") as fh:
                fh.write(f"{len(scores)}\n")
        except Exception:
            pass
        return real_calculate_metrics(scores, labels, percentile, *args, **kwargs)

    analyzer_mod.calculate_metrics = probed
    return real_calculate_metrics


def test_production_engine_drops_fewer_rows_and_emits_softrbv(tmp_path):
    s_path, a_path, n = _write_inputs(tmp_path)

    record_path = tmp_path / "reached_metric.log"
    original = _install_metric_probe(record_path)
    try:
        df = run_consensus_analysis(
            scoring_data_path=s_path,
            activity_data_path=a_path,
            thresholds=[1, 5],
            n_jobs=2,
            include_single=True,
            batch_size=25,
        )
    finally:
        analyzer_mod.calculate_metrics = original

    # --- Evidence 1: number of (combination, method) rows actually produced. ---
    # 57 combinations x 5 methods would yield 285 pairs if nothing were ever dropped.
    # fast_analyzer (the production engine as of Task 2) is strictly better than
    # analyzer.py here (223/285 kept vs 192/285) but is NOT a full fix: rbv/softrbv
    # still divide by values.std(axis=0) for the degenerate column, producing NaN;
    # roc_auc_score then raises "Input contains NaN", which
    # _apply_consensus_method (fast_analyzer.py:609) catches and papers over with
    # np.zeros(...), and _process_combination (fast_analyzer.py:641) silently
    # `continue`s past the resulting all-zero score, dropping that
    # (combination, method) row. Only the 31 combinations containing the degenerate
    # SF lose rows, and only for methods rbv/softrbv (2 of 5) -- ecr, rbr and zscore
    # are unaffected. This pins that verified, still-imperfect-but-improved contract.
    consensus_rows = df[df["combination"].notna()]
    produced_pairs = set(
        zip(consensus_rows["combination"], consensus_rows["consensus_method"])
    )
    n_combos = 57
    n_methods = 5
    expected_pairs = n_combos * n_methods
    n_degenerate_combos = 31  # combinations containing KORP-PL
    n_methods_dropped_per_combo = 2  # rbv, softrbv
    expected_kept_pairs = expected_pairs - (n_degenerate_combos * n_methods_dropped_per_combo)
    assert len(produced_pairs) == expected_kept_pairs, (
        f"expected {expected_kept_pairs}/{expected_pairs} (combination, method) pairs "
        f"from fast_analyzer (analyzer.py's more severe drop keeps only 192); got "
        f"{len(produced_pairs)}. fast_analyzer's drop behavior may have changed."
    )
    assert len(produced_pairs) > 192, (
        "fast_analyzer must keep strictly more pairs than the buggy analyzer.py (192)"
    )

    # --- Evidence 2: production output uses the "softrbv" label, not "soft_rbv". ---
    labels = set(df["consensus_method"].dropna().unique())
    assert "softrbv" in labels, f"expected 'softrbv' label in output, got {labels}"
    assert "soft_rbv" not in labels, (
        f"internal dispatch key 'soft_rbv' leaked into output labels: {labels}"
    )
    assert labels == {"ecr", "rbr", "rbv", "softrbv", "zscore"}, (
        f"unexpected consensus_method label set: {labels}"
    )

    # --- Evidence 3: every score-set that reached calculate_metrics had all N. ---
    # (For the pairs that DO make it through, no partial/short compound set reaches
    # the metric -- confirms the kept rows are not themselves silently truncated.)
    reached = [int(x) for x in record_path.read_text().split()] if record_path.exists() else []
    short_calls = [c for c in reached if c != n]
    assert not short_calls, (
        f"{len(short_calls)} calculate_metrics calls received fewer than {n} "
        f"compounds (min={min(reached) if reached else 'n/a'})."
    )

    # --- Evidence 4: the reported n_compounds column never shrinks below N. ---
    short_rows = df[df["n_compounds"] != n]
    assert short_rows.empty, (
        f"{len(short_rows)} rows report n_compounds != {n} (silent compound drop):\n"
        f"{short_rows[['combination', 'consensus_method', 'n_compounds']].head(10)}"
    )

    # --- Oracle cross-check (optional): the corrected contract keeps every pair. ---
    # `reference_consensus_analysis` (analysis/cupy_consensus.py) implements the
    # intended "constant-column 0/0 -> 0, keep compound" contract on a fixed universe.
    # On this exact input it retains all expected_pairs while fast_analyzer still
    # drops rbv/softrbv rows for degenerate combinations -- confirming fast_analyzer
    # is an improvement, not the final fix. Soft-skip if the untracked oracle is
    # unavailable, so this test stays self-contained.
    try:
        from analysis.cupy_consensus import reference_consensus_analysis
    except Exception:
        reference_consensus_analysis = None
    if reference_consensus_analysis is not None:
        ref = reference_consensus_analysis(
            str(s_path), str(a_path), thresholds=[1, 5], include_single=True
        )
        ref_pairs = ref[ref["combination"].notna()][
            ["combination", "consensus_method"]
        ].drop_duplicates()
        assert len(ref_pairs) == expected_pairs, (
            f"oracle produced {len(ref_pairs)} pairs (expected {expected_pairs}); "
            f"the analyzer produced {len(produced_pairs)} -- gap = silently dropped."
        )

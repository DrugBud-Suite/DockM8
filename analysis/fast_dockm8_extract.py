#!/usr/bin/env python3
"""
Fast, memory-lean reimplementation of the DockM8-best / DockM8-median selection.

This module produces numerically identical output to
``analysis.data_extraction.analyze_dockm8_best`` and
``analysis.data_extraction.analyze_dockm8_median`` while avoiding the
per-row Python loops, per-target sorts of multi-million-row frames, and the
~1100 full multi-key string joins of the originals.

Key facts exploited (verified on the corrected v1.1 aggregates):
  * ``dockm8_data`` is ``{metric: {threshold: polars.DataFrame}}`` where every
    frame is a WIDE pivot: the four WORKFLOW_COMPONENTS columns (Utf8) followed
    by one Float64 column per target.
  * Every pivot shares a byte-identical row order (component columns are
    ``.equals()`` across all (metric, threshold) frames) and workflow
    component-tuples are unique. Therefore workflow row ``i`` is the same
    workflow in every frame, joins are unnecessary, and positional indexing
    aligns everything.

The originals' config/helpers are imported and reused (not duplicated).

Direct usage:
    python -m analysis.fast_dockm8_extract --aggregated-dir <dir> --output-dir <dir>
"""

import logging
import os
from pathlib import Path

import numpy as np
import polars as pl

from . import data_extraction as de
from .data_extraction import (
    BEST_CRITERIA,
    CUSTOM_WORKFLOWS_RESULTS,
    DATASET_AGGREGATED_DIRS,
    INTERNAL_DOCKING_METHODS,
    INTERNAL_SCORING_FUNCTIONS,
    METRIC_FILE_MAP,
    NON_THRESHOLD_METRICS,
    THRESHOLD_METRICS,
    WORKFLOW_COMPONENTS,
    export_combined_results,
)
from .config import get_aggregated_dir, get_dockm8_results_dir

logger = logging.getLogger('DockM8Extraction')

# Map polars metric key -> output metric column name (matches the originals).
_METRIC_COL = {'ef': 'EF', 'ref': 'REF', 'pm': 'PM', 'auc_roc': 'AUC_ROC', 'bedroc': 'BEDROC'}

# Chunk size (rows) for the memory-safe median computation.
_MEDIAN_CHUNK_ROWS = 1_000_000


def _metric_col(metric: str) -> str:
    return _METRIC_COL.get(metric.lower(), metric.upper())


def _gather_rows(df: pl.DataFrame, indices) -> pl.DataFrame:
    """Gather rows by positional index, preserving the requested order."""
    return df[list(indices)]


def _component_cols(dockm8_data: dict) -> list[str]:
    """Component columns in frame column order (first non-empty frame)."""
    for metric_data in dockm8_data.values():
        for df in metric_data.values():
            if df is not None and df.height > 0:
                return [c for c in df.columns if c in WORKFLOW_COMPONENTS]
    return []


# =============================================================================
# DockM8-best (fast)
# =============================================================================

def analyze_dockm8_best_fast(dockm8_data: dict, criteria: list | None = None) -> dict:
    """Vectorized equivalent of ``analyze_dockm8_best``.

    For each target the winning workflow row is the lexicographic argmin over
    the present BEST_CRITERIA keys, where the original sort key is
    ``inf if value is null else -value`` (so higher metric values rank first and
    nulls rank last/worst), and the FIRST present criterion is primary. This is
    reproduced with ``np.lexsort`` (stable; final ties broken by original row
    order, matching the originals' stable single-threaded slice(0,1)).
    """
    logger.info("Running DockM8-best analysis (fast)")
    if not criteria:
        criteria = BEST_CRITERIA

    component_cols = _component_cols(dockm8_data)
    if not component_cols:
        return {}

    all_targets: set = set()
    for thresh_dfs in dockm8_data.values():
        for df in thresh_dfs.values():
            if df is not None and df.height > 0:
                all_targets.update([c for c in df.columns if c not in component_cols])

    # Which criteria are actually available, in BEST_CRITERIA order.
    present_criteria = []
    for criterion in criteria:
        metric, threshold = criterion['metric'], criterion['threshold']
        if metric in dockm8_data and threshold in dockm8_data[metric]:
            df = dockm8_data[metric][threshold]
            if df is not None and df.height > 0:
                present_criteria.append(criterion)

    if not present_criteria:
        return {}

    # Pull each criterion frame's target columns once (Float64, exact).
    # Build, per target, the lexsort winner index.
    best_idx_by_target: dict[str, int] = {}
    for target in sorted(all_targets):
        # Collect criterion key arrays for criteria present AND containing target.
        keys = []  # in present-criteria order; reversed for lexsort below
        for criterion in present_criteria:
            df = dockm8_data[criterion['metric']][criterion['threshold']]
            if target not in df.columns:
                continue
            vals = df[target].to_numpy()
            vals = np.asarray(vals, dtype=np.float64)
            # sort key: inf if null else -value  (ascending sort => best first)
            key = np.where(np.isnan(vals), np.inf, -vals)
            keys.append(key)
        if not keys:
            continue
        # np.lexsort: LAST sequence is the primary key. The original's primary
        # is the FIRST present criterion, so reverse the key list.
        order = np.lexsort(keys[::-1])
        best_idx_by_target[target] = int(order[0])

    if not best_idx_by_target:
        return {}

    # Extraction: for each frame gather only the needed (unique) winning rows.
    unique_idx = sorted(set(best_idx_by_target.values()))
    pos_of_idx = {idx: pos for pos, idx in enumerate(unique_idx)}

    best_results: dict = {}
    for metric, threshold_dfs in dockm8_data.items():
        best_results[metric] = {}
        mcol = _metric_col(metric)
        for threshold, df in threshold_dfs.items():
            if df is None or df.height == 0:
                continue
            gathered = _gather_rows(df, unique_idx)
            comp_arrays = {c: gathered[c].to_list() for c in component_cols}
            df_cols = set(df.columns)
            result_rows = []
            for target, best_idx in best_idx_by_target.items():
                if target not in df_cols:
                    continue
                pos = pos_of_idx[best_idx]
                target_val = gathered[target][pos]
                if target_val is None:
                    continue
                r = {'Target': target, 'Model': 'DockM8-best', 'Percentile': threshold, mcol: target_val}
                for comp in component_cols:
                    r[comp] = comp_arrays[comp][pos]
                result_rows.append(r)
            if result_rows:
                best_results[metric][threshold] = pl.DataFrame(result_rows)

    return best_results


# =============================================================================
# DockM8-median (fast)
# =============================================================================

def _row_median_argmax(ref_df: pl.DataFrame, target_cols: list[str]) -> int:
    """Index of the row with the highest non-null median across target columns.

    Memory-safe: processes the frame in row chunks, computing
    ``np.nanmedian(axis=1)`` per chunk and tracking the global first-argmax
    (np.nanargmax semantics: an all-null row never wins). Matches the original
    ``max(range, key=median or -inf)`` first-on-ties behaviour.
    """
    height = ref_df.height
    best_val = -np.inf
    best_idx = 0
    found_any = False
    sel = ref_df.select(target_cols)
    for start in range(0, height, _MEDIAN_CHUNK_ROWS):
        end = min(start + _MEDIAN_CHUNK_ROWS, height)
        chunk = sel.slice(start, end - start).to_numpy()
        with np.errstate(invalid='ignore'):
            med = np.nanmedian(chunk, axis=1)
        # rows that are entirely null -> nan median -> treated as -inf (never win)
        valid = ~np.isnan(med)
        if valid.any():
            found_any = True
            local = np.where(valid, med, -np.inf)
            local_argmax = int(np.argmax(local))  # first max within chunk
            local_max = local[local_argmax]
            if local_max > best_val:
                best_val = local_max
                best_idx = start + local_argmax
    if not found_any:
        # All rows all-null: original picks the first index (max of all -inf).
        return 0
    return best_idx


def analyze_dockm8_median_fast(dockm8_data: dict) -> dict:
    """Vectorized equivalent of ``analyze_dockm8_median`` (uses REF@1%)."""
    logger.info("Running DockM8-median analysis (fast)")
    median_results: dict = {}

    if 'ref' not in dockm8_data or 1.0 not in dockm8_data['ref'] or dockm8_data['ref'][1.0] is None:
        logger.error("REF@1% not found")
        return median_results

    ref_df = dockm8_data['ref'][1.0]
    component_cols = [c for c in ref_df.columns if c in WORKFLOW_COMPONENTS]
    target_cols = [c for c in ref_df.columns if c not in component_cols]

    best_idx = _row_median_argmax(ref_df, target_cols)

    best_comp_row = _gather_rows(ref_df, [best_idx])
    best_workflow = {comp: best_comp_row[comp][0] for comp in component_cols}
    logger.info(f"Best median workflow: {best_workflow}")

    for metric, threshold_dfs in dockm8_data.items():
        median_results[metric] = {}
        mcol = _metric_col(metric)
        for threshold, df in threshold_dfs.items():
            if df is None or df.height == 0:
                continue
            df_target_cols = [c for c in df.columns if c not in component_cols]
            best_row = _gather_rows(df, [best_idx])
            result_rows = []
            for target in df_target_cols:
                tv = best_row[target][0]
                if tv is not None:
                    r = {'Target': target, 'Model': 'DockM8-median', 'Percentile': threshold, mcol: tv}
                    for comp, val in best_workflow.items():
                        r[comp] = val
                    result_rows.append(r)
            if result_rows:
                median_results[metric][threshold] = pl.DataFrame(result_rows)

    return median_results


# =============================================================================
# Memory-lean selection (reads only the columns/rows needed)
# =============================================================================

def _thresh_str(threshold) -> str:
    # Match load_dockm8_data's naming: THRESHOLD_METRICS uses int 1/5 -> "1"/"5"
    # and float 0.1/0.5 -> "0p1"/"0p5". BEST_CRITERIA / median use float 1.0,
    # which equals int 1 as a dict key, so normalize integer-valued floats to int.
    if isinstance(threshold, float):
        if threshold.is_integer():
            return str(int(threshold))
        return str(threshold).replace('.', 'p')
    return str(threshold)


def _pivot_path(data_dir: str, metric: str, threshold) -> str | None:
    fname = METRIC_FILE_MAP[metric].format(thresh=_thresh_str(threshold)) if threshold is not None \
        else METRIC_FILE_MAP[metric]
    csv_path = os.path.join(data_dir, fname)
    pq_path = csv_path[:-4] + ".parquet"
    if os.path.exists(pq_path):
        return pq_path
    if os.path.exists(csv_path):
        return csv_path
    return None


def _read_columns(path: str, columns: list[str] | None = None) -> pl.DataFrame:
    if path.endswith(".parquet"):
        return pl.read_parquet(path, columns=columns)
    return pl.read_csv(path, columns=columns)


def _all_columns(path: str) -> list[str]:
    """Column names without reading data (parquet schema / CSV header)."""
    if path.endswith(".parquet"):
        return list(pl.read_parquet_schema(path).keys())
    return pl.read_csv(path, n_rows=0).columns


def _select_indices_lean(data_dir: str, excluded_sf: str | None = None,
                         selection_dtype: pl.DataType = pl.Float64):
    """Compute best_idx_by_target and median best_idx without holding all frames.

    Reads only the BEST_CRITERIA columns (+ REF@1 target columns) needed for the
    selection math. When ``excluded_sf`` is set, rows containing it (scoring or
    selection_method) are dropped first so positional indices map into the
    filtered row space (matching ``filter_dockm8_data`` + originals).

    Returns ``(component_cols, all_targets, best_idx_by_target, median_best_idx,
    keep_mask_or_None)``. ``keep_mask`` (numpy bool, length = original rows) is
    returned when filtering so the extraction phase can apply the same filter.
    """
    # Reference path for layout / row identity: REF@1.0 (always present here).
    ref1_path = _pivot_path(data_dir, 'ref', 1.0)
    if ref1_path is None:
        raise FileNotFoundError(f"REF@1 pivot not found in {data_dir}")
    all_cols = _all_columns(ref1_path)
    component_cols = [c for c in all_cols if c in WORKFLOW_COMPONENTS]
    target_cols = [c for c in all_cols if c not in component_cols]

    keep_mask = None
    if excluded_sf is not None:
        comp_df = _read_columns(ref1_path, component_cols)
        scoring = comp_df['scoring'].fill_null("").to_numpy() if 'scoring' in component_cols else None
        sel = comp_df['selection_method'].to_numpy() if 'selection_method' in component_cols else None
        mask = np.ones(comp_df.height, dtype=bool)
        if scoring is not None:
            mask &= np.array([excluded_sf not in (s or "") for s in scoring], dtype=bool)
        if sel is not None:
            mask &= np.array([(s != excluded_sf) for s in sel], dtype=bool)
        keep_mask = mask

    # ---- best_idx per target ----
    # Which criteria are present (in BEST_CRITERIA order, primary first).
    present_criteria = [
        c for c in BEST_CRITERIA
        if _pivot_path(data_dir, c['metric'], c['threshold']) is not None
    ]

    # Memory-bounded, Float64-exact lexicographic argmin via frontier narrowing.
    # We hold only ONE criterion frame at a time. For each target we keep the set
    # of candidate row indices still tied on all criteria processed so far; each
    # subsequent criterion narrows candidates to those achieving the best key
    # (inf if null else -value) within the current candidate set. This is
    # provably identical to ``np.lexsort(keys[::-1])[0]`` (primary = first
    # criterion; final ties broken by smallest original index).
    sel_np = np.float32 if selection_dtype == pl.Float32 else np.float64

    # candidates[target] = np.ndarray of candidate row indices (into kept rows).
    candidates: dict[str, np.ndarray] = {}
    for ci, criterion in enumerate(present_criteria):
        path = _pivot_path(data_dir, criterion['metric'], criterion['threshold'])
        assert path is not None  # present_criteria already filtered to existing paths
        frame = _read_columns(path, target_cols)
        mat = frame.to_numpy().astype(sel_np)
        del frame
        if keep_mask is not None:
            mat = mat[keep_mask]
        for t_idx, target in enumerate(target_cols):
            vals = mat[:, t_idx].astype(np.float64)
            key = np.where(np.isnan(vals), np.inf, -vals)
            if ci == 0:
                cand = np.flatnonzero(key == key.min())
            else:
                cand = candidates[target]
                sub = key[cand]
                cand = cand[sub == sub.min()]
            candidates[target] = cand
        del mat

    best_idx_by_target: dict[str, int] = {
        # Remaining ties broken by smallest original index (lexsort stability).
        target: int(cand.min())
        for target, cand in candidates.items()
    }
    del candidates

    # ---- median best_idx (REF@1) ----
    ref_frame = _read_columns(ref1_path, target_cols)
    if keep_mask is not None:
        # Apply filter via a polars boolean mask to keep memory low.
        ref_frame = ref_frame.filter(pl.Series(keep_mask))
    median_best_idx = _row_median_argmax(ref_frame, target_cols)
    del ref_frame

    return component_cols, target_cols, best_idx_by_target, median_best_idx, keep_mask


def _extract_lean(data_dir: str, component_cols, best_idx_by_target, median_best_idx,
                  keep_mask, metrics_thresholds):
    """Build best/median result dicts reading each needed frame once at Float64.

    ``metrics_thresholds`` is an iterable of (metric, threshold). For each frame
    we gather only the rows we need (winning rows for best + the single median
    row) and release the frame.
    """
    unique_best = sorted(set(best_idx_by_target.values()))
    needed = sorted(set(unique_best) | {median_best_idx})
    pos_of_idx = {idx: pos for pos, idx in enumerate(needed)}

    best_results: dict = {m: {} for m, _ in metrics_thresholds}
    median_results: dict = {m: {} for m, _ in metrics_thresholds}

    # Determine median workflow component values once (from REF@1).
    ref1_path = _pivot_path(data_dir, 'ref', 1.0)
    assert ref1_path is not None  # _select_indices_lean already required REF@1
    comp_full = _read_columns(ref1_path, component_cols)
    if keep_mask is not None:
        comp_full = comp_full.filter(pl.Series(keep_mask))
    median_comp_row = _gather_rows(comp_full, [median_best_idx])
    best_workflow = {comp: median_comp_row[comp][0] for comp in component_cols}
    del comp_full

    for metric, threshold in metrics_thresholds:
        path = _pivot_path(data_dir, metric, threshold)
        if path is None:
            continue
        all_cols = _all_columns(path)
        target_cols = [c for c in all_cols if c not in component_cols]
        mcol = _metric_col(metric)

        if keep_mask is not None:
            frame = _read_columns(path, None).filter(pl.Series(keep_mask))
        else:
            frame = _read_columns(path, None)
        gathered = _gather_rows(frame, needed)
        del frame

        comp_arrays = {c: gathered[c].to_list() for c in component_cols}

        # best
        best_rows = []
        for target, best_idx in best_idx_by_target.items():
            if target not in target_cols:
                continue
            pos = pos_of_idx[best_idx]
            tv = gathered[target][pos]
            if tv is None:
                continue
            r = {'Target': target, 'Model': 'DockM8-best', 'Percentile': threshold, mcol: tv}
            for comp in component_cols:
                r[comp] = comp_arrays[comp][pos]
            best_rows.append(r)
        if best_rows:
            best_results[metric][threshold] = pl.DataFrame(best_rows)

        # median
        mpos = pos_of_idx[median_best_idx]
        median_rows = []
        for target in target_cols:
            tv = gathered[target][mpos]
            if tv is not None:
                r = {'Target': target, 'Model': 'DockM8-median', 'Percentile': threshold, mcol: tv}
                for comp, val in best_workflow.items():
                    r[comp] = val
                median_rows.append(r)
        if median_rows:
            median_results[metric][threshold] = pl.DataFrame(median_rows)

        del gathered

    return best_results, median_results


def _all_metric_thresholds() -> list:
    mts = []
    for metric, thresholds in THRESHOLD_METRICS.items():
        for threshold in thresholds:
            mts.append((metric, threshold))
    for metric in NON_THRESHOLD_METRICS:
        mts.append((metric, None))
    return mts


def _extract_custom_lean(data_dir: str, specs: list) -> list:
    """Memory-lean equivalent of looping ``extract_custom_workflow`` over specs.

    Resolves each spec to its matching workflow row index once -- matching the
    original ``df.filter(...).row(0)`` (first match, frame order, null-safe) -- via
    the cheap component columns, then reads each needed pivot ONCE at Float64 and
    gathers only the matched rows. Never holds all 14 frames at once.
    Returns ``[{'name': name, 'results': {metric: {threshold: pl.DataFrame}}}, ...]``.
    """
    mts = [mt for mt in _all_metric_thresholds() if _pivot_path(data_dir, *mt) is not None]
    metrics_seen: list[str] = []
    for metric, _ in mts:
        if metric not in metrics_seen:
            metrics_seen.append(metric)

    # Component layout/values from REF@1 (row order identical across all frames).
    ref1_path = _pivot_path(data_dir, 'ref', 1.0) or (_pivot_path(data_dir, *mts[0]) if mts else None)
    if ref1_path is None:
        return [{'name': s.get('name', f"DockM8-custom-{i + 1}"), 'results': {}}
                for i, s in enumerate(specs)]
    component_cols = [c for c in _all_columns(ref1_path) if c in WORKFLOW_COMPONENTS]
    comp_df = _read_columns(ref1_path, component_cols)
    idx_df = comp_df.with_row_index(name='_ridx')

    spec_rows: list[tuple] = []  # (name, matched_row_index_or_None)
    for i, spec in enumerate(specs):
        cname = spec.get('name', f"DockM8-custom-{i + 1}")
        filters = [
            pl.col("docking") == spec['docking'],
            pl.col("selection_method") == spec['selection_method'],
            pl.col("scoring") == spec['scoring'],
        ]
        if spec.get('consensus_method') is not None:
            filters.append(pl.col("consensus_method") == spec['consensus_method'])
        m = idx_df.filter(pl.all_horizontal(filters))
        spec_rows.append((cname, int(m['_ridx'][0]) if m.height else None))

    results_by_name: dict = {name: {mc: {} for mc in metrics_seen} for name, _ in spec_rows}
    needed = sorted({r for _, r in spec_rows if r is not None})
    if not needed:
        return [{'name': name, 'results': results_by_name[name]} for name, _ in spec_rows]
    pos_of_idx = {idx: pos for pos, idx in enumerate(needed)}
    comp_gathered = _gather_rows(comp_df, needed)
    del comp_df, idx_df

    for metric, threshold in mts:
        path = _pivot_path(data_dir, metric, threshold)
        if path is None:
            continue
        target_cols = [c for c in _all_columns(path) if c not in component_cols]
        mcol = _metric_col(metric)
        frame = _read_columns(path, None)
        gathered = _gather_rows(frame, needed)
        del frame
        for name, row_idx in spec_rows:
            if row_idx is None:
                continue
            pos = pos_of_idx[row_idx]
            rows = []
            for target in target_cols:
                tv = gathered[target][pos]
                if tv is None:
                    continue
                r = {'Target': target, 'Model': name, 'Percentile': threshold, mcol: tv}
                for comp in component_cols:
                    r[comp] = comp_gathered[comp][pos]
                rows.append(r)
            if rows:
                results_by_name[name][metric][threshold] = pl.DataFrame(rows)
        del gathered

    return [{'name': name, 'results': results_by_name[name]} for name, _ in spec_rows]


# =============================================================================
# Memory-lean entrypoint
# =============================================================================

def run_analysis_fast(data_dir: str, custom_workflow_specs: list | None,
                      output_dir: str, output_filename: str,
                      excluded_sf: str | None = None,
                      selection_dtype: pl.DataType = pl.Float64) -> str:
    """Literature-comparison extraction for one dataset, memory-lean.

    Selection reads only the criteria/REF@1 columns; extraction reads each
    needed parquet once at Float64 and gathers only the winning rows. Custom
    workflows reuse the original ``extract_custom_workflow`` (loads frames on
    demand below).
    """
    mts = [mt for mt in _all_metric_thresholds() if _pivot_path(data_dir, *mt) is not None]

    component_cols, _, best_idx_by_target, median_best_idx, keep_mask = _select_indices_lean(
        data_dir, excluded_sf=excluded_sf, selection_dtype=selection_dtype
    )
    best, median = _extract_lean(
        data_dir, component_cols, best_idx_by_target, median_best_idx, keep_mask, mts
    )

    suffix = ""
    if excluded_sf is not None:
        suffix = f"-no{excluded_sf.replace('-', '')}"
    all_results = [{'suffix': suffix, 'best': best, 'median': median}]

    # Custom workflows are only requested when excluded_sf is None (the noRFScoreVS
    # variant passes no custom specs), so the lean extractor never needs to filter.
    custom_workflows = _extract_custom_lean(data_dir, custom_workflow_specs) if custom_workflow_specs else []

    return export_combined_results(all_results, custom_workflows or None, output_dir, output_filename)


def run_analysis_filtered_fast(data_dir: str, output_dir: str, output_filename: str,
                               excluded_sf: str = "RFScoreVS",
                               selection_dtype: pl.DataType = pl.Float64) -> str:
    """NoRFScoreVS DUD-E variant, memory-lean (no custom workflows)."""
    return run_analysis_fast(
        data_dir, custom_workflow_specs=None, output_dir=output_dir,
        output_filename=output_filename, excluded_sf=excluded_sf,
        selection_dtype=selection_dtype,
    )


def run_extraction_fast(aggregated_dir: Path | None = None,
                        dockm8_results_dir: Path | None = None,
                        datasets: list[str] | None = None,
                        selection_dtype: pl.DataType = pl.Float64) -> None:
    """Memory-lean mirror of ``data_extraction.run_extraction``.

    NEVER holds all 14 Float64 frames at once for the best/median selection.
    The internal (per-SF) extraction reuses the original on-demand path.
    """
    if aggregated_dir is None:
        aggregated_dir = get_aggregated_dir()
    if dockm8_results_dir is None:
        dockm8_results_dir = get_dockm8_results_dir()
    if datasets is None:
        datasets = ["DEKOIS", "DUD-E", "Lit-PCBA"]

    aggregated_dir = Path(aggregated_dir)
    dockm8_results_dir = Path(dockm8_results_dir)
    dockm8_results_dir.mkdir(parents=True, exist_ok=True)
    output_dir_str = str(dockm8_results_dir)

    for dataset in datasets:
        agg_subdir = str(aggregated_dir / DATASET_AGGREGATED_DIRS[dataset])
        ds_lower = dataset.lower()

        print(f"\n--- Literature extraction (fast): {dataset} ---")
        out_name = f"{ds_lower}_dockm8_results.csv" if dataset == "Lit-PCBA" else f"{dataset}_dockm8_results.csv"
        run_analysis_fast(
            data_dir=agg_subdir,
            custom_workflow_specs=CUSTOM_WORKFLOWS_RESULTS[dataset],
            output_dir=output_dir_str,
            output_filename=out_name,
            selection_dtype=selection_dtype,
        )

        if dataset == "DUD-E":
            print(f"\n--- Literature extraction (fast): {dataset} (no RFScoreVS) ---")
            run_analysis_filtered_fast(
                data_dir=agg_subdir,
                output_dir=output_dir_str,
                output_filename="DUD-E_dockm8_results_noRFScoreVS.csv",
                excluded_sf="RFScoreVS",
                selection_dtype=selection_dtype,
            )

        print(f"\n--- Internal extraction (fast): {dataset} ---")
        sfs = INTERNAL_SCORING_FUNCTIONS[dataset]
        docking_methods = INTERNAL_DOCKING_METHODS[dataset]
        internal_workflows = [
            {'name': f'{sf}/{docking}@DockM8', 'docking': docking, 'selection_method': sf, 'scoring': sf}
            for sf in sfs for docking in docking_methods
        ]
        int_name = "lit-pcba_dockm8_internal.csv" if dataset == "Lit-PCBA" else f"{dataset}_dockm8_internal.csv"
        export_combined_results(
            [], _extract_custom_lean(agg_subdir, internal_workflows), output_dir_str, int_name,
        )

    print("\n--- Generating best/worst summary ---")
    de.generate_best_worst(output_dir_str)
    print("\nExtraction complete.")


if __name__ == '__main__':
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()],
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregated-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--datasets', type=str, default=None)
    args = parser.parse_args()

    from .config import DATASETS, parse_list_arg
    ds = parse_list_arg(args.datasets, DATASETS) if args.datasets else None
    run_extraction_fast(
        aggregated_dir=args.aggregated_dir,
        dockm8_results_dir=args.output_dir,
        datasets=ds,
    )

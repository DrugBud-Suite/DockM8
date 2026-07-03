#!/usr/bin/env python3
"""
DockM8 Data Extraction

Converts aggregated pivot tables (parquet/CSV) into the long-format multi-metric
CSVs consumed by the boxplot scripts.

Outputs written to results/output/dockm8_results/:
  - {DATASET}_dockm8_results.csv        (literature comparison, all SFs)
  - DUD-E_dockm8_results_noRFScoreVS.csv (DUD-E without RFScoreVS)
  - {DATASET}_dockm8_internal.csv        (individual SF × docking breakdowns)
  - best_worst_analysis.csv              (summary stats across all result CSVs)

Usage (via run_all.py):
    python -m analysis.run_all extract

Direct usage:
    python -m analysis.data_extraction --aggregated-dir results/output/aggregated
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

from .config import get_aggregated_dir, get_dockm8_results_dir

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('DockM8Extraction')

# =============================================================================
# CONFIGURATION
# =============================================================================

THRESHOLD_METRICS = {
    "ef":  [0.1, 0.5, 1, 5],
    "ref": [0.1, 0.5, 1, 5],
    "pm":  [0.1, 0.5, 1, 5],
}
NON_THRESHOLD_METRICS = ["auc_roc", "bedroc"]
METRIC_FILE_MAP = {
    "ef":     "combined_results_ef_pivot_thresh{thresh}.csv",
    "ref":    "combined_results_ref_pivot_thresh{thresh}.csv",
    "pm":     "combined_results_pm_pivot_thresh{thresh}.csv",
    "auc_roc": "combined_results_auc_roc_pivot_thresh0p1.csv",
    "bedroc": "combined_results_bedroc_pivot_thresh0p1.csv",
}
WORKFLOW_COMPONENTS = ["docking", "scoring", "consensus_method", "selection_method"]

BEST_CRITERIA = [
    {'metric': 'ref',     'threshold': 1.0,  'name': 'ref_1p0'},
    {'metric': 'ref',     'threshold': 0.1,  'name': 'ref_0p1'},
    {'metric': 'pm',      'threshold': 1.0,  'name': 'pm_1p0'},
    {'metric': 'bedroc',  'threshold': None, 'name': 'bedroc'},
    {'metric': 'auc_roc', 'threshold': None, 'name': 'auc_roc'},
]

DATASET_AGGREGATED_DIRS = {
    "DEKOIS":   "DEKOIS_aggregated_results",
    "DUD-E":    "DUD-E_aggregated_results",
    "Lit-PCBA": "Lit-PCBA_aggregated_results",
}

CUSTOM_WORKFLOWS_RESULTS = {
    "Lit-PCBA": [
        {'name': 'GNINA@DockM8',    'docking': 'gnina',  'selection_method': 'CNN-Score',       'scoring': 'CNN-Score'},
        {'name': 'GenScore@DockM8', 'docking': 'gnina',  'selection_method': 'GenScore-scoring', 'scoring': 'GenScore-scoring'},
        {'name': 'PLANTS@DockM8',   'docking': 'plants', 'selection_method': 'CHEMPLP',          'scoring': 'CHEMPLP'},
        {'name': 'ConvexPLR@DockM8','docking': 'gnina',  'selection_method': 'ConvexPLR',        'scoring': 'ConvexPLR'},
        {'name': 'RFScoreVS@DockM8','docking': 'gnina',  'selection_method': 'RFScoreVS',        'scoring': 'RFScoreVS'},
        {'name': 'KORP-PL@DockM8',  'docking': 'gnina',  'selection_method': 'KORP-PL',          'scoring': 'KORP-PL'},
        {'name': 'RTMScore@DockM8',          'docking': 'gnina', 'selection_method': 'RTMScore',          'scoring': 'RTMScore'},
        {'name': 'GenScore-docking@DockM8',  'docking': 'gnina', 'selection_method': 'GenScore-docking',  'scoring': 'GenScore-docking'},
        {'name': 'GenScore-balanced@DockM8', 'docking': 'gnina', 'selection_method': 'GenScore-balanced', 'scoring': 'GenScore-balanced'},
    ],
    "DUD-E": [
        {'name': 'GNINA@DockM8',    'docking': 'gnina',  'selection_method': 'CNN-Score',       'scoring': 'CNN-Score'},
        {'name': 'GenScore@DockM8', 'docking': 'gnina',  'selection_method': 'GenScore-scoring', 'scoring': 'GenScore-scoring'},
        {'name': 'PLANTS@DockM8',   'docking': 'plants', 'selection_method': 'CHEMPLP',          'scoring': 'CHEMPLP'},
        {'name': 'ConvexPLR@DockM8','docking': 'gnina',  'selection_method': 'ConvexPLR',        'scoring': 'ConvexPLR'},
        {'name': 'RFScoreVS@DockM8','docking': 'gnina',  'selection_method': 'RFScoreVS',        'scoring': 'RFScoreVS'},
        {'name': 'KORP-PL@DockM8',  'docking': 'gnina',  'selection_method': 'KORP-PL',          'scoring': 'KORP-PL'},
        {'name': 'RTMScore@DockM8',          'docking': 'gnina', 'selection_method': 'RTMScore',          'scoring': 'RTMScore'},
        {'name': 'GenScore-docking@DockM8',  'docking': 'gnina', 'selection_method': 'GenScore-docking',  'scoring': 'GenScore-docking'},
        {'name': 'GenScore-balanced@DockM8', 'docking': 'gnina', 'selection_method': 'GenScore-balanced', 'scoring': 'GenScore-balanced'},
    ],
    "DEKOIS": [
        {'name': 'GNINA@DockM8',    'docking': 'gnina',  'selection_method': 'CNN-Score',       'scoring': 'CNN-Score'},
        {'name': 'GenScore@DockM8', 'docking': 'gnina',  'selection_method': 'GenScore-scoring', 'scoring': 'GenScore-scoring'},
        {'name': 'PLANTS@DockM8',   'docking': 'plants', 'selection_method': 'CHEMPLP',          'scoring': 'CHEMPLP'},
        {'name': 'ConvexPLR@DockM8','docking': 'gnina',  'selection_method': 'ConvexPLR',        'scoring': 'ConvexPLR'},
        {'name': 'RFScoreVS@DockM8','docking': 'gnina',  'selection_method': 'RFScoreVS',        'scoring': 'RFScoreVS'},
        {'name': 'KORP-PL@DockM8',  'docking': 'gnina',  'selection_method': 'KORP-PL',          'scoring': 'KORP-PL'},
        {'name': 'RTMScore@DockM8',          'docking': 'gnina', 'selection_method': 'RTMScore',          'scoring': 'RTMScore'},
        {'name': 'GenScore-docking@DockM8',  'docking': 'gnina', 'selection_method': 'GenScore-docking',  'scoring': 'GenScore-docking'},
        {'name': 'GenScore-balanced@DockM8', 'docking': 'gnina', 'selection_method': 'GenScore-balanced', 'scoring': 'GenScore-balanced'},
    ],
}

INTERNAL_SCORING_FUNCTIONS = {
    "Lit-PCBA": [
        'CNN-Score', 'CNN-Affinity', 'GNINA-Affinity', 'KORP-PL',
        'ConvexPLR', 'RFScoreVS', 'AD4', 'GenScore-scoring',
        'GenScore-balanced', 'Vinardo', 'CHEMPLP', 'LinF9', 'NNScore',
    ],
    "DUD-E": [
        'CNN-Score', 'CNN-Affinity', 'GNINA-Affinity', 'KORP-PL',
        'ConvexPLR', 'RFScoreVS', 'AD4', 'GenScore-scoring',
        'GenScore-balanced', 'Vinardo', 'CHEMPLP', 'LinF9',
        'NNScore', 'RTMScore',
    ],
    "DEKOIS": [
        'CNN-Score', 'CNN-Affinity', 'GNINA-Affinity', 'KORP-PL',
        'ConvexPLR', 'RFScoreVS', 'AD4', 'GenScore-scoring',
        'GenScore-balanced', 'Vinardo', 'CHEMPLP', 'LinF9',
        'NNScore', 'GenScore-docking', 'RTMScore',
    ],
}

INTERNAL_DOCKING_METHODS = {
    "Lit-PCBA": ['gnina', 'smina', 'plants'],
    "DUD-E":    ['gnina', 'smina', 'plants'],
    "DEKOIS":   ['gnina', 'smina', 'plants', 'qvina2', 'qvinaw'],
}

PERFORMANCE_METRICS_BW = [
    'EF_lit', 'NEF_lit', 'EF_calc', 'NEF_calc', 'PM', 'ROCE',
    'CCR', 'REF', 'MCC', 'CKC', 'RIE', 'AUC_ROC', 'BEDROC',
]

# =============================================================================
# CORE EXTRACTION FUNCTIONS (from #ARCHIVE/dockm8_data_extraction.ipynb)
# =============================================================================

def _read_pivot(csv_path: str):
    """Load an aggregated pivot, preferring the parquet sibling (CSV is optional now).

    The aggregation step writes parquet by default and only writes the large CSVs
    on demand, so prefer ``<name>.parquet`` and fall back to ``<name>.csv``.
    """
    pq_path = csv_path[:-4] + ".parquet" if csv_path.endswith(".csv") else csv_path + ".parquet"
    if os.path.exists(pq_path):
        return pl.read_parquet(pq_path), pq_path
    if os.path.exists(csv_path):
        return pl.read_csv(csv_path), csv_path
    return None, pq_path


def load_dockm8_data(data_dir: str) -> Dict:
    """Load aggregated pivot tables (parquet preferred, CSV fallback) for all metrics."""
    logger.info(f"Loading data from {data_dir}")
    result_data: Dict = {}

    for metric, thresholds in THRESHOLD_METRICS.items():
        result_data[metric] = {}
        for threshold in thresholds:
            thresh_str = str(threshold).replace('.', 'p') if isinstance(threshold, float) else str(threshold)
            csv_path = os.path.join(data_dir, METRIC_FILE_MAP[metric].format(thresh=thresh_str))
            try:
                df, used = _read_pivot(csv_path)
                if df is not None:
                    result_data[metric][threshold] = df
                    logger.info(f"  Loaded {metric}@{threshold}: {df.height} rows from {os.path.basename(used)}")
                else:
                    logger.warning(f"  Not found: {used} (or .csv)")
            except Exception as e:
                logger.error(f"  Error loading {csv_path}: {e}")

    for metric in NON_THRESHOLD_METRICS:
        result_data[metric] = {}
        csv_path = os.path.join(data_dir, METRIC_FILE_MAP[metric])
        try:
            df, used = _read_pivot(csv_path)
            if df is not None:
                result_data[metric][None] = df
                logger.info(f"  Loaded {metric}: {df.height} rows from {os.path.basename(used)}")
            else:
                logger.warning(f"  Not found: {used} (or .csv)")
        except Exception as e:
            logger.error(f"  Error loading {csv_path}: {e}")

    return result_data


def filter_dockm8_data(dockm8_data: Dict, excluded_sf: str) -> Dict:
    """Return a copy of dockm8_data with workflows containing excluded_sf removed."""
    logger.info(f"Filtering out '{excluded_sf}'")
    filtered: Dict = {}

    for metric, threshold_dfs in dockm8_data.items():
        filtered[metric] = {}
        for threshold, df in threshold_dfs.items():
            if df is None or df.height == 0:
                filtered[metric][threshold] = df
                continue
            cur = df.clone()
            if "scoring" in cur.columns:
                if cur["scoring"].dtype != pl.Utf8:
                    cur = cur.with_columns(pl.col("scoring").cast(pl.Utf8))
                cur = cur.filter(~pl.col("scoring").fill_null("").str.contains(excluded_sf))
            if "selection_method" in cur.columns:
                if cur["selection_method"].dtype != pl.Utf8:
                    cur = cur.with_columns(pl.col("selection_method").cast(pl.Utf8))
                cur = cur.filter(
                    (pl.col("selection_method") != excluded_sf) | pl.col("selection_method").is_null()
                )
            filtered[metric][threshold] = cur

    return filtered


def analyze_dockm8_best(dockm8_data: Dict, criteria: Optional[List] = None) -> Dict:
    """Find the best workflow per target using the given criteria."""
    logger.info("Running DockM8-best analysis")
    if not criteria:
        criteria = BEST_CRITERIA

    component_cols: List[str] = []
    for metric_data in dockm8_data.values():
        for df in metric_data.values():
            if df is not None and df.height > 0:
                component_cols = [c for c in df.columns if c in WORKFLOW_COMPONENTS]
                break
        if component_cols:
            break

    all_targets: set = set()
    for thresh_dfs in dockm8_data.values():
        for df in thresh_dfs.values():
            if df is not None and df.height > 0:
                all_targets.update([c for c in df.columns if c not in component_cols])

    best_workflows_by_target: Dict = {}
    for target in all_targets:
        criteria_scores = None
        for criterion in criteria:
            metric, threshold, crit_name = criterion['metric'], criterion['threshold'], criterion['name']
            if metric not in dockm8_data or threshold not in dockm8_data[metric]:
                continue
            df = dockm8_data[metric][threshold]
            if df is None or df.height == 0 or target not in df.columns:
                continue
            wt = df.select(component_cols + [target]).rename({target: crit_name})
            if criteria_scores is None:
                criteria_scores = wt
            else:
                criteria_scores = criteria_scores.join(wt.select(component_cols + [crit_name]), on=component_cols, how='left')

        if criteria_scores is None:
            continue

        sort_exprs = []
        for criterion in criteria:
            cn = criterion['name']
            if cn in criteria_scores.columns:
                criteria_scores = criteria_scores.with_columns(
                    pl.when(pl.col(cn).is_null()).then(float('inf')).otherwise(-1 * pl.col(cn)).alias(f"{cn}_sort")
                )
                sort_exprs.append(f"{cn}_sort")

        if sort_exprs:
            criteria_scores = criteria_scores.sort(sort_exprs)
            best_workflows_by_target[target] = criteria_scores.slice(0, 1).select(component_cols)

    best_results: Dict = {}
    for metric, threshold_dfs in dockm8_data.items():
        best_results[metric] = {}
        for threshold, df in threshold_dfs.items():
            if df is None or df.height == 0:
                continue
            metric_col = {'ef': 'EF', 'ref': 'REF', 'pm': 'PM', 'auc_roc': 'AUC_ROC', 'bedroc': 'BEDROC'}.get(metric.lower(), metric.upper())
            result_rows = []
            for target, bw_df in best_workflows_by_target.items():
                if target not in df.columns:
                    continue
                matched = df.join(bw_df, on=component_cols, how='inner').select([target] + component_cols)
                if matched.height == 0:
                    continue
                row = matched.row(0)
                target_val = row[0]
                if target_val is not None:
                    r = {'Target': target, 'Model': 'DockM8-best', 'Percentile': threshold, metric_col: target_val}
                    for i, comp in enumerate(component_cols):
                        r[comp] = row[i + 1]
                    result_rows.append(r)
            if result_rows:
                best_results[metric][threshold] = pl.DataFrame(result_rows)

    return best_results


def analyze_dockm8_median(dockm8_data: Dict) -> Dict:
    """Find the workflow with the highest median REF@1% and return its performance."""
    logger.info("Running DockM8-median analysis")
    median_results: Dict = {}

    if 'ref' not in dockm8_data or 1.0 not in dockm8_data['ref'] or dockm8_data['ref'][1.0] is None:
        logger.error("REF@1% not found")
        return median_results

    ref_df = dockm8_data['ref'][1.0].clone()
    component_cols = [c for c in ref_df.columns if c in WORKFLOW_COMPONENTS]
    target_cols = [c for c in ref_df.columns if c not in component_cols]

    median_values = []
    for row_idx in range(ref_df.height):
        row = ref_df.row(row_idx)
        vals = [row[ref_df.columns.index(t)] for t in target_cols if row[ref_df.columns.index(t)] is not None]
        median_values.append(float(np.median(vals)) if vals else None)

    best_idx = max(range(len(median_values)), key=lambda i: median_values[i] if median_values[i] is not None else float('-inf'))
    best_workflow = {comp: ref_df.row(best_idx)[ref_df.columns.index(comp)] for comp in component_cols}
    logger.info(f"Best median workflow: {best_workflow}")

    for metric, threshold_dfs in dockm8_data.items():
        median_results[metric] = {}
        metric_col = {'ef': 'EF', 'ref': 'REF', 'pm': 'PM', 'auc_roc': 'AUC_ROC', 'bedroc': 'BEDROC'}.get(metric.lower(), metric.upper())
        for threshold, df in threshold_dfs.items():
            if df is None or df.height == 0:
                continue
            df_target_cols = [c for c in df.columns if c not in component_cols]
            filters = [
                (pl.col(comp) == val) if val is not None else pl.col(comp).is_null()
                for comp, val in best_workflow.items()
            ]
            matched = df.filter(pl.all_horizontal(filters))
            if matched.height == 0:
                continue
            best_row = matched.row(0)
            result_rows = []
            for target in df_target_cols:
                tidx = df.columns.index(target)
                tv = best_row[tidx]
                if tv is not None:
                    r = {'Target': target, 'Model': 'DockM8-median', 'Percentile': threshold, metric_col: tv}
                    for comp, val in best_workflow.items():
                        r[comp] = val
                    result_rows.append(r)
            if result_rows:
                median_results[metric][threshold] = pl.DataFrame(result_rows)

    return median_results


def extract_custom_workflow(dockm8_data: Dict, docking: str, selection_method: str,
                            scoring: str, consensus_method=None, custom_name: str = "DockM8-custom") -> Dict:
    """Extract per-target results for a specific workflow combination."""
    custom_results: Dict = {}
    for metric, threshold_dfs in dockm8_data.items():
        custom_results[metric] = {}
        metric_col = {'ef': 'EF', 'ref': 'REF', 'pm': 'PM', 'auc_roc': 'AUC_ROC', 'bedroc': 'BEDROC'}.get(metric.lower(), metric.upper())
        for threshold, df in threshold_dfs.items():
            if df is None or df.height == 0:
                continue
            component_cols = [c for c in df.columns if c in WORKFLOW_COMPONENTS]
            target_cols = [c for c in df.columns if c not in component_cols]
            filters = [
                pl.col("docking") == docking,
                pl.col("selection_method") == selection_method,
                pl.col("scoring") == scoring,
            ]
            if consensus_method is not None:
                filters.append(pl.col("consensus_method") == consensus_method)
            matched = df.filter(pl.all_horizontal(filters))
            if matched.height == 0:
                continue
            best_row = matched.row(0)
            result_rows = []
            for target in target_cols:
                tidx = matched.columns.index(target)
                tv = best_row[tidx]
                if tv is not None:
                    r = {'Target': target, 'Model': custom_name, 'Percentile': threshold, metric_col: tv}
                    for comp in component_cols:
                        r[comp] = best_row[matched.columns.index(comp)]
                    result_rows.append(r)
            if result_rows:
                custom_results[metric][threshold] = pl.DataFrame(result_rows)

    return custom_results


def export_combined_results(all_results: List, custom_workflows: Optional[List],
                            output_dir: str, output_filename: str) -> str:
    """Export consolidated results to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    output_columns = [
        'Target', 'Model', 'Percentile', 'Factor_Type',
        'EF_lit', 'NEF_lit', 'EF_calc', 'NEF_calc',
        'n', 'N', 'Ns', 'ns', 'PM', 'ROCE', 'CCR', 'REF', 'MCC', 'CKC', 'RIE', 'AUC_ROC', 'BEDROC',
    ] + WORKFLOW_COMPONENTS

    result_sets = []
    for rg in all_results:
        suffix = rg['suffix']
        result_sets.append((f"DockM8-best{suffix}", rg['best'], suffix))
        result_sets.append((f"DockM8-median{suffix}", rg['median'], suffix))
    if custom_workflows:
        for cw in custom_workflows:
            if 'name' in cw and 'results' in cw:
                result_sets.append((cw['name'], cw['results'], ""))

    ti_values: Dict = {}
    for model_name, result_set, suffix in result_sets:
        for metric in ['auc_roc', 'bedroc']:
            if metric in result_set and None in result_set[metric]:
                df = result_set[metric][None]
                if df is not None and df.height > 0:
                    for row_dict in df.to_dicts():
                        target = row_dict['Target']
                        key = (target, model_name, suffix)
                        if key not in ti_values:
                            ti_values[key] = {}
                        mc = 'AUC_ROC' if metric == 'auc_roc' else 'BEDROC'
                        if mc in row_dict:
                            ti_values[key][mc] = row_dict[mc]
                        for comp in WORKFLOW_COMPONENTS:
                            if comp in row_dict:
                                ti_values[key][comp] = row_dict[comp]

    consolidated: Dict = {}
    for model_name, result_set, suffix in result_sets:
        for metric, threshold_dfs in result_set.items():
            if metric in ['auc_roc', 'bedroc']:
                continue
            for threshold, df in threshold_dfs.items():
                if df is None or df.height == 0 or threshold is None:
                    continue
                out_col = {'ef': 'NEF_calc', 'ref': 'REF', 'pm': 'PM'}.get(metric.lower(), metric.upper())
                for row_dict in df.to_dicts():
                    target = row_dict['Target']
                    row_model = row_dict.get('Model', model_name)
                    in_col = next((c for c in row_dict if c.upper() == metric.upper()), None)
                    if in_col is None or row_dict[in_col] is None:
                        continue
                    key = (target, row_model, suffix, threshold)
                    if key not in consolidated:
                        consolidated[key] = {'Target': target, 'Model': row_model, 'Percentile': threshold, 'Factor_Type': 'NEF'}
                        for comp in WORKFLOW_COMPONENTS:
                            if comp in row_dict:
                                consolidated[key][comp] = row_dict[comp]
                        ti_key = (target, row_model, suffix)
                        if ti_key in ti_values:
                            for ti_col, ti_val in ti_values[ti_key].items():
                                if ti_col not in WORKFLOW_COMPONENTS:
                                    consolidated[key][ti_col] = ti_val
                    consolidated[key][out_col] = row_dict[in_col]

    if not consolidated:
        logger.error("No results to export")
        raise ValueError("No results to export")

    final_df = pl.DataFrame([dict(v) for v in consolidated.values()])
    for col in output_columns:
        if col not in final_df.columns:
            final_df = final_df.with_columns(pl.lit(None).alias(col))
    final_cols = [c for c in output_columns if c in final_df.columns]
    final_df = final_df.select(final_cols).unique()
    sort_cols = [c for c in ['Target', 'Model', 'Percentile'] if c in final_df.columns]
    if sort_cols:
        final_df = final_df.sort(sort_cols)

    output_path = os.path.join(output_dir, output_filename)
    final_df.write_csv(output_path)
    logger.info(f"Exported {final_df.height} rows to {output_path}")
    return output_path


def run_analysis(data_dir: str, custom_workflow_specs: Optional[List],
                 output_dir: str, output_filename: str,
                 best_criteria: Optional[List] = None) -> str:
    """Run literature-comparison extraction for one dataset."""
    dockm8_data = load_dockm8_data(data_dir)
    best = analyze_dockm8_best(dockm8_data, criteria=best_criteria or BEST_CRITERIA)
    median = analyze_dockm8_median(dockm8_data)
    all_results = [{'suffix': '', 'best': best, 'median': median}]

    custom_workflows = []
    if custom_workflow_specs:
        for i, spec in enumerate(custom_workflow_specs):
            cname = spec.get('name', f"DockM8-custom-{i+1}")
            custom_workflows.append({
                'name': cname,
                'results': extract_custom_workflow(
                    dockm8_data,
                    docking=spec['docking'],
                    selection_method=spec['selection_method'],
                    scoring=spec['scoring'],
                    consensus_method=spec.get('consensus_method'),
                    custom_name=cname,
                )
            })

    return export_combined_results(all_results, custom_workflows, output_dir, output_filename)


def run_analysis_filtered(data_dir: str, output_dir: str, output_filename: str,
                          best_criteria: Optional[List] = None,
                          excluded_sf: str = "RFScoreVS") -> str:
    """Run extraction excluding a specific scoring function."""
    dockm8_data = load_dockm8_data(data_dir)
    filtered = filter_dockm8_data(dockm8_data, excluded_sf)
    best = analyze_dockm8_best(filtered, criteria=best_criteria or BEST_CRITERIA)
    median = analyze_dockm8_median(filtered)
    suffix = f"-no{excluded_sf.replace('-', '')}"
    all_results = [{'suffix': suffix, 'best': best, 'median': median}]
    return export_combined_results(all_results, None, output_dir, output_filename)


def run_analysis_onlycustom(data_dir: str, custom_workflow_specs: List,
                             output_dir: str, output_filename: str) -> str:
    """Run extraction for individual SF × docking combinations (internal comparison)."""
    dockm8_data = load_dockm8_data(data_dir)
    custom_workflows = []
    for i, spec in enumerate(custom_workflow_specs):
        cname = spec.get('name', f"DockM8-custom-{i+1}")
        custom_workflows.append({
            'name': cname,
            'results': extract_custom_workflow(
                dockm8_data,
                docking=spec['docking'],
                selection_method=spec['selection_method'],
                scoring=spec['scoring'],
                consensus_method=spec.get('consensus_method'),
                custom_name=cname,
            )
        })
    return export_combined_results([], custom_workflows, output_dir, output_filename)


# =============================================================================
# BEST-WORST SUMMARY
# =============================================================================

def generate_best_worst(dockm8_results_dir: str) -> str:
    """Compute summary statistics across all dockm8_results CSVs."""
    cols_to_ignore = ['docking', 'scoring', 'consensus_method', 'selection_method', 'n', 'N', 'Ns', 'ns', 'Factor_Type']
    output_path = os.path.join(dockm8_results_dir, 'best_worst_analysis.csv')
    file_paths = [
        os.path.join(dockm8_results_dir, 'DEKOIS_dockm8_results.csv'),
        os.path.join(dockm8_results_dir, 'DUD-E_dockm8_results.csv'),
        os.path.join(dockm8_results_dir, 'lit-pcba_dockm8_results.csv'),
        os.path.join(dockm8_results_dir, 'DUD-E_dockm8_results_noRFScoreVS.csv'),
    ]

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Combined Analysis Report\n" + "="*80 + "\n\n")

    for file_path in file_paths:
        source_name = os.path.basename(file_path)
        if not os.path.exists(file_path):
            logger.warning(f"Not found, skipping: {file_path}")
            with open(output_path, 'a') as f:
                f.write(f"--- Skipped (not found): {source_name} ---\n\n")
            continue

        df_raw = pd.read_csv(file_path)
        original_cols = df_raw.columns.tolist()
        df = df_raw.drop(columns=[c for c in cols_to_ignore if c in df_raw.columns], errors='ignore')

        if 'Percentile' in df.columns:
            df['Percentile'] = pd.to_numeric(df['Percentile'], errors='coerce')
        valid_metrics = [m for m in PERFORMANCE_METRICS_BW if m in df.columns and
                         pd.api.types.is_numeric_dtype(pd.to_numeric(df[m], errors='coerce')) and
                         not pd.to_numeric(df[m], errors='coerce').isnull().all()]

        if not valid_metrics or 'Model' not in df.columns or 'Percentile' not in df.columns:
            with open(output_path, 'a') as f:
                f.write(f"--- Skipped (missing columns): {source_name} ---\n\n")
            continue

        for m in valid_metrics:
            df[m] = pd.to_numeric(df[m], errors='coerce')

        stats = df.groupby(['Model', 'Percentile'])[valid_metrics].agg(['mean', 'median']).round(4)
        bw_rows = []
        for (model_name, pct_val) in stats.index:
            sub = df[(df['Model'] == model_name) & (df['Percentile'] == pct_val)]
            for metric in valid_metrics:
                if sub[metric].isnull().all():
                    continue
                for fn, label in [('idxmax', 'Best (Highest)'), ('idxmin', 'Worst (Lowest)')]:
                    try:
                        idx = getattr(sub[metric], fn)()
                        entry = df_raw.loc[idx].copy()
                        entry['analysis_metric'] = metric
                        entry['entry_type'] = label
                        entry['metric_value'] = entry[metric]
                        entry['Percentile_Analyzed'] = pct_val
                        bw_rows.append(entry)
                    except ValueError:
                        pass

        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(f"--- {source_name} ---\n\n")
            f.write("Model & Percentile Statistics\n")
            stats.to_csv(f, mode='a', header=True)
            f.write("\n\nBest/Worst Entries\n")
            if bw_rows:
                bw_df = pd.DataFrame(bw_rows)
                id_cols = ['Model', 'Percentile_Analyzed', 'analysis_metric', 'entry_type', 'metric_value', 'Target']
                other = [c for c in original_cols if c not in id_cols]
                ordered = [c for c in id_cols + other if c in bw_df.columns]
                bw_df[ordered].sort_values(['Model', 'Percentile_Analyzed', 'analysis_metric', 'entry_type']).to_csv(f, index=False)
            else:
                f.write("(no entries)\n")
            f.write("\n\n" + "="*80 + "\n\n")

    logger.info(f"Best/worst analysis written to {output_path}")
    return output_path


# =============================================================================
# ENTRY POINT
# =============================================================================

def run_extraction(aggregated_dir: Optional[Path] = None,
                   dockm8_results_dir: Optional[Path] = None,
                   datasets: Optional[List[str]] = None) -> None:
    """
    Run full extraction pipeline for all datasets.

    Args:
        aggregated_dir: Path to aggregated pivot tables (default: results/output/aggregated/)
        dockm8_results_dir: Output path for extracted CSVs (default: results/output/dockm8_results/)
        datasets: Datasets to process (default: all three)
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
        ds_lower = dataset.lower().replace("-", "-")

        print(f"\n--- Literature extraction: {dataset} ---")
        out_name = f"{ds_lower}_dockm8_results.csv" if dataset == "Lit-PCBA" else f"{dataset}_dockm8_results.csv"
        run_analysis(
            data_dir=agg_subdir,
            custom_workflow_specs=CUSTOM_WORKFLOWS_RESULTS[dataset],
            output_dir=output_dir_str,
            output_filename=out_name,
        )

        if dataset == "DUD-E":
            print(f"\n--- Literature extraction: {dataset} (no RFScoreVS) ---")
            run_analysis_filtered(
                data_dir=agg_subdir,
                output_dir=output_dir_str,
                output_filename="DUD-E_dockm8_results_noRFScoreVS.csv",
                excluded_sf="RFScoreVS",
            )

        print(f"\n--- Internal extraction: {dataset} ---")
        sfs = INTERNAL_SCORING_FUNCTIONS[dataset]
        docking_methods = INTERNAL_DOCKING_METHODS[dataset]
        internal_workflows = [
            {'name': f'{sf}/{docking}@DockM8', 'docking': docking, 'selection_method': sf, 'scoring': sf}
            for sf in sfs for docking in docking_methods
        ]
        int_name = f"lit-pcba_dockm8_internal.csv" if dataset == "Lit-PCBA" else f"{dataset}_dockm8_internal.csv"
        run_analysis_onlycustom(
            data_dir=agg_subdir,
            custom_workflow_specs=internal_workflows,
            output_dir=output_dir_str,
            output_filename=int_name,
        )

    print("\n--- Generating best/worst summary ---")
    generate_best_worst(output_dir_str)
    print("\nExtraction complete.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregated-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--datasets', type=str, default=None)
    args = parser.parse_args()

    from .config import parse_list_arg, DATASETS
    datasets = parse_list_arg(args.datasets, DATASETS) if args.datasets else None
    run_extraction(
        aggregated_dir=args.aggregated_dir,
        dockm8_results_dir=args.output_dir,
        datasets=datasets,
    )

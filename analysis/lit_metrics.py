#!/usr/bin/env python3
"""
Literature metrics calculation module.

Calculates performance metrics from literature-reported EF/NEF values.
This module includes both the calculation functions and the CLI interface.

Usage:
    python lit_metrics.py --lit-file literature.csv --count-file counts.csv --columns "EF1,NEF1" --output results.csv
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


# =============================================================================
# METRIC CALCULATIONS
# =============================================================================

def calculate_metrics(
    target: str,
    model: str,
    percentile: float,
    factor_type: str,
    ef_lit: Optional[float],
    nef_lit: Optional[float],
    n: int,
    N: int,
    Ns: int,
    ns: float,
    auc_roc: Optional[float] = None,
    bedroc: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate all performance metrics for a given data point.

    Args:
        target: Target name
        model: Model/method name
        percentile: Threshold percentile (e.g., 1.0 for top 1%)
        factor_type: 'EF' or 'NEF' indicating source of ns calculation
        ef_lit: Literature-reported EF value
        nef_lit: Literature-reported NEF value
        n: Number of actives in dataset
        N: Total number of compounds
        Ns: Number of compounds in selection (top X%)
        ns: Number of actives in selection
        auc_roc: Literature-reported AUC-ROC (optional)
        bedroc: Literature-reported BEDROC (optional)

    Returns:
        Dictionary with all calculated metrics
    """
    ef_calc = (ns / Ns) * 100 if Ns != 0 else 0
    nef_calc = (ns * N) / (Ns * n) if (Ns * n) != 0 else 0

    numerator = ns * N - n * ns
    denominator = ns * N - 2 * n * ns + n * Ns
    pm = numerator / denominator if denominator != 0 else 0

    if n * (Ns - ns) <= 0:
        roce = float('inf')
    else:
        roce = (ns * (N - n)) / (n * (Ns - ns))

    if n == 0 or N - n == 0:
        ccr = 0
    else:
        ccr = 0.5 * ((ns / n) + ((N - Ns - n + ns) / (N - n)))

    max_possible = min(Ns, n)
    ref = 100 * (ns / max_possible) if max_possible != 0 else 0
    ref = min(ref, 100)

    numerator = N * ns - Ns * n
    denominator = np.sqrt(Ns * n * (N - n) * (N - Ns))
    mcc = numerator / denominator if denominator != 0 else 0

    numerator = N * n + N * Ns - 2 * ns * N
    denominator = N * n + N * Ns - 2 * n * Ns
    ckc = 1 - (numerator / denominator) if denominator != 0 else 0

    return {
        'Target': target,
        'Model': model,
        'Percentile': percentile,
        'Factor_Type': factor_type,
        'EF_lit': ef_lit,
        'NEF_lit': nef_lit,
        'EF_calc': ef_calc,
        'NEF_calc': nef_calc,
        'n': n,
        'N': N,
        'Ns': Ns,
        'ns': ns,
        'PM': pm,
        'ROCE': roce,
        'CCR': ccr,
        'REF': ref,
        'MCC': mcc,
        'CKC': ckc,
        'AUC_ROC': auc_roc,
        'BEDROC': bedroc
    }


# =============================================================================
# PARSING UTILITIES
# =============================================================================

def parse_ef_nef_columns(columns: str) -> Tuple[List[str], List[str]]:
    """
    Parse comma-separated EF/NEF column specification.

    Args:
        columns: Comma-separated column names (e.g., 'EF0.5,EF1,NEF1,NEF5')

    Returns:
        Tuple of (ef_cols, nef_cols) lists
    """
    ef_cols = []
    nef_cols = []

    for col in columns.split(','):
        col = col.strip()
        if col.upper().startswith('EF'):
            ef_cols.append(col)
        elif col.upper().startswith('NEF'):
            nef_cols.append(col)

    return ef_cols, nef_cols


def extract_percentile(col_name: str) -> Optional[float]:
    """
    Extract percentile value from column name like 'EF1' or 'NEF0.5'.

    Args:
        col_name: Column name containing percentile

    Returns:
        Percentile value as float, or None if parsing fails
    """
    try:
        num_str = col_name.upper().replace('NEF', '').replace('EF', '')
        return float(num_str)
    except ValueError:
        return None


# =============================================================================
# DATA PROCESSING
# =============================================================================

def process_literature_data(
    lit_file: Path,
    count_file: Path,
    ef_cols: List[str],
    nef_cols: List[str],
    target_col: str = 'Target',
    model_col: str = 'Model'
) -> pd.DataFrame:
    """
    Process literature data to calculate performance metrics.

    Args:
        lit_file: Path to literature CSV with EF/NEF values
        count_file: Path to count data CSV with actives/total counts
        ef_cols: List of EF column names to process
        nef_cols: List of NEF column names to process
        target_col: Column name for target in literature file
        model_col: Column name for model in literature file

    Returns:
        DataFrame with calculated metrics
    """
    lit_df = pd.read_csv(lit_file, encoding='utf-8')
    lit_df = lit_df.dropna(axis=1, how='all')
    lit_df[target_col] = lit_df[target_col].astype(str).str.strip()
    lit_df[model_col] = lit_df[model_col].astype(str).str.strip()

    count_df = pd.read_csv(count_file)
    count_df['target'] = count_df['target'].str.lower().str.strip()

    available_ef = [c for c in ef_cols if c in lit_df.columns]
    available_nef = [c for c in nef_cols if c in lit_df.columns]

    if not available_ef and not available_nef:
        print(f"Warning: No EF/NEF columns found in {lit_file}")
        return pd.DataFrame()

    print(f"Processing {lit_file.name}")
    print(f"  EF columns: {available_ef}")
    print(f"  NEF columns: {available_nef}")

    results = []

    for _, lit_row in lit_df.iterrows():
        target = lit_row[target_col]
        model = lit_row[model_col]

        count_row = count_df[count_df['target'] == target.lower()]
        if len(count_row) == 0:
            continue

        n = count_row['actives'].iloc[0]
        N = count_row['total'].iloc[0]

        for ef_col in available_ef:
            ef_value = lit_row.get(ef_col)
            if pd.isna(ef_value):
                continue

            percentile = extract_percentile(ef_col)
            if percentile is None:
                continue

            Ns = int(np.ceil(N * percentile / 100))
            ns = (ef_value / 100) * Ns

            result = calculate_metrics(
                target=target,
                model=model,
                percentile=percentile,
                factor_type='EF',
                ef_lit=ef_value,
                nef_lit=None,
                n=n,
                N=N,
                Ns=Ns,
                ns=ns,
                auc_roc=lit_row.get('AUC_ROC'),
                bedroc=lit_row.get('BEDROC')
            )
            results.append(result)

        for nef_col in available_nef:
            nef_value = lit_row.get(nef_col)
            if pd.isna(nef_value):
                continue

            percentile = extract_percentile(nef_col)
            if percentile is None:
                continue

            Ns = int(np.ceil(N * percentile / 100))
            ns = int(np.round((nef_value * n * Ns) / N))

            result = calculate_metrics(
                target=target,
                model=model,
                percentile=percentile,
                factor_type='NEF',
                ef_lit=None,
                nef_lit=nef_value,
                n=n,
                N=N,
                Ns=Ns,
                ns=ns,
                auc_roc=lit_row.get('AUC_ROC'),
                bedroc=lit_row.get('BEDROC')
            )
            results.append(result)

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    numeric_cols = [
        'EF_lit', 'NEF_lit', 'EF_calc', 'NEF_calc', 'ns',
        'PM', 'ROCE', 'CCR', 'REF', 'MCC', 'CKC', 'AUC_ROC', 'BEDROC'
    ]
    for col in numeric_cols:
        if col in results_df.columns:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce').round(3)

    return results_df


# =============================================================================
# CLI INTERFACE
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate performance metrics from literature EF/NEF values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python lit_metrics.py \\
    --lit-file literature.csv \\
    --count-file counts.csv \\
    --columns "EF1,NEF1" \\
    --output results.csv

  # Multiple thresholds
  python lit_metrics.py \\
    --lit-file DEKOIS_literature.csv \\
    --count-file DEKOIS_count_data.csv \\
    --columns "EF0.5,EF1,EF5,NEF0.5,NEF1,NEF5" \\
    --output dekois_metrics.csv

Input File Formats:
  Literature CSV:
    - Must have 'Target' and 'Model' columns
    - EF columns named like 'EF1', 'EF0.5', 'EF5'
    - NEF columns named like 'NEF1', 'NEF0.5'
    - Optional: 'AUC_ROC', 'BEDROC' columns

  Count CSV:
    - Must have 'target', 'actives', 'total' columns
    - Target names are matched case-insensitively
        """
    )

    parser.add_argument(
        '--lit-file',
        type=Path,
        required=True,
        help='Path to literature CSV with EF/NEF values'
    )
    parser.add_argument(
        '--count-file',
        type=Path,
        required=True,
        help='Path to count data CSV with actives/total per target'
    )
    parser.add_argument(
        '--columns',
        type=str,
        required=True,
        help='Comma-separated EF/NEF columns to process (e.g., "EF1,EF5,NEF1")'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--target-col',
        type=str,
        default='Target',
        help='Column name for target in literature file (default: Target)'
    )
    parser.add_argument(
        '--model-col',
        type=str,
        default='Model',
        help='Column name for model in literature file (default: Model)'
    )

    return parser.parse_args()


def main():
    """Main entry point for CLI."""
    args = parse_args()

    if not args.lit_file.exists():
        print(f"Error: Literature file not found: {args.lit_file}")
        sys.exit(1)
    if not args.count_file.exists():
        print(f"Error: Count file not found: {args.count_file}")
        sys.exit(1)

    ef_cols, nef_cols = parse_ef_nef_columns(args.columns)

    if not ef_cols and not nef_cols:
        print("Error: No valid EF or NEF columns specified")
        print("Columns should be like: EF1,EF0.5,NEF1,NEF5")
        sys.exit(1)

    print("=" * 60)
    print("Literature Metrics Calculator")
    print("=" * 60)
    print(f"Literature file: {args.lit_file}")
    print(f"Count file: {args.count_file}")
    print(f"EF columns: {ef_cols}")
    print(f"NEF columns: {nef_cols}")
    print()

    results_df = process_literature_data(
        lit_file=args.lit_file,
        count_file=args.count_file,
        ef_cols=ef_cols,
        nef_cols=nef_cols,
        target_col=args.target_col,
        model_col=args.model_col
    )

    if results_df.empty:
        print("Error: No results generated. Check input data and column names.")
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(args.output, index=False, encoding='utf-8')

    print(f"\nSaved {len(results_df)} rows to {args.output}")

    print("\nSummary by Percentile:")
    for pct in sorted(results_df['Percentile'].unique()):
        subset = results_df[results_df['Percentile'] == pct]
        print(f"  {pct}%: {len(subset)} entries")

    print("\n" + "=" * 60)
    print("Complete")
    print("=" * 60)


if __name__ == '__main__':
    main()

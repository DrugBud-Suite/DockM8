# Reproducing DockM8 Benchmark Results

This directory contains everything needed to reproduce the plots and analyses from the DockM8 paper: analysis scripts, the extraction utility, and committed literature baselines.

All commands are run from the **project root** (`DockM8/`, not `analysis/`).

**One workflow.** Everything runs through `python -m analysis.run_all` (aggregate → extract → plots). The aggregation is bounded-memory (streaming pivot — it will not OOM on full DEKOIS), and the extraction is the fast, memory-lean variant that surfaces the correct scoring-function variants (e.g. `GenScore-balanced` = `GT_ft_0.5`, and `RTMScore` for all three datasets). The corrected v1.1.1 results are produced by the fixed DockM8 source (whole-residue RTMScore/GenScore pocket, complete decoy pool in pose selection, `fast_analyzer` consensus); this pipeline aggregates and plots those results.

## Prerequisites

- DockM8 conda environment activated (`conda activate dockm8`)
- **~500 GB free disk space** for full extraction of all benchmark datasets
- Python packages: numpy, pandas, polars, matplotlib, seaborn, scipy, tqdm (all included in the DockM8 environment)

## 1. Download Raw Data

Download the five `.tar.bz2` archives from Zenodo:

| Archive | Contents | Approx. size | Zenodo DOI |
| --- | --- | --- | --- |
| `DEKOIS.tar.bz2` | 79 DEKOIS 2.0 targets | 26 GB | [10.5281/zenodo.15430058](https://doi.org/10.5281/zenodo.15430058) |
| `DUD-E.tar.bz2` | 28 DUD-E targets | 20 GB | [10.5281/zenodo.15430186](https://doi.org/10.5281/zenodo.15430186) |
| `lit-pcba_1.tar.bz2` | 5 Lit-PCBA targets (part 1 of 3) | 37 GB | [10.5281/zenodo.16436211](https://doi.org/10.5281/zenodo.16436211) |
| `lit-pcba_2.tar.bz2` | 4 Lit-PCBA targets (part 2 of 3) | 47 GB | [10.5281/zenodo.16436304](https://doi.org/10.5281/zenodo.16436304) |
| `lit-pcba_3.tar.bz2` | 5 Lit-PCBA targets (part 3 of 3) | 27 GB | [10.5281/zenodo.16436306](https://doi.org/10.5281/zenodo.16436306) |

## 2. Extract Archives

The Zenodo archives use nested compression (per-target tarballs with gzipped SDF files inside an outer archive). Use the provided extraction script:

```bash
# Extract all 5 archives (~500 GB free space required)
python -m analysis.extract_zenodo /path/to/downloads /path/to/dockm8_data

# Or extract specific archives
python -m analysis.extract_zenodo /path/to/downloads /path/to/dockm8_data \
    --archives DEKOIS.tar.bz2,DUD-E.tar.bz2
```

After extraction, the output directory should contain:

```text
/path/to/dockm8_data/
├── DEKOIS_2.0x/          # 79 target directories
│   ├── ace/
│   ├── ache/
│   └── ...
├── DUD-E/                # 28 target directories
│   ├── abl1/
│   ├── ace/
│   └── ...
└── lit-pcba/             # 14 targets across 3 parts
    ├── PART_1/
    ├── PART_2/
    └── PART_3/
```

Each target directory contains:

```text
[target]/
├── [target]_activity_data.csv
├── [target]_docking_library.sdf
├── [target]_ligand.sdf
├── [target]_pocket.pdb
├── [target]_protein_prepared.pdb
└── results/
    ├── allposes_rescored.csv
    ├── [docking_program]/
    ├── performance/
    │   └── [target]_[program]_[selection]_performance.csv
    ├── scores/
    └── clustering/
```

## 3. Run the Pipeline

```bash
# Full pipeline: aggregate → extract → all plots
python -m analysis.run_all all --base-path /path/to/dockm8_data

# Or run steps individually:
python -m analysis.run_all aggregate --base-path /path/to/dockm8_data
python -m analysis.run_all extract
python -m analysis.run_all boxplots --type internal
python -m analysis.run_all boxplots --type literature
python -m analysis.run_all heatmaps --type docking
python -m analysis.run_all heatmaps --type interaction
python -m analysis.run_all barplots --type frequency
python -m analysis.run_all barplots --type impact
python -m analysis.run_all ave --split-dir PATH --full-dir PATH
```

The `--base-path` flag is only required for `aggregate` and `all` (which calls aggregate). All other commands read from the already-generated `results/output/` directory.

Use `--output-dir` to override the default output location (`results/output/`).

## 4. What Each Step Generates

| Step | Command | Output | What is produced |
| --- | --- | --- | --- |
| Aggregate | `aggregate` | `results/output/aggregated/` | Parquet pivot tables — one per metric x threshold, workflows as rows, targets as columns |
| Extract | `extract` | `results/output/dockm8_results/` | Consolidated CSVs: `*_dockm8_results.csv` (3), `*_dockm8_internal.csv` (3), `best_worst_analysis.csv` |
| Boxplots (internal) | `boxplots --type internal` | `results/output/plots/internal/` | Individual SFs vs DockM8 consensus — 1 figure per dataset x metric (Fig 2) |
| Boxplots (literature) | `boxplots --type literature` | `results/output/plots/literature_per_dataset/` | DockM8 vs published methods — up to 4 variants per dataset x metric (Fig 3) |
| Heatmaps (docking) | `heatmaps --type docking` | `results/output/plots/docking_selection/` | Docking engine x pose-selection method performance (Fig 1A/1B) |
| Heatmaps (interaction) | `heatmaps --type interaction` | `results/output/plots/interaction_sfs_consensus/` | (Number of SFs) x consensus method success rates (Fig 4C) |
| Barplots (frequency) | `barplots --type frequency` | `results/output/plots/scoring_function_frequency/` | SF enrichment in top-ranked workflows (Fig 4A) |
| Barplots (impact) | `barplots --type impact` | `results/output/plots/scoring_function_impact_pvsa_percentiles/` | Median performance shift: SF present vs absent (Fig 4B) |
| AVE analysis | `ave` | `results/output/plots/ave_analysis/` | Training vs validation workflow performance (Fig 5) |

## 5. Output Reference

### Plot naming conventions

| Pattern | Meaning |
| --- | --- |
| `thresh0p1` / `thresh0p5` / `thresh1` / `thresh5` | Threshold: 0.1% / 0.5% / 1% / 5% |
| `1A_` | All workflows vs top 1% |
| `1B_` | Top 1% vs top 0.1% |
| `4C_` | Interaction analysis across percentile bands |
| `mean_ranking_` / `median_ranking_` | Ranked by mean / median performance |
| `_performance` / `_count` | Metric values / workflow counts |
| `_frequency` / `_success_rate` / `_pvsa_percentiles` | SF frequency / success fraction / present-vs-absent impact |

### Literature boxplot suffixes

| Suffix | Filter |
| --- | --- |
| (none) | DockM8 targets only, exclusion list applied |
| `_allmodels` | DockM8 targets only, all literature models |
| `_alltargets` | All targets, exclusion list applied |
| `_allmodels_alltargets` | All targets, all literature models |

### Extracted CSV files

| File | Description |
| --- | --- |
| `{DATASET}_dockm8_results.csv` | Consolidated DockM8 workflow results |
| `{DATASET}_dockm8_internal.csv` | DockM8 internal per-SF results |
| `DUD-E_dockm8_results_noRFScoreVS.csv` | DUD-E excluding RFScoreVS |
| `best_worst_analysis.csv` | Cross-dataset summary by model and percentile |

## 6. Directory Layout

```text
analysis/                         # This directory
├── run_all.py                    # Master CLI orchestrator (the single workflow entry point)
├── extract_zenodo.py             # Zenodo archive extraction utility
├── config.py                     # Path management and constants
├── data_aggregation.py           # Raw CSVs → parquet pivot tables (bounded-memory streaming)
├── streaming_pivot.py            #   └ bounded-memory pivot engine used by data_aggregation
├── data_extraction.py            # Parquets → consolidated comparison CSVs (variant maps)
├── fast_dockm8_extract.py        #   └ vectorized, memory-lean DockM8-best/median extractor
├── docking_selection.py          # Fig 1A, 1B
├── internal_boxplots.py          # Fig 2
├── literature_boxplots.py        # Fig 3
├── sf_frequency.py               # Fig 4A
├── sf_impact.py                  # Fig 4B
├── interaction_plots.py          # Fig 4C
├── ave_analysis.py + plotting.py # Fig 5
├── casf_redocking_benchmark.py   # CASF-2016 re-docking benchmark (Appendix G)
├── plot_casf_redocking.py        #   └ CASF figure
├── redocking_benchmark.py        #   └ symmetric-RMSD helper used by the CASF benchmark
├── lit_metrics.py                # Literature metric calculation
├── data_loader.py                # Data loading utilities
├── workflow_ranking.py           # Workflow ranking utilities
├── plot_helpers.py               # Shared plotting functions
├── utils.py                      # General utilities
├── data/literature/              # Committed baseline CSVs (no Zenodo needed)
│   ├── DEKOIS_literature.csv
│   ├── DUD-E_literature.csv
│   ├── Lit-PCBA_literature.csv
│   ├── *_count_data.csv
│   ├── lit_calc_data_rounded/
│   └── lit_calc_data_unrounded/
└── README.md                     # This file

results/output/                   # Generated outputs (gitignored)
├── aggregated/                   # Parquet pivot tables
├── dockm8_results/               # Consolidated CSVs
└── plots/                        # All generated figures
```

## 7. Data Flow

```text
Zenodo .tar.bz2 archives
        ↓  extract_zenodo.py
Raw CSV files (per target)
        ↓  data_aggregation.py
results/output/aggregated/*.parquet
        ↓  data_extraction.py
results/output/dockm8_results/*.csv
        ↓  visualization modules
results/output/plots/**/*.png
```

## 8. Metrics Reference

| Metric | Full Name | Threshold-Dependent |
| --- | --- | --- |
| `auc_roc` | Area Under ROC Curve | No |
| `bedroc` | Boltzmann-Enhanced Discrimination of ROC | No |
| `ef` | Enrichment Factor | Yes |
| `ref` | Relative Enrichment Factor | Yes |
| `pm` | Power Metric | Yes |
| `roce` | ROC Enrichment | Yes |
| `mcc` | Matthews Correlation Coefficient | Yes |
| `ccr` | Correct Classification Rate | Yes |
| `ckc` | Cohen's Kappa Coefficient | Yes |

## 9. Scoring Function Categories

| Category | Color | Functions |
| --- | --- | --- |
| ML | Blue (royalblue) | RTMScore, NNScore, CNN-Affinity, RFScoreVS, GenScore variants, CNN-Score |
| Empirical | Green (mediumseagreen) | AD4, LinF9, CHEMPLP, Vinardo, GNINA-Affinity |
| Knowledge | Red (lightcoral) | KORP-PL, ConvexPLR |
| Consensus | Orange (#ff7700) | DockM8-* methods |
| Other | Gray (darkgray) | Uncategorized |

## 10. Dependencies

- numpy, pandas, polars (data processing)
- matplotlib, seaborn (plotting)
- scipy (correlation metrics)
- tqdm (progress bars)

## Defaults

```python
DATASETS = ["DEKOIS", "DUD-E", "Lit-PCBA"]
METRICS = ["ref", "ef", "auc_roc", "bedroc", "pm"]
THRESHOLDS = ["0p1", "0p5", "1", "5"]
```

All defaults can be overridden via CLI flags. Run `python -m analysis.run_all <command> --help` for details.

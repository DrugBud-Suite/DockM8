import sys
from pathlib import Path

import medchem as mc
import pandas as pd
import os

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.parallel_executor import parallel_executor


def process_batch(args):
	batch, selected_alerts = args
	rule_filters = mc.structural.CommonAlertsFilters(alerts_set=selected_alerts)
	processed_batch = rule_filters(mols=batch['Molecule'].tolist(), n_jobs=1, progress=False)
	return processed_batch


def apply_alerts_rules(molecule_df: pd.DataFrame, selected_alerts: list, n_cpus: int = int(os.cpu_count() * 0.9)):
	"""
    Apply a set of alerts rules to filter a DataFrame of molecules.

    Args:
        molecule_df (pd.DataFrame): DataFrame containing the molecules to be filtered.
        selected_alerts (list): The set of alerts rules to be applied.
        n_cpus (int): Number of CPUs to be used for parallel processing.
        batch_size (int): Size of each batch for processing.

    Returns:
        Tuple: A tuple containing the filtered DataFrame, the number of molecules filtered out, and the number of remaining molecules.
    """
	batch_size = max(1, len(molecule_df) // (n_cpus*4))

	# Split the dataframe into batches
	batches = [molecule_df[i:i + batch_size] for i in range(0, len(molecule_df), batch_size)]

	# Prepare arguments for parallel processing
	batch_args = [(batch, selected_alerts) for batch in batches]

	# Process batches in parallel
	results = parallel_executor(process_batch,
								batch_args,
								n_cpus=n_cpus,
								job_manager="concurrent_process",
								display_name="Applying Alerts Rules")

	# Combine results
	processed_df = pd.concat(results, ignore_index=True)

	# Merge the processed results with the original DataFrame
	merged_df = pd.concat([molecule_df.reset_index(drop=True), processed_df], axis=1)

	# Filter the merged DataFrame
	filtered_df = merged_df[(merged_df['pass_filter'] != False) & (merged_df['pass_filter'] != 'False')]
	filtered_df = filtered_df[molecule_df.columns]

	num_filtered = len(molecule_df) - len(filtered_df)
	num_remaining = len(filtered_df)
	return filtered_df, num_filtered, num_remaining


ALERTS_RULES = {
	'Glaxo':
  'Glaxo Wellcome Hard filters',
	'Dundee':
  'University of Dundee NTD Screening Library Filters',
	'BMS':
  'Bristol-Myers Squibb HTS Deck filters',
	'PAINS':
  'PAINS filters',
	'SureChEMBL':
  'SureChEMBL Non-MedChem Friendly SMARTS',
	'MLSMR':
  'NIH MLSMR Excluded Functionality filters (MLSMR)',
	'Inpharmatica':
  'Unwanted fragments derived by Inpharmatica Ltd.',
	'LINT':
  'Pfizer lint filters (lint)',
	'Alarm-NMR':
  'Reactive False Positives in Biochemical Screens (Huth et al. https://doi.org/10.1021/ja0455547)',
	'AlphaScreen-Hitters':
  'Structural filters for compounds that may be alphascreen frequent hitters',
	'GST-Hitters':
  'Structural filters for compounds may prevent GST/GSH interaction during HTS',
	'HIS-Hitters':
  'Structural filters for compounds prevents the binding of the protein His-tag moiety to nickel chelate',
	'LuciferaseInhibitor':
  'Structural filters for compounds that may inhibit luciferase.',
	'DNABinder':
  'Structural filters for compounds that may bind to DNA.',
	'Chelator':
  'Structural filters for compounds that may inhibit metalloproteins (chelator).',
	'Frequent-Hitter':
  'Structural filters for compounds that are frequent hitters.',
	'Electrophilic':
  'Structural filters for compounds that could take part in electrophilic reaction and unselectively bind to proteins',
	'Genotoxic-Carcinogenicity':
  'Structural filters for compounds that may cause carcinogenicity or/and mutagenicity through genotoxic mechanisms (Benigni rules, https://publications.jrc.ec.europa.eu/repository/handle/JRC43157)',
	'LD50-Oral':
  'Structural filters for compounds that may cause acute toxicity during oral administration',
	'Non-Genotoxic-Carcinogenicity':
  'Structural filters for compounds that may cause carcinogenicity or/and mutagenicity through non-genotoxic mechanisms (Benigni rules, https://publications.jrc.ec.europa.eu/repository/handle/JRC43157)',
	'Reactive-Unstable-Toxic':
  'General very reactive/unstable or Toxic compounds',
	'Skin':
  'Skin Sensitization filters (irritables)',
	'Toxicophore':
  'General Toxicophores'}

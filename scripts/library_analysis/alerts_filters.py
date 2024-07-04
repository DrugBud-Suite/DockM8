import medchem as mc
import pandas as pd
import os


def apply_alerts_rules(molecule_df: pd.DataFrame, selected_alerts, list, n_cpus: int = int(os.cpu_count() * 0.9)):
	"""
	Apply a set of alerts rules to filter a DataFrame of molecules.

	Args:
		molecule_df (pd.DataFrame): DataFrame containing the molecules to be filtered.
		selected_alerts: The set of alerts rules to be applied.
		list: A list parameter (not specified in the code, please provide more information).
		n_cpus (int): Number of CPUs to be used for parallel processing. Defaults to 90% of the available CPUs.

	Returns:
		Tuple: A tuple containing the filtered DataFrame, the number of molecules filtered out, and the number of remaining molecules.
	"""
	rule_filters = mc.structural.CommonAlertsFilters(alerts_set=selected_alerts)
	processed_df = rule_filters(mols=molecule_df['Molecule'].tolist(), n_jobs=n_cpus, progress=True)
	print(processed_df.columns)
	filtered_df = processed_df[processed_df['pass_all'] != False]
	filtered_df = filtered_df[filtered_df['pass_all'] != 'False']
	filtered_df = filtered_df.drop(columns=['pass_all', 'pass_any'])
	filtered_df = filtered_df.drop(columns=selected_alerts)
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

import sys
from pathlib import Path
import os
import medchem as mc
import pandas as pd

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.parallel_executor import parallel_executor


def process_batch(args):
	batch, selected_rules = args
	rule_filters = mc.rules.RuleFilters(rule_list=selected_rules)
	processed_batch = rule_filters(mols=batch['Molecule'].tolist(), n_jobs=1, progress=False)
	return processed_batch


def apply_medchem_rules(molecule_df: pd.DataFrame, selected_rules: list, n_cpus: int = int(os.cpu_count() * 0.9)):
	"""
    Apply MedChem rules to filter a DataFrame of molecules based on selected rules.

    Args:
        molecule_df (pd.DataFrame): DataFrame containing the molecules to be filtered.
        selected_rules (list): List of selected rules to be applied.
        n_cpus (int): Number of CPUs to use for parallel processing.
        batch_size (int): Size of each batch for processing.

    Returns:
        tuple: A tuple containing the filtered DataFrame, the number of molecules filtered out, and the number of remaining molecules.
    """
	batch_size = max(1, len(molecule_df) // (n_cpus*4))

	# Split the dataframe into batches
	batches = [molecule_df[i:i + batch_size] for i in range(0, len(molecule_df), batch_size)]

	# Prepare arguments for parallel processing
	batch_args = [(batch, selected_rules) for batch in batches]

	# Process batches in parallel
	results = parallel_executor(process_batch,
								batch_args,
								n_cpus=n_cpus,
								job_manager="concurrent_process",
								display_name="Applying MedChem Rules")

	# Combine results
	processed_df = pd.concat(results, ignore_index=True)

	# Merge the processed results with the original DataFrame
	merged_df = pd.concat([molecule_df.reset_index(drop=True), processed_df], axis=1)

	# Filter the merged DataFrame
	filtered_df = merged_df[(merged_df['pass_all'] != False) & (merged_df['pass_all'] != 'False')]
	filtered_df = filtered_df[molecule_df.columns]

	num_filtered = len(molecule_df) - len(filtered_df)
	num_remaining = len(filtered_df)
	return filtered_df, num_filtered, num_remaining


MEDCHEM_RULES = {
	'rule_of_five': {
	'rules':
   '''
        • Molecular Weight ≤ 500
        • LogP ≤ 5
        • Hydrogen Bond Donors ≤ 5
        • Hydrogen Bond Acceptors ≤ 10
        ''',
	'description':
   'Leadlike / Druglike / Small molecule / Library design',
	'alias':
   'Rule Of Five'},
	'rule_of_five_beyond': {
	'rules':
   '''
        • Molecular Weight ≤ 1000
        • LogP between -2 and 10
        • Hydrogen Bond Donors ≤ 6
        • Hydrogen Bond Acceptors ≤ 15
        • Total Polar Surface Area ≤ 250
        • Rotatable Bonds ≤ 20
        ''',
	'description':
   'Leadlike / Druglike / Small molecule / Library design',
	'alias':
   'Beyond Rule Of Five'},
	'rule_of_four': {
	'rules':
   '''
        • Molecular Weight ≥ 400
        • LogP ≥ 4
        • Number of Rings ≥ 4
        • Hydrogen Bond Acceptors ≥ 4
        ''',
	'description':
   'Protein-Protein Interaction inhibitor / Druglike',
	'alias':
   'Rule of Four'},
	'rule_of_three': {
	'rules':
   '''
        • Molecular Weight ≤ 300
        • LogP ≤ 3
        • Hydrogen Bond Acceptors ≤ 3
        • Hydrogen Bond Donors ≤ 3
        • Rotatable Bonds ≤ 3
        ''',
	'description':
   'Fragment / Building block',
	'alias':
   'Rule of Three (Fragments)'},
	'rule_of_three_extended': {
	'rules':
   '''
        • Molecular Weight ≤ 300
        • LogP between -3 and 3
        • Hydrogen Bond Acceptors ≤ 6
        • Hydrogen Bond Donors ≤ 3
        • Rotatable Bonds ≤ 3
        • Total Polar Surface Area ≤ 60
        ''',
	'description':
   'Fragment / Building block',
	'alias':
   'Extended Rule of Three (Fragments)'},
	'rule_of_two': {
	'rules':
   '''
        • Molecular Weight ≤ 200
        • LogP ≤ 2
        • Hydrogen Bond Acceptors ≤ 4
        • Hydrogen Bond Donors ≤ 2
        ''',
	'description':
   'Fragment / Reagent / Building block',
	'alias':
   'Rule of Two'},
	'rule_of_ghose': {
	'rules':
   '''
        • Molecular Weight between 160 and 480
        • LogP between -0.4 and 5.6
        • Number of Atoms between 20 and 70
        • Molar Refractivity between 40 and 130
        ''',
	'description':
   'Leadlike / Druglike / Small molecule / Library design',
	'alias':
   'Ghose Filter'},
	'rule_of_veber': {
	'rules': '''
        • Rotatable Bonds ≤ 10
        • Total Polar Surface Area < 140
        ''',
	'description': 'Druglike / Leadlike / Small molecule / Oral bioavailability',
	'alias': 'Veber Filter'},
	'rule_of_reos': {
	'rules':
   '''
        • Molecular Weight between 200 and 500
        • LogP between -5 and 5
        • Hydrogen Bond Acceptors between 0 and 10
        • Hydrogen Bond Donors between 0 and 5
        • Molecular Charge between -2 and 2
        • Rotatable Bonds between 0 and 8
        • Heavy Atoms between 15 and 50
        ''',
	'description':
   'Druglike / Small molecule / Library design / High-Throughput Screening',
	'alias':
   'REOS Filter'},
	'rule_of_chemaxon_druglikeness': {
	'rules':
   '''
        • Molecular Weight < 400
        • LogP < 5
        • Hydrogen Bond Acceptors ≤ 10
        • Hydrogen Bond Donors ≤ 5
        • Rotatable Bonds < 5
        • Number of Rings > 0
        ''',
	'description':
   'Leadlike / Druglike / Small molecule',
	'alias':
   'ChemAxon Druglikeness Filter'},
	'rule_of_egan': {
	'rules': '''
        • Total Polar Surface Area between 0 and 132
        • LogP between -1 and 6
        ''',
	'description': 'Druglike / Small molecule / ADMET / Absorption / Permeability',
	'alias': 'Egan Filter'},
	'rule_of_pfizer_3_75': {
	'rules': 'Not (Total Polar Surface Area < 75 AND LogP > 3)',
	'description': 'Druglike / Toxicity / In vivo / Small molecule',
	'alias': 'Pfizer Filter'},
	'rule_of_gsk_4_400': {
	'rules': '''
        • Molecular Weight ≤ 400
        • LogP ≤ 4
        ''',
	'description': 'Druglike / ADMET / Small molecule',
	'alias': 'GSK Filter'},
	'rule_of_oprea': {
	'rules':
   '''
        • Hydrogen Bond Donors between 0 and 2
        • Hydrogen Bond Acceptors between 2 and 9
        • Rotatable Bonds between 2 and 8
        • Number of Rings between 1 and 4
        ''',
	'description':
   'Druglike / Small molecule',
	'alias':
   'Oprea Filter'},
	'rule_of_xu': {
	'rules':
   '''
        • Hydrogen Bond Donors ≤ 5
        • Hydrogen Bond Acceptors ≤ 10
        • Rotatable Bonds between 2 and 35
        • Number of Rings between 1 and 7
        • Heavy Atoms between 10 and 50
        ''',
	'description':
   'Druglike / Small molecule / Library design',
	'alias':
   'Xu Filter'},
	'rule_of_cns': {
	'rules':
   '''
        • Molecular Weight between 135 and 582
        • LogP between -0.2 and 6.1
        • Total Polar Surface Area between 3 and 118
        • Hydrogen Bond Donors ≤ 3
        • Hydrogen Bond Acceptors ≤ 5
        ''',
	'description':
   'Druglike / Central Nervous System / Blood-Brain Barrier / Small molecule',
	'alias':
   'CNS-focused Filter'},
	'rule_of_respiratory': {
	'rules':
   '''
        • Molecular Weight between 240 and 520
        • LogP between -2 and 4.7
        • Hydrogen Bonds between 6 and 12
        • Total Polar Surface Area between 51 and 135
        • Rotatable Bonds between 3 and 8
        • Number of Rings between 1 and 5
        ''',
	'description':
   'Druglike / Respiratory / Small molecule / Nasal / Inhalatory',
	'alias':
   'Respiratory-focused Filter'},
	'rule_of_zinc': {
	'rules':
   '''
        • Molecular Weight between 60 and 600
        • LogP between -4 and 6
        • Hydrogen Bond Donors ≤ 6
        • Hydrogen Bond Acceptors ≤ 11
        • Total Polar Surface Area ≤ 150
        • Rotatable Bonds ≤ 12
        • Rigid Bonds ≤ 50
        • Number of Rings ≤ 7
        • Maximum Ring Size ≤ 12
        • Number of Carbon Atoms ≥ 3
        • Hydrogen-Carbon Ratio ≤ 2.0
        • Molecular Charge between -4 and 4
        ''',
	'description':
   'Druglike / Small molecule / Library design / ZINC database',
	'alias':
   'ZINC Druglikeness Filter'},
	'rule_of_leadlike_soft': {
	'rules':
   '''
        • Molecular Weight between 150 and 400
        • LogP between -3 and 4
        • Hydrogen Bond Donors ≤ 4
        • Hydrogen Bond Acceptors ≤ 7
        • Total Polar Surface Area ≤ 160
        • Rotatable Bonds ≤ 9
        • Rigid Bonds ≤ 30
        • Number of Rings ≤ 4
        • Maximum Ring Size ≤ 18
        • Number of Carbon Atoms between 3 and 35
        • Number of Heteroatoms between 1 and 15
        • Hydrogen-Carbon Ratio between 0.1 and 1.1
        • Molecular Charge between -4 and 4
        • Number of Charged Atoms ≤ 4
        • Number of Stereo Centers ≤ 2
        ''',
	'description':
   'Leadlike / Small molecule / Library design / ADMET',
	'alias':
   'Soft Leadlike Filter'},
	'rule_of_druglike_soft': {
	'rules':
   '''
        • Molecular Weight between 100 and 600
        • LogP between -3 and 6
        • Hydrogen Bond Donors ≤ 7
        • Hydrogen Bond Acceptors ≤ 12
        • Total Polar Surface Area ≤ 180
        • Rotatable Bonds ≤ 11
        • Rigid Bonds ≤ 30
        • Number of Rings ≤ 6
        • Maximum Ring Size ≤ 18
        • Number of Carbon Atoms between 3 and 35
        • Number of Heteroatoms between 1 and 15
        • Hydrogen-Carbon Ratio between 0.1 and 1.1
        • Molecular Charge between -4 and 4
        • Number of Charged Atoms ≤ 4
        ''',
	'description':
   'Druglike / Small molecule / Library design',
	'alias':
   'Soft Druglike Filter'},
	'rule_of_generative_design': {
	'rules':
   '''
        • Molecular Weight between 200 and 600
        • LogP between -3 and 6
        • Hydrogen Bond Donors ≤ 7
        • Hydrogen Bond Acceptors ≤ 12
        • Total Polar Surface Area between 40 and 180
        • Rotatable Bonds ≤ 15
        • Rigid Bonds ≤ 30
        • Number of Aromatic Rings ≤ 5
        • Number of Fused Aromatic Rings ≤ 2
        • Maximum Ring System Size ≤ 18
        • Number of Carbon Atoms between 3 and 40
        • Number of Heteroatoms between 1 and 15
        • Molecular Charge between -2 and 2
        • Number of Charged Atoms ≤ 2
        • Total Number of Atoms < 70
        • Number of Heavy Metals < 1
        ''',
	'description':
   'Druglike / Small molecule / De novo design / Generative models / Permissive',
	'alias':
   'Generative Design Filter'},
	'rule_of_generative_design_strict': {
	'rules':
   '''
        • Molecular Weight between 200 and 600
        • LogP between -3 and 6
        • Hydrogen Bond Donors ≤ 7
        • Hydrogen Bond Acceptors ≤ 12
        • Total Polar Surface Area between 40 and 180
        • Rotatable Bonds ≤ 15
        • Rigid Bonds ≤ 30
        • Number of Aromatic Rings ≤ 5
        • Number of Fused Aromatic Rings ≤ 2
        • Maximum Ring System Size ≤ 18
        • Number of Carbon Atoms between 3 and 40
        • Number of Heteroatoms between 1 and 15
        • Molecular Charge between -2 and 2
        • Number of Charged Atoms ≤ 2
        • Total Number of Atoms < 70
        • Number of Heavy Metals < 1
        • Number of Stereo Centers ≤ 3
        • No spider-like side chains
        • Fraction of atoms in ring systems ≥ 0.25
        ''',
	'description':
   'Druglike / Small molecule / De novo design / Generative models / Restricted long chains / Limited stereo centers',
	'alias':
   'Strict Generative Design Filter'}}

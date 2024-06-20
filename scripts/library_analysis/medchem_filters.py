import medchem as mc
import pandas as pd
import os


def apply_medchem_rules(molecule_df: pd.DataFrame, selected_rules: list, n_cpus: int = int(os.cpu_count() * 0.9)):
	"""
	Apply MedChem rules to filter a DataFrame of molecules based on selected rules.

	Args:
		molecule_df (pd.DataFrame): DataFrame containing the molecules to be filtered.
		selected_rules (list): List of selected rules to be applied.
		n_cpus (int, optional): Number of CPUs to use for parallel processing. Defaults to 90% of available CPUs.

	Returns:
		tuple: A tuple containing the filtered DataFrame, the number of molecules filtered out, and the number of remaining molecules.
	"""
	rule_filters = mc.rules.RuleFilters(rule_list=selected_rules)
	processed_df = rule_filters(mols=molecule_df['Molecule'].tolist(), n_jobs=n_cpus, progress=True)
	filtered_df = processed_df[processed_df['pass_all'] != False]
	filtered_df = filtered_df[filtered_df['pass_all'] != 'False']
	filtered_df = filtered_df.drop(columns=['pass_all', 'pass_any'])
	filtered_df = filtered_df.drop(columns=selected_rules)
	num_filtered = len(molecule_df) - len(filtered_df)
	num_remaining = len(filtered_df)
	return filtered_df, num_filtered, num_remaining


MEDCHEM_RULES = {
	'rule_of_five': {
		'rules': 'MW <= 500 & logP <= 5 & HBD <= 5 & HBA <= 10',
		'description': 'leadlike;druglike;small molecule;library design',
		'alias': 'Rule Of Five'},
	'rule_of_five_beyond': {
		'rules': 'MW <= 1000 & logP in [-2, 10] & HBD <= 6 & HBA <= 15 & TPSA <=250 & rotatable bond <= 20',
		'description': 'leadlike;druglike;small molecule;library design',
		'alias': 'Beyond Rule Of Five'},
	'rule_of_four': {
		'rules': 'MW >= 400 & logP >= 4 & RINGS >=4 & HBA >= 4',
		'description': 'PPI inhibitor;druglike',
		'alias': 'Rule of Four'},
	'rule_of_three': {
		'rules': 'MW <= 300 & logP <= 3 & HBA <= 3 & HBD <= 3 & ROTBONDS <= 3',
		'description': 'fragment;building block',
		'alias': 'Rule of Three (Fragments)'},
	'rule_of_three_extended': {
		'rules': 'MW <= 300 & logP in [-3, 3] & HBA <= 6 & HBD <= 3 & ROTBONDS <= 3 & TPSA <= 60',
		'description': 'fragment;building block',
		'alias': 'Extended Rule of Three (Fragments)'},
	'rule_of_two': {
		'rules': 'MW <= 200 & logP <= 2 & HBA <= 4 & HBD <= 2',
		'description': 'fragment;reagent;building block',
		'alias': 'Rule of Two'},
	'rule_of_ghose': {
		'rules': 'MW in [160, 480] & logP in [-0.4, 5.6] & Natoms in [20, 70] & refractivity in [40, 130]',
		'description': 'leadlike;druglike;small molecule;library design',
		'alias': 'Ghose Filter'},
	'rule_of_veber': {
		'rules': 'rotatable bond <= 10 & TPSA < 140',
		'description': 'druglike;leadlike;small molecule;oral',
		'alias': 'Veber Filter'},
	'rule_of_reos': {
		'rules':
			'MW in [200, 500] & logP in [-5, 5] & HBA in [0, 10] & HBD in [0, 5] & charge in [-2, 2] & rotatable bond in [0, 8] & NHeavyAtoms in [15, 50]',
		'description':
			'druglike;small molecule;library design;HTS',
		'alias':
			'REOS Filter'},
	'rule_of_chemaxon_druglikeness': {
		'rules': 'MW < 400 & logP < 5 & HBA <= 10 & HBD <= 5 & rotatable bond < 5 & ring > 0',
		'description': 'leadlike;druglike;small molecule',
		'alias': 'ChemAxon Druglikeness Filter'},
	'rule_of_egan': {
		'rules': 'TPSA in [0, 132] & logP in [-1, 6]',
		'description': 'druglike;small molecule;admet;absorption;permeability',
		'alias': 'Egan Filter'},
	'rule_of_pfizer_3_75': {
		'rules': 'not (TPSA < 75 & logP > 3)',
		'description': 'druglike;toxicity;invivo;small molecule',
		'alias': 'Pfizer Filter'},
	'rule_of_gsk_4_400': {
		'rules': 'MW <= 400 & logP <= 4', 'description': 'druglike;admet;small molecule', 'alias': 'GSK Filter'},
	'rule_of_oprea': {
		'rules': 'HBD in [0, 2] & HBA in [2, 9] & ROTBONDS in [2,8] and RINGS in [1, 4]',
		'description': 'druglike;small molecule',
		'alias': 'Oprea Filter'},
	'rule_of_xu': {
		'rules': 'HBD <= 5 & HBA <= 10 & ROTBONDS in [2, 35] & RINGS in [1, 7] & NHeavyAtoms in [10, 50]',
		'description': 'druglike;small molecule;library design',
		'alias': 'Xu Filter'},
	'rule_of_cns': {
		'rules': 'MW in [135, 582] & logP in [-0.2, 6.1] & TPSA in [3, 118] & HBD <= 3 & HBA <= 5',
		'description': 'druglike;CNS;BBB;small molecule',
		'alias': 'CNS-focused Filter'},
	'rule_of_respiratory': {
		'rules':
			'MW in [240, 520]  & logP in [-2, 4.7] & HBONDS in [6, 12] & TPSA in [51, 135] & ROTBONDS in [3,8] & RINGS in [1,5]',
		'description':
			'druglike;respiratory;small molecule;nasal;inhalatory',
		'alias':
			'Respiratory-focused Filter'},
	'rule_of_zinc': {
		'rules':
			'MW in [60, 600] & logP < in [-4, 6] & HBD <= 6 & HBA <= 11 & TPSA <=150 & ROTBONDS <= 12 & RIGBONDS <= 50 & N_RINGS <= 7 & MAX_SIZE_RING <= 12 & N_CARBONS >=3 & HC_RATIO <= 2.0 & CHARGE in [-4, 4]',
		'description':
			'druglike;small molecule;library design;zinc',
		'alias':
			'ZINC Druglikeness Filter'},
	'rule_of_leadlike_soft': {
		'rules':
			'MW in [150, 400] & logP < in [-3, 4] & HBD <= 4 & HBA <= 7 & TPSA <=160 & ROTBONDS <= 9 & RIGBONDS <= 30 & N_RINGS <= 4 & MAX_SIZE_RING <= 18 & N_CARBONS in [3, 35] &  N_HETEROATOMS in [1, 15] & HC_RATIO in [0.1, 1.1] & CHARGE in [-4, 4] & N_ATOM_CHARGE <= 4 & N_STEREO_CENTER <= 2',
		'description':
			'leadlike;small molecule;library design;admet',
		'alias':
			'Soft Leadlike Filter'},
	'rule_of_druglike_soft': {
		'rules':
			'MW in [100, 600] & logP < in [-3, 6] & HBD <= 7 & HBA <= 12 & TPSA <=180 & ROTBONDS <= 11 & RIGBONDS <= 30 & N_RINGS <= 6 & MAX_SIZE_RING <= 18 & N_CARBONS in [3, 35] &  N_HETEROATOMS in [1, 15] & HC_RATIO in [0.1, 1.1] & CHARGE in [-4, 4] & N_ATOM_CHARGE <= 4',
		'description':
			'druglike;small molecule;library design',
		'alias':
			'Soft Druglike Filter'},
	'rule_of_generative_design': {
		'rules':
			'MW in [200, 600] & logP < in [-3, 6] & HBD <= 7  & HBA <= 12 & TPSA in [40, 180] & ROTBONDS <= 15 & RIGID BONDS <= 30 & N_AROMATIC_RINGS <= 5 & N_FUSED_AROMATIC_RINGS_TOGETHER <= 2 & MAX_SIZE_RING_SYSTEM <= 18  & N_CARBONS in [3, 40] & N_HETEROATOMS in [1, 15] & CHARGE in [-2, 2] & N_ATOM_CHARGE <= 2 & N_TOTAL_ATOMS < 70 & N_HEAVY_METALS < 1',
		'description':
			'druglike;small molecule;de novo design;generative models;permissive',
		'alias':
			'Generative Design Filter'},
	'rule_of_generative_design_strict': {
		'rules':
			'MW in [200, 600] & logP < in [-3, 6] & HBD <= 7  & HBA <= 12 & TPSA in [40, 180] & ROTBONDS <= 15 & RIGID BONDS <= 30 & N_AROMATIC_RINGS <= 5 & N_FUSED_AROMATIC_RINGS_TOGETHER <= 2 & MAX_SIZE_RING_SYSTEM <= 18  & N_CARBONS in [3, 40] & N_HETEROATOMS in [1, 15] & CHARGE in [-2, 2] & N_ATOM_CHARGE <= 2 & N_TOTAL_ATOMS < 70 & N_HEAVY_METALS < 1 & N_STEREO_CENTER <= 3 & HAS_NO_SPIDER_SIDE_CHAINS & FRACTION_RING_SYSTEM>=0.25',
		'description':
			'druglike;small molecule;de novo design;generative models;long chains;stereo centers',
		'alias':
			'Strict Generative Design Filter'}}

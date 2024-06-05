from rdkit.Chem import Descriptors, rdMolDescriptors, QED


def calculate_properties(df, properties):
	if 'MW' in properties:
		df['MW'] = df['Molecule'].apply(lambda x: Descriptors.MolWt(x))
		df['MW'] = df['MW'].astype(float)
	if 'TPSA' in properties:
		df['TPSA'] = df['Molecule'].apply(lambda x: Descriptors.TPSA(x))
		df['TPSA'] = df['TPSA'].astype(float)
	if 'HBA' in properties:
		df['HBA'] = df['Molecule'].apply(lambda x: rdMolDescriptors.CalcNumHBA(x))
		df['HBA'] = df['HBA'].astype(int)
	if 'HBD' in properties:
		df['HBD'] = df['Molecule'].apply(lambda x: rdMolDescriptors.CalcNumHBD(x))
		df['HBD'] = df['HBD'].astype(int)
	if 'Rotatable Bonds' in properties:
		df['Rotatable Bonds'] = df['Molecule'].apply(lambda x: Descriptors.NumRotatableBonds(x))
		df['Rotatable Bonds'] = df['Rotatable Bonds'].astype(int)
	if 'QED' in properties:
		df['QED'] = df['Molecule'].apply(lambda x: QED.qed(x))
		df['QED'] = df['QED'].astype(float)
	if 'sp3 percentage' in properties:
		df['sp3 percentage'] = df['Molecule'].apply(lambda x: rdMolDescriptors.CalcFractionCSP3(x))
		df['sp3 percentage'] = df['sp3 percentage'].astype(float)
	if 'Ring Count' in properties:
		df['Ring Count'] = df['Molecule'].apply(lambda x: rdMolDescriptors.CalcNumRings(x))
		df['Ring Count'] = df['Ring Count'].astype(int)
	return df

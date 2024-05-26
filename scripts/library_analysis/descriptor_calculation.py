from rdkit.Chem import Descriptors, rdMolDescriptors, QED


def calculate_properties(df, properties):
	if 'MW' in properties:
		df['MW'] = df['Molecule'].apply(lambda x: Descriptors.MolWt(x))
	if 'TPSA' in properties:
		df['TPSA'] = df['Molecule'].apply(lambda x: Descriptors.TPSA(x))
	if 'LogD' in properties:
		df['LogD'] = df['Molecule'].apply(lambda x: rdMolDescriptors.CalcCrippenDescriptors(x)[1])
	if 'HBA' in properties:
		df['HBA'] = df['Molecule'].apply(lambda x: rdMolDescriptors.CalcNumHBA(x))
	if 'HBD' in properties:
		df['HBD'] = df['Molecule'].apply(lambda x: rdMolDescriptors.CalcNumHBD(x))
	if 'Rotatable Bonds' in properties:
		df['Rotatable Bonds'] = df['Molecule'].apply(lambda x: Descriptors.NumRotatableBonds(x))
	if 'QED' in properties:
		df['QED'] = df['Molecule'].apply(lambda x: QED.qed(x))
	if 'sp3 percentage' in properties:
		df['sp3 percentage'] = df['Molecule'].apply(lambda x: rdMolDescriptors.CalcFractionCSP3(x))
	if 'Ring Count' in properties:
		df['Ring Count'] = df['Molecule'].apply(lambda x: rdMolDescriptors.CalcNumRings(x))
	return df

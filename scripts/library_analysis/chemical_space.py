import os
from rdkit import Chem
import numpy as np
import pandas as pd
import plotly.express as px
from skfp.fingerprints import ECFPFingerprint, MACCSFingerprint, TopologicalTorsionFingerprint
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


def visualize_chemical_space(df, method='UMAP', fingerprint='ECFP4'):
	# Compute SMILES if not present in df
	if 'SMILES' not in df.columns:
		df['SMILES'] = df['Molecule'].apply(lambda mol: Chem.MolToSmiles(mol) if mol is not None else None)

	# Continue with the rest of the code
	# Compute fingerprints
	if fingerprint == 'ECFP4':
		fp_list = ECFPFingerprint(fp_size=2048, radius=2, n_jobs=int(os.cpu_count() * 0.9)).transform(df['SMILES'])
	elif fingerprint == 'FCFP4':
		fp_list = ECFPFingerprint(fp_size=2048, radius=2, use_fcfp=True,
									n_jobs=int(os.cpu_count() * 0.9)).transform(df['SMILES'])
	elif fingerprint == 'MACCS':
		fp_list = MACCSFingerprint(n_jobs=int(os.cpu_count() * 0.9)).transform(df['SMILES'])
	elif fingerprint == 'Torsion':
		fp_list = TopologicalTorsionFingerprint(n_jobs=int(os.cpu_count() * 0.9)).transform(df['SMILES'])

	fingerprint_df = pd.DataFrame(np.array([list(fp) for fp in fp_list]))

	# Perform dimensionality reduction
	if method == 'UMAP':
		reducer = UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric='jaccard', n_jobs=int(os.cpu_count() * 0.9))
	elif method == 'T-SNE':
		reducer = TSNE(n_components=2, n_jobs=int(os.cpu_count() * 0.9))
	elif method == 'PCA':
		reducer = PCA(n_components=2)

	embedding = reducer.fit_transform(fingerprint_df)
	embedding_df = pd.DataFrame(embedding, columns=[f'{method} Dimension 1', f'{method} Dimension 2'])
	embedding_df['ID'] = df['ID'].values
	fig = px.scatter(embedding_df,
						x=f'{method} Dimension 1',
						y=f'{method} Dimension 2',
						labels={
							'x': f'{method} Dimension 1', 'y': f'{method} Dimension 2'},
						title=f"{method} Projection of Chemical Space using {fingerprint}",
						color_discrete_sequence=['#FF871F'],
						hover_data=['ID'])                                                  # Now 'ID' is a valid column in embedding_df

	return fig, embedding_df

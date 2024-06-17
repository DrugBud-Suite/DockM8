import os

import numpy as np
import pandas as pd
import plotly.express as px
from skfp.fingerprints import ECFPFingerprint, MACCSFingerprint, TopologicalTorsionFingerprint
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


def visualize_chemical_space(df, method='UMAP', fingerprint='ECFP4'):
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
		reducer = UMAP(n_neighbors=15, min_dist=0.1, metric='jaccard')
	elif method == 'T-SNE':
		reducer = TSNE(n_components=2)
	elif method == 'PCA':
		reducer = PCA(n_components=2)

	embedding = reducer.fit_transform(fingerprint_df)

	embedding_df = pd.DataFrame(embedding)

	# Create a plot
	fig = px.scatter(embedding_df,
						labels={
							'x': f'{method} Dimension 1', 'y': f'{method} Dimension 2'},
						title=f"{method} Projection of Chemical Space using {fingerprint}",
						color_discrete_sequence=['#FF871F'],
						hover_name='ID')

	return fig

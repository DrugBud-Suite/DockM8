import pickle
import sys
from pathlib import Path

import oddt
import pandas as pd
from joblib import Parallel, delayed
from oddt.fingerprints import PLEC
from tqdm import tqdm

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.utilities import printlog


def classy_pose_filter(dataframe: pd.DataFrame, protein_file: Path, model_file: Path, n_cpus: int) -> pd.DataFrame:
	"""
	Predicts docked pose quality using the Classy_Pose model.

	Args:
		dataframe (pandas.DataFrame): The input DataFrame containing ligand structures.
		protein_file (Path): The path to the protein file in PDB format.
		model_file (Path): The path to the trained Classy_Pose model file.
		n_cpus (int): The number of CPUs to use for parallel processing.

	Returns:
		pandas.DataFrame: A DataFrame containing the filtered poses with predicted class and probability.

	Raises:
		FileNotFoundError: If any of the input files cannot be found.
	"""
	dockm8_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()),
						None).parent
	if model_file is None or model_file == 'SVM':
		printlog("Using default Classy Pose model (SVM).")
		model_file = dockm8_path / "software" / "models" / "classy_pose_SVM_paper.pkl"
	elif model_file == 'LGBM':
		printlog("Using Classy Pose model (LGBM).")
		model_file = dockm8_path / "software" / "models" / "classy_pose_LGBMClassifier_optimized.pkl"

	printlog(f"Predicting docked pose quality using Classy_Pose model : {model_file}")

	mols = [oddt.toolkits.rdk.Molecule(mol) for mol in dataframe['Molecule']]

	# Load the protein file (assuming it's in PDB format)
	protein = next(oddt.toolkit.readfile('pdb', str(protein_file)))

	printlog("Calculating PLEC features...")

	global parallel_plec

	def parallel_plec(mol):
		feature = PLEC(mol, protein, size=4092, depth_protein=4, depth_ligand=2, distance_cutoff=4.5, sparse=False)
		return feature

	features = Parallel(n_jobs=n_cpus, backend="multiprocessing")(delayed(parallel_plec)(mol) for mol in tqdm(mols))

	plec_dataframe = pd.DataFrame(features)

	feature_columns = {plec_dataframe.columns[i]: f'f{i-2}' for i in range(4, plec_dataframe.shape[1])}

	plec_dataframe.rename(columns=feature_columns, inplace=True)

	test_data = plec_dataframe
	test_data.columns = test_data.columns.astype(str)

	model = pickle.load(open(model_file, 'rb'))
	printlog("Predicting pose quality...")
	if hasattr(model, "predict_proba"):
		pred_probs = model.predict_proba(test_data)
		# Handle models that only return one column of probabilities for the positive class
		if pred_probs.shape[1] == 1:
			prediction_test_prob = pred_probs.ravel()
		else:
			prediction_test_prob = pred_probs[:, 1]
	elif hasattr(model, "decision_function"):                                                    # For models like SVM
		decision_scores = model.decision_function(test_data)
		                                                                                              # Normalize decision scores to [0, 1] as an example normalization
		prediction_test_prob = (decision_scores - decision_scores.min()) / (decision_scores.max() -
																			decision_scores.min())
	else:
		prediction_test_prob = model.predict(test_data).ravel()                                     # Use predict directly if no probabilities are provided

	dataframe['Good_Pose_Prob'] = prediction_test_prob

	predicted_class = ['Good' if prob > 0.5 else 'Bad' for prob in prediction_test_prob]

	dataframe['Predicted_Class'] = predicted_class

	filtered_dataframe = dataframe[dataframe['Predicted_Class'] == 'Good']

	filtered_dataframe = filtered_dataframe.drop(columns=['Good_Pose_Prob', 'Predicted_Class'])

	return filtered_dataframe

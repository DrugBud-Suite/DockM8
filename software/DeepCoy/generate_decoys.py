import os
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

from scripts.utilities.utilities import convert_molecules, printlog

from ..DeepCoy.data.prepare_data import preprocess, read_file
from ..DeepCoy.DeepCoy import DenseGGNNChemModel
from ..DeepCoy.evaluation.select_and_evaluate_decoys import select_and_evaluate_decoys


def generate_decoys(input_sdf: Path, n_decoys: int, model: str, software: Path) -> Path:
    """
    Generate decoys based on a given input molecule file.

    Args:
        input_sdf (Path): The path to the input SDF file containing the molecules for which decoys need to be generated.
        n_decoys (int): The number of decoys to be generated for each input molecule.
        model (str): The name of the DeepCoy model to be used for decoy generation.
        software (Path): The path to the folder containing the DeepCoy software and models.

    Returns:
        Path: The path to the output SDF file containing the generated decoys and actives.
    """
    printlog('Generating decoys...')
    tic = time.perf_counter()
    DeepCoy_folder = (input_sdf.parent / 'DeepCoy')
    DeepCoy_folder.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(DeepCoy_folder / 'actives.smi'):
        try: 
            #Convert to SMILES
            convert_molecules(input_sdf, DeepCoy_folder / 'actives.smi', 'sdf', 'smi')
            #Remove IDs from SMILES file
            with open(DeepCoy_folder / 'actives.smi', 'r') as f:
                lines = f.readlines()
                smiles = [line.split()[0] for line in lines]
            with open(DeepCoy_folder / 'actives.smi', 'w') as f:
                f.writelines('\n'.join(smiles))
        except Exception as e:
            printlog(e)
            printlog('Error converting library to SMILES for decoy generation')
    if not os.path.exists(DeepCoy_folder / 'molecules_actives.json'):
        try:
            preprocess(read_file(DeepCoy_folder / 'actives.smi'), "zinc", 'actives', str(DeepCoy_folder) + "/")
        except Exception as e:
            printlog(e)
            printlog('Error preprocessing library for decoy generation')
    if not os.path.exists(DeepCoy_folder / 'decoys.smi'):
        try:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            
            # Arguments for DeepCoy
            args = defaultdict(None)
            args['--dataset'] = 'zinc'
            args['--config'] = f'{{"generation": true, \
                                "batch_size": 1, \
                                "number_of_generation_per_valid": {n_decoys*10}, \
                                "train_file": "{DeepCoy_folder / "molecules_actives.json"}", \
                                "valid_file": "{DeepCoy_folder / "molecules_actives.json"}", \
                                "output_name": "{DeepCoy_folder / "decoys.smi"}", \
                                "use_subgraph_freqs": false}}'
            args['--freeze-graph-model'] = False
            if model == 'DUDE':
                args['--restore'] = f'{software}/models/DeepCoy_DUDE_model_e09.pickle'
            elif model == 'DEKOIS':
                args['--restore'] = f'{software}/models/DeepCoy_DEKOIS_model_e10.pickle'
            elif model == 'DUDE_P':
                args['--restore'] = f'{software}/models/DeepCoy_DUDE_phosphorus_model_e10.pickle'
            else:
                raise ValueError('DeepCoy Model not recognized!')

            model = DenseGGNNChemModel(args)
            model.train()
            model = ' '
        except Exception as e:
            printlog(e)
            printlog('Error generating decoys!')
    # Delete files ending with params_zinc.json
    for file in os.listdir():
        if file.endswith("params_zinc.json"):
            os.remove(file)

    # Delete files ending with generated_smiles_zinc
    for file in os.listdir():
        if file.endswith("generated_smiles_zinc"):
            os.remove(file)
    if not os.path.exists(DeepCoy_folder / 'decoys-selected.smi'):
        try:
            results = select_and_evaluate_decoys('/decoys.smi', 
                                                file_loc=str(DeepCoy_folder), 
                                                output_loc=str(DeepCoy_folder)+'/', 
                                                dataset="ALL", 
                                                num_cand_dec_per_act=n_decoys*2, 
                                                num_dec_per_act=n_decoys)
            
            printlog("DOE score: \t\t\t%.3f" % results[8])
            printlog("Average Doppelganger score: \t%.3f" % results[10])
            printlog("Max Doppelganger score: \t%.3f" % results[11])
        except Exception as e:
            printlog(e)
            printlog('Error selecting and evaluating decoys!')
    if not os.path.exists(DeepCoy_folder / 'test_set.sdf'):
        with open(DeepCoy_folder / 'decoys-selected.smi', 'r') as f:
            lines = f.readlines()
            actives = set([line.split()[0] for line in lines])
            decoys = [line.split()[1] for line in lines]

        # Load the actives and decoys as RDKit molecules from the SMILES format
        actives_mols = [Chem.MolFromSmiles(smiles) for smiles in actives]
        decoys_mols = [Chem.MolFromSmiles(smiles) for smiles in decoys]

        # Add them to a dataframe
        actives_df = pd.DataFrame()
        actives_df['Molecule'] = actives_mols
        actives_df['Activity'] = 1
        actives_df['ID'] = ['Active-' + str(i) for i in range(1, len(actives_df) + 1)]
        decoys_df = pd.DataFrame()
        decoys_df['Molecule'] = decoys_mols
        decoys_df['Activity'] = 0
        decoys_df['ID'] = ['Decoy-' + str(i) for i in range(1, len(decoys_df) + 1)]
        output_df = pd.concat([actives_df, decoys_df], ignore_index=True)

        # Save the dataframe as an SDF file
        PandasTools.WriteSDF(output_df, str(DeepCoy_folder / 'test_set.sdf'), molColName='Molecule', idName='ID', properties=list(output_df.columns))
    toc = time.perf_counter()
    printlog(f'Finished generating decoys in {toc-tic:0.4f}!...')
    return DeepCoy_folder / 'test_set.sdf'
#!/usr/bin/env python
"""
Usage:
    select_and_evaluate_decoys.py [options]

Options:
    -h --help                       Show this screen.
    --data_path NAME                Path to data file or directory containing multiple files
    --output_path NAME              Path to output location [default: ./]
    --dataset_name NAME             Name of dataset (Options: dude, dude-ext, dekois, MUV, ALL)
    --num_decoys_per_active INT     Number of decoys to select per active [default: 50]
    --min_num_candidates INT        Minimum number of candidate decoys [default: 100]
    --min_active_size INT           Minimum number of atoms in active molecule [default: 10]
    --num_cores INT                 Number of cores to use [default: 1]
    --max_idx_cmpd INT              Maximum number of decoys per active to consider [default: 10000]
"""

import os, csv

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

from ..evaluation import decoy_utils, sascorer

from joblib import Parallel, delayed
from docopt import docopt


# Worker function
def select_and_evaluate_decoys(f,
                               file_loc='./',
                               output_loc='./',
                               dataset='dude',
                               num_cand_dec_per_act=100,
                               num_dec_per_act=50,
                               max_idx_cmpd=10000,
                               min_active_size=10):
    print("Processing: ", f)
    dec_results = [f]
    dec_results.append(dataset)
    # Read data
    print(f'###{file_loc+f}###')
    data = decoy_utils.read_paired_file(file_loc + f)
    # Filter dupes and actives that are too small
    dec_results.append(len(set([d[0] for d in data])))
    seen = set()
    data = [
        d for d in data
        if Chem.MolFromSmiles(d[0]).GetNumHeavyAtoms() > min_active_size
    ]
    unique_data = [
        x for x in data if not (tuple(x) in seen or seen.add(tuple(x)))
    ]
    in_mols = [d[0] for d in data]
    gen_mols = [d[1] for d in data]
    dec_results.extend([len(set(in_mols)), len(data), len(unique_data)])

    # Calculate properties of in_mols and gen_mols
    used = set([])
    in_mols_set = [
        x for x in in_mols if x not in used and (used.add(x) or True)
    ]

    if dataset == "dude_ext":
        in_props_temp = decoy_utils.calc_dataset_props_dude_extended(
            in_mols_set, verbose=True)
        gen_props = decoy_utils.calc_dataset_props_dude_extended(gen_mols,
                                                                 verbose=True)
    elif dataset == "dekois":
        in_props_temp = decoy_utils.calc_dataset_props_dekois(in_mols_set,
                                                              verbose=True)
        gen_props = decoy_utils.calc_dataset_props_dekois(gen_mols,
                                                          verbose=True)
    elif dataset == "MUV":
        in_props_temp = decoy_utils.calc_dataset_props_muv(in_mols_set,
                                                           verbose=True)
        gen_props = decoy_utils.calc_dataset_props_muv(gen_mols, verbose=True)
    elif dataset == "ALL":
        in_props_temp = decoy_utils.calc_dataset_props_all(in_mols_set,
                                                           verbose=True)
        gen_props = decoy_utils.calc_dataset_props_all(gen_mols, verbose=True)
    elif dataset == "dude":
        in_props_temp = decoy_utils.calc_dataset_props_dude(in_mols_set,
                                                            verbose=True)
        gen_props = decoy_utils.calc_dataset_props_dude(gen_mols, verbose=True)
    else:
        print("Incorrect dataset")
        exit()
    in_mols_temp = list(in_mols_set)  # copy
    in_props = []
    for i, smi in enumerate(in_mols):
        in_props.append(in_props_temp[in_mols_temp.index(smi)])

    in_basic_temp = decoy_utils.calc_dataset_props_basic(in_mols_set,
                                                         verbose=True)
    in_mols_temp = list(in_mols_set)  # copy
    in_basic = []
    for i, smi in enumerate(in_mols):
        in_basic.append(in_basic_temp[in_mols_temp.index(smi)])

    gen_basic_props = decoy_utils.calc_dataset_props_basic(gen_mols,
                                                           verbose=True)

    # Scale properties based on in_mols props
    active_props_scaled_all = []
    decoy_props_scaled_all = []

    active_min_all = []
    active_max_all = []
    active_scale_all = []

    active_props = in_props_temp
    # Exclude errors from min/max calc
    act_prop = np.array(active_props)

    active_maxes = np.amax(act_prop, axis=0)
    active_mins = np.amin(act_prop, axis=0)

    active_max_all.append(active_maxes)
    active_min_all.append(active_mins)

    scale = []
    for (a_max, a_min) in zip(active_maxes, active_mins):
        if a_max != a_min:
            scale.append(a_max - a_min)
        else:
            scale.append(a_min)
    scale = np.array(scale)
    scale[scale == 0.0] = 1.0
    active_scale_all.append(scale)
    active_props_scaled = (active_props - active_mins) / scale
    active_props_scaled_all.append(active_props_scaled)

    # Calc SA scores
    in_sa_temp = [
        sascorer.calculateScore(Chem.MolFromSmiles(mol)) for mol in set(in_mols)
    ]
    in_mols_temp = list(set(in_mols))
    in_sa = []
    for i, smi in enumerate(in_mols):
        in_sa.append(in_sa_temp[in_mols_temp.index(smi)])
    gen_sa_props = [
        sascorer.calculateScore(Chem.MolFromSmiles(mol)) for mol in gen_mols
    ]

    # Calc Morgan fingerprints
    in_fps = []
    for i, mol in enumerate(in_mols):
        in_fps.append(
            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(mol),
                                                  2,
                                                  nBits=1024))
    gen_fps = []
    for i, mol in enumerate(gen_mols):
        gen_fps.append(
            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(mol),
                                                  2,
                                                  nBits=1024))

    # Calc DG scores
    dg_scores, dg_ids = decoy_utils.dg_score_rev(set(in_mols), gen_mols)

    # Calc LADS scores
    lads_scores = decoy_utils.lads_score_v2(set(in_mols), gen_mols)

    # Construct dictionary of results
    results_dict = {}
    for i in range(len(in_mols)):
        # Get scaling
        in_props_scaled = (in_props[i] - active_min_all) / active_scale_all
        gen_props_scaled = (gen_props[i] - active_min_all) / active_scale_all
        prop_diff = np.linalg.norm(
            np.array(in_props_scaled) - np.array(gen_props_scaled))

        # Get basic props diff
        basic_diff = np.sum(
            abs(np.array(in_basic[i]) - np.array(gen_basic_props[i])))

        if in_mols[i] in results_dict:
            sim = DataStructs.TanimotoSimilarity(in_fps[i], gen_fps[i])
            results_dict[in_mols[i]].append([
                in_mols[i], gen_mols[i], in_props[i], gen_props[i], prop_diff,
                sim, basic_diff,
                abs(gen_sa_props[i] - in_sa[i]), dg_scores[i], lads_scores[i]
            ])
        else:
            sim = DataStructs.TanimotoSimilarity(in_fps[i], gen_fps[i])
            results_dict[in_mols[i]] = [[
                in_mols[i], gen_mols[i], in_props[i], gen_props[i], prop_diff,
                sim, basic_diff,
                abs(gen_sa_props[i] - in_sa[i]), dg_scores[i], lads_scores[i]
            ]]

    # Get decoy matches
    results = []
    results_success_only = []
    sorted_mols_success = []
    for key in results_dict:
        # Set initial criteria - Note most of these are relatively weak
        prop_max_diff = 5
        max_basic_diff = 3
        max_sa_diff = 1.51
        max_dg_score = 0.35
        max_lads_score = 5
        while True:
            count_success = sum([
                i[4] < prop_max_diff and i[6] < max_basic_diff and
                i[7] < max_sa_diff and i[8] < max_dg_score and
                i[9] < max_lads_score for i in results_dict[key][0:max_idx_cmpd]
            ])
            # Adjust criteria if not enough successes
            if count_success < num_cand_dec_per_act:
                #print("Widening search", count_success)
                prop_max_diff *= 1.1
                max_basic_diff += 1
                max_sa_diff *= 1.1
                max_dg_score *= 1.1
                max_lads_score *= 1.1
            else:
                #print("Reached threshold", count_success)
                # Sort by sum of LADS and property difference (smaller better)
                sorted_mols_success.append([
                    (i[0], i[1], i[4], i[9], i[4] + i[9])
                    for i in sorted(results_dict[key][0:max_idx_cmpd],
                                    key=lambda i: i[4] + i[9],
                                    reverse=False)
                    if i[4] < prop_max_diff and i[6] < max_basic_diff and
                    i[7] < max_sa_diff and i[8] < max_dg_score and
                    i[9] < max_lads_score
                ])
                #assert count_success == len(sorted_mols_success[-1])
                break

    # Choose decoys
    active_mols_gen = []
    decoy_mols_gen = []

    embed_fails = 0
    dupes_wanted = 0
    for act_res in sorted_mols_success:
        count = 0
        # Greedy selection based on sum of LADS score and property difference (smaller better)
        for ent in act_res:
            # Check can gen conformer
            m = Chem.AddHs(Chem.MolFromSmiles(ent[1]))
            if AllChem.EmbedMolecule(m, randomSeed=42) != -1 and ent[
                    1] not in decoy_mols_gen:  # Check conf and not a decoy for another ligand
                decoy_mols_gen.append(ent[1])
                count += 1
                if count >= num_dec_per_act:
                    break
            elif ent[1] in decoy_mols_gen:
                dupes_wanted += 1
            else:
                embed_fails += 1
        active_mols_gen.append(act_res[0][0])
    #dec_results.extend([len(active_mols_gen), len(decoy_mols_gen), len(decoy_mols_gen)/num_dec_per_act, embed_fails, dupes_wanted])

    # Calc props for chosen decoys
    if dataset == "dude_ext":
        actives_feat = decoy_utils.calc_dataset_props_dude_extended(
            active_mols_gen, verbose=True)
        decoys_feat = decoy_utils.calc_dataset_props_dude_extended(
            decoy_mols_gen, verbose=True)
    elif dataset == "dekois":
        actives_feat = decoy_utils.calc_dataset_props_dekois(active_mols_gen,
                                                             verbose=True)
        decoys_feat = decoy_utils.calc_dataset_props_dekois(decoy_mols_gen,
                                                            verbose=True)
    elif dataset == "MUV":
        actives_feat = decoy_utils.calc_dataset_props_muv(active_mols_gen,
                                                          verbose=True)
        decoys_feat = decoy_utils.calc_dataset_props_muv(decoy_mols_gen,
                                                         verbose=True)
    elif dataset == "ALL":
        actives_feat = decoy_utils.calc_dataset_props_all(active_mols_gen,
                                                          verbose=True)
        decoys_feat = decoy_utils.calc_dataset_props_all(decoy_mols_gen,
                                                         verbose=True)
    elif dataset == "dude":
        actives_feat = decoy_utils.calc_dataset_props_dude(active_mols_gen)
        decoys_feat = decoy_utils.calc_dataset_props_dude(decoy_mols_gen)
    else:
        print("Incorrect dataset")
        exit()

    # ML model performance
    try:
        dec_results.extend(
            list(
                decoy_utils.calc_xval_performance(actives_feat,
                                                  decoys_feat,
                                                  n_jobs=1)))
    except:
        dec_results.extend([-1, -1])
        print(
            "Unable to assess ML model prediction. Check there are sufficient active molecules if these metrics are desired."
        )

    # DEKOIS paper metrics (LADS, DOE, Doppelganger score)
    dec_results.append(decoy_utils.doe_score(actives_feat, decoys_feat))
    lads_scores = decoy_utils.lads_score_v2(active_mols_gen, decoy_mols_gen)
    dec_results.append(np.mean(lads_scores))
    dg_scores, dg_ids = decoy_utils.dg_score(active_mols_gen, decoy_mols_gen)
    dec_results.extend([np.mean(dg_scores), max(dg_scores)])

    # Save intermediate performance results in unique file
    #with open(output_loc+'results_'+f+'.csv', 'w') as csvfile:
    #    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #    writer.writerow(dec_results)

    # Save decoy mols
    output_name = output_loc + f.split('.')[0] + '-selected.smi'
    with open(output_name, 'w') as outfile:
        for i, smi in enumerate(decoy_mols_gen):
            outfile.write(active_mols_gen[i // num_dec_per_act] + ' ' + smi +
                          '\n')

    return dec_results


if __name__ == "__main__":
    # Parse args
    args = docopt(__doc__)
    #print(args)
    if not args.get('--data_path'):
        print("Please specify a valid file or directory.")
        exit()
    file_loc = args.get('--data_path')
    output_loc = args.get('--output_path')
    dataset = args.get('--dataset_name')
    num_dec_per_act = int(args.get('--num_decoys_per_active'))
    num_cand_dec_per_act = int(args.get('--min_num_candidates'))
    n_cores = int(args.get('--num_cores'))
    min_active_size = int(args.get('--min_active_size'))
    max_idx_cmpd = int(args.get('--max_idx_cmpd'))

    # Get input files
    if os.path.isdir(file_loc):
        res_files = [
            f for f in os.listdir(file_loc) if os.path.isfile(file_loc + f)
        ]
        res_files = sorted(res_files)
    elif os.path.isfile(file_loc):
        res_files = [file_loc]
        file_loc = './'
    else:
        print("Please specify a valid file or directory")
        exit()

    # Declare metric variables
    columns = [
        'File name',
        'Dataset',
        'Orig num actives',
        'Num actives',
        'Num generated mols',
        'Num unique gen mols',
        'AUC ROC - 1NN',
        'AUC ROC - RF',
        'DOE score',
        'LADS score',
        'Doppelganger score mean',
        'Doppelganger score max',
    ]

    # Populate CSV file with headers
    with open(output_loc + '/results.csv', 'w') as csvfile:
        writer = csv.writer(csvfile,
                            delimiter=',',
                            quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(columns)

    # Select decoys and evaluate
    with Parallel(n_jobs=n_cores, backend='multiprocessing') as parallel:
        results = parallel(
            delayed(select_and_evaluate_decoys)(
                f,
                file_loc=file_loc,
                output_loc=output_loc,
                dataset=dataset,
                num_cand_dec_per_act=num_cand_dec_per_act,
                num_dec_per_act=num_dec_per_act,
                max_idx_cmpd=max_idx_cmpd,
                min_active_size=min_active_size) for f in res_files)

    # Write performance results
    for dec_results in results:
        with open(output_loc + '/results.csv', 'a') as csvfile:
            writer = csv.writer(csvfile,
                                delimiter=',',
                                quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
            writer.writerow(dec_results)

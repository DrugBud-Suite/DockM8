# IMPORTS 

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys

from itertools import chain
from os import listdir
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

import sklearn
from sklearn import datasets

from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold

import sascorer

# CALC FUNCTIONS
def calc_props_dude(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)

        # Calculate properties and store in dict
        prop_dict = {}
        # molweight
        prop_dict.update({'mol_wg': Descriptors.MolWt(mol)})
        # logP
        prop_dict.update({'log_p': Chem.Crippen.MolLogP(mol)})
        # HBA
        prop_dict.update({'hba': Chem.rdMolDescriptors.CalcNumLipinskiHBA(mol)})
        # HBD
        prop_dict.update({'hbd': Chem.rdMolDescriptors.CalcNumLipinskiHBD(mol)})
        # rotatable bonds
        prop_dict.update({'rot_bnds': Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)})
        # Formal (net) charge
        prop_dict.update({'net_charge': Chem.rdmolops.GetFormalCharge(mol)})

        prop_array = [prop_dict['mol_wg'], prop_dict['log_p'], prop_dict['hba'],
                      prop_dict['hbd'], prop_dict['rot_bnds'], prop_dict['net_charge']]

        return (prop_dict, prop_array)

    except:
        return ({}, [0, 0, 0, 0, 0, 0])


def calc_props_dude_extended(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)

        # Calculate properties and store in dict
        prop_dict = {}
        # molweight
        prop_dict.update({'mol_wg': Descriptors.MolWt(mol)})
        # logP
        prop_dict.update({'log_p': Chem.Crippen.MolLogP(mol)})
        # HBA
        prop_dict.update({'hba': Chem.rdMolDescriptors.CalcNumLipinskiHBA(mol)})
        # HBD
        prop_dict.update({'hbd': Chem.rdMolDescriptors.CalcNumLipinskiHBD(mol)})
        # ring count
        prop_dict.update({'ring_ct': Chem.rdMolDescriptors.CalcNumRings(mol)})
        # rotatable bonds
        prop_dict.update({'rot_bnds': Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)})
        # Formal (net) charge
        prop_dict.update({'net_charge': Chem.rdmolops.GetFormalCharge(mol)})
        # Topological polar surface area
        prop_dict.update({'tpsa': Chem.rdMolDescriptors.CalcTPSA(mol)})
        # Stereo centers
        prop_dict.update({'stereo_cnts': len(Chem.FindMolChiralCenters(mol,force=True,includeUnassigned=True))})

        prop_array = [prop_dict['mol_wg'], prop_dict['log_p'], prop_dict['hba'],
                      prop_dict['hbd'], prop_dict['ring_ct'], prop_dict['rot_bnds'],
                      prop_dict['net_charge'], prop_dict['tpsa'], prop_dict['stereo_cnts']]

        return (prop_dict, prop_array)

    except:
        return ({}, [-10, -10, -10, -10, -10, -10, -10, -10, -10])


def calc_props_basic(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)

        props = []
        # num heavy atoms
        props.append(mol.GetNumHeavyAtoms())
        # num carbons
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#6]"), maxMatches=mol.GetNumAtoms())))
        # num nitrogen
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#7]"), maxMatches=mol.GetNumAtoms())))
        # num oxygen
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#8]"), maxMatches=mol.GetNumAtoms())))
        # num fluorine
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#9]"), maxMatches=mol.GetNumAtoms())))
        # num sulfur
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#16]"), maxMatches=mol.GetNumAtoms())))
        # num chlorine
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#17]"), maxMatches=mol.GetNumAtoms())))

        return props

    except:
        return [0, 0, 0, 0, 0, 0, 0]


def calc_props_muv(smiles):
    try:
        # Create RDKit mol
        mol = Chem.MolFromSmiles(smiles)

        props = []
        # num atoms (incl. H)
        props.append(mol.GetNumAtoms(onlyExplicit=False))
        # num heavy atoms
        props.append(mol.GetNumHeavyAtoms())
        # num boron
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#5]"), maxMatches=mol.GetNumAtoms())))
        # num carbons
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#6]"), maxMatches=mol.GetNumAtoms())))
        # num nitrogen
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#7]"), maxMatches=mol.GetNumAtoms())))
        # num oxygen
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#8]"), maxMatches=mol.GetNumAtoms())))
        # num fluorine
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#9]"), maxMatches=mol.GetNumAtoms())))
        # num phosphorus
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#15]"), maxMatches=mol.GetNumAtoms())))
        # num sulfur
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#16]"), maxMatches=mol.GetNumAtoms())))
        # num chlorine
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#17]"), maxMatches=mol.GetNumAtoms())))
        # num bromine
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#35]"), maxMatches=mol.GetNumAtoms())))
        # num iodine
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#53]"), maxMatches=mol.GetNumAtoms())))

        # logP
        props.append(Chem.Crippen.MolLogP(mol))
        # HBA
        props.append(Chem.rdMolDescriptors.CalcNumLipinskiHBA(mol))
        # HBD
        props.append(Chem.rdMolDescriptors.CalcNumLipinskiHBD(mol))
        # ring count
        props.append(Chem.rdMolDescriptors.CalcNumRings(mol))
        # Stereo centers
        props.append(len(Chem.FindMolChiralCenters(mol,force=True,includeUnassigned=True)))

        return props

    except:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

from rdkit.Chem import rdPartialCharges
def calc_partial_charges(mol):
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
    positive_charge, negative_charge = 0, 0
    for atom in mol.GetAtoms():
        charge = float(atom.GetProp("_GasteigerCharge"))
        positive_charge += max(charge, 0)
        negative_charge -= min(charge, 0)

    return positive_charge, negative_charge

def calc_charges(mol):
    positive_charge, negative_charge = 0, 0
    for atom in mol.GetAtoms():
        charge = float(atom.GetFormalCharge())
        positive_charge += max(charge, 0)
        negative_charge -= min(charge, 0)

    return positive_charge, negative_charge
    
def calc_props_dekois(smiles):
    # Create RDKit mol
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        # Calculate properties and store in dict
        prop_dict = {}
        # molweight
        prop_dict.update({'mol_wg': Descriptors.MolWt(mol)})
        # logP
        prop_dict.update({'log_p': Chem.Crippen.MolLogP(mol)})
        # HBA
        prop_dict.update({'hba': Chem.rdMolDescriptors.CalcNumLipinskiHBA(mol)})
        # HBD
        prop_dict.update({'hbd': Chem.rdMolDescriptors.CalcNumLipinskiHBD(mol)})
        # aromatic ring count
        prop_dict.update({'ring_ct': Chem.rdMolDescriptors.CalcNumAromaticRings(mol)})
        # rotatable bonds
        prop_dict.update({'rot_bnds': Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)})
        # Formal charges
        pos, neg = calc_charges(mol)
        prop_dict.update({'pos_charge': pos})
        prop_dict.update({'neg_charge': neg})

        prop_array = [prop_dict['mol_wg'], prop_dict['log_p'], prop_dict['hba'],
                      prop_dict['hbd'], prop_dict['ring_ct'], prop_dict['rot_bnds'],
                      prop_dict['pos_charge'], prop_dict['neg_charge']]

        return (prop_dict, prop_array)

    except:
        return ({}, [0, 0, 0, 0, 0, 0, 0, 0])
    
def calc_props_all(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)

        props = []
             
        ### MUV properties ###
        # num atoms (incl. H)
        props.append(mol.GetNumAtoms(onlyExplicit=False))
        # num heavy atoms
        props.append(mol.GetNumHeavyAtoms())
        # num boron
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#5]"), maxMatches=mol.GetNumAtoms())))
        # num carbons
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#6]"), maxMatches=mol.GetNumAtoms())))
        # num nitrogen
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#7]"), maxMatches=mol.GetNumAtoms())))
        # num oxygen
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#8]"), maxMatches=mol.GetNumAtoms())))
        # num fluorine
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#9]"), maxMatches=mol.GetNumAtoms())))
        # num phosphorus
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#15]"), maxMatches=mol.GetNumAtoms())))
        # num sulfur
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#16]"), maxMatches=mol.GetNumAtoms())))
        # num chlorine
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#17]"), maxMatches=mol.GetNumAtoms())))
        # num bromine
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#35]"), maxMatches=mol.GetNumAtoms())))
        # num iodine
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#53]"), maxMatches=mol.GetNumAtoms())))

        # logP
        props.append(Chem.Crippen.MolLogP(mol))
        # HBA
        props.append(Chem.rdMolDescriptors.CalcNumLipinskiHBA(mol))
        # HBD
        props.append(Chem.rdMolDescriptors.CalcNumLipinskiHBD(mol))
        # ring count
        props.append(Chem.rdMolDescriptors.CalcNumRings(mol))
        # Stereo centers
        props.append(len(Chem.FindMolChiralCenters(mol,force=True,includeUnassigned=True)))
        
        ### DEKOIS properties (additional) ###
        # molweight
        props.append(Descriptors.MolWt(mol))
        # aromatic ring count
        props.append(Chem.rdMolDescriptors.CalcNumAromaticRings(mol))
        # rotatable bonds
        props.append(Chem.rdMolDescriptors.CalcNumRotatableBonds(mol))
        # Pos, neg charges
        pos, neg = calc_charges(mol)
        props.append(pos)
        props.append(neg)
        
        ### DUD-E extended (additional) ###
        # Formal (net) charge
        props.append(Chem.rdmolops.GetFormalCharge(mol))
        # Topological polar surface area
        props.append(Chem.rdMolDescriptors.CalcTPSA(mol))
        
        ### Additional
        # QED
        props.append(Chem.QED.qed(mol))
        # SA score
        props.append(sascorer.calculateScore(mol))
        
        return props

    except:
        return [0]*27

def calc_dataset_props_all(dataset, verbose=False):
    # Calculate mol props for an entire dataset of smiles strings
    results = []
    for i, smiles in enumerate(dataset):
        props = calc_props_all(smiles)
        if props is not None:
            results.append(props)
        if verbose and i % 1000 == 0:
            print("\rProcessed smiles: " + str(i), end='')
    print("\nDone calculating properties")
    return np.array(results)
    
def calc_dataset_props_dude(dataset, verbose=False):
    # Calculate mol props for an entire dataset of smiles strings
    results = []
    for i, smiles in enumerate(dataset):
        props = calc_props_dude(smiles)
        if props is not None:
            results.append(props[1])
        if verbose and i % 1000 == 0:
            print("\rProcessed smiles: " + str(i), end='')
    print("\nDone calculating properties")
    return np.array(results)

def calc_dataset_props_dude_extended(dataset, verbose=False):
    # Calculate mol props for an entire dataset of smiles strings
    results = []
    for i, smiles in enumerate(dataset):
        props = calc_props_dude_extended(smiles)
        if props is not None:
            results.append(props[1])
        if verbose and i % 1000 == 0:
            print("\rProcessed smiles: " + str(i), end='')
    print("\nDone calculating properties")
    return np.array(results)

def calc_dataset_props_muv(dataset, verbose=False):
    # Calculate mol props for an entire dataset of smiles strings
    results = []
    for i, smiles in enumerate(dataset):
        props = calc_props_muv(smiles)
        if props is not None:
            results.append(props)
        if verbose and i % 1000 == 0:
            print("\rProcessed smiles: " + str(i), end='')
    print("\nDone calculating properties")
    return np.array(results)

def calc_dataset_props_basic(dataset, verbose=False):
    # Calculate mol props for an entire dataset of smiles strings
    results = []
    for i, smiles in enumerate(dataset):
        props = calc_props_basic(smiles)
        if props is not None:
            results.append(props)
        if verbose and i % 1000 == 0:
            print("\rProcessed smiles: " + str(i), end='')
    print("\nDone calculating properties")
    return np.array(results)

def calc_dataset_props_dekois(dataset, verbose=False):
    # Calculate mol props for an entire dataset of smiles strings
    results = []
    for i, smiles in enumerate(dataset):
        props = calc_props_dekois(smiles)
        if props is not None:
            results.append(props[1])
        if verbose and i % 1000 == 0:
            print("\rProcessed smiles: " + str(i), end='')
    print("\nDone calculating properties")
    return np.array(results)

def doe_score(actives, decoys):
    all_feat = list(actives) + list(decoys)
    up_p = np.percentile(all_feat, 95, axis=0)
    low_p = np.percentile(all_feat, 5, axis=0)
    norms = up_p - low_p
    for i in range(len(norms)):
        if norms[i] == 0:
            norms[i] = 1.

    active_norm = [act/norms for act in actives]
    decoy_norm = [dec/norms for dec in decoys]
    all_norm = active_norm + decoy_norm

    active_embed = []
    labels = [1] * (len(active_norm)-1) + [0] * len(decoy_norm)
    for i, act in enumerate(active_norm):
        comp = list(all_norm)
        del comp[i]
        dists = [100 - np.linalg.norm(c-act) for c in comp] # arbitrary large number to get scores in reverse order
        fpr, tpr, _ = roc_curve(labels, dists)
        fpr = fpr[::]
        tpr = tpr[::]
        a_score = 0
        for i in range(len(fpr)-1):
            a_score += (abs(0.5*( (tpr[i+1]+tpr[i])*(fpr[i+1]-fpr[i]) - (fpr[i+1]+fpr[i])*(fpr[i+1]-fpr[i]) )))
        active_embed.append(a_score)

    #print(np.average(active_embed))
    return np.average(active_embed)

from collections import defaultdict
def lads_score_v2(actives, decoys):
    # Similar to DEKOIS (v2)
    # Lower is better (less like actives), higher is worse (more like actives)
    active_fps = []
    active_info = {}
    info={}
    atoms_per_bit = defaultdict(int)
    for smi in actives:
        m = Chem.MolFromSmiles(smi)
        active_fps.append(AllChem.GetMorganFingerprint(m,3,useFeatures=True, bitInfo=info))
        for key in info:
            if key not in active_info:
                active_info[key] = info[key]
                env = Chem.FindAtomEnvironmentOfRadiusN(m, info[key][0][1], info[key][0][0])
                amap={}
                submol=Chem.PathToSubmol(m,env,atomMap=amap)
                if info[key][0][1] == 0:
                    atoms_per_bit[key] = 1
                else:
                    atoms_per_bit[key] = submol.GetNumHeavyAtoms()

    decoys_fps = [AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi),3,useFeatures=True) for smi in decoys] # Roughly FCFP_6

    master_active_fp_freq = defaultdict(int)
    for fp in active_fps:
        fp_dict = fp.GetNonzeroElements()
        for k, v in fp_dict.items():
            master_active_fp_freq[k] += 1
    # Reweight
    for k in master_active_fp_freq:
        # Normalise
        master_active_fp_freq[k] /= len(active_fps)
        # Weight by size of bit
        master_active_fp_freq[k] *= atoms_per_bit[k]

    decoys_lads_avoid_scores = [sum([master_active_fp_freq[k] for k in decoy_fp.GetNonzeroElements()])/len(decoy_fp.GetNonzeroElements()) 
                                for decoy_fp in decoys_fps]
    
    return decoys_lads_avoid_scores

def dg_score(actives, decoys):
    # Similar to DEKOIS
    # Lower is better (less like actives), higher is worse (more like actives)
    active_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi),3,useFeatures=True) for smi in actives] # Roughly FCFP_6
    decoys_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi),3,useFeatures=True) for smi in decoys] # Roughly FCFP_6

    closest_sims = []
    closest_sims_id = []
    for active_fp in active_fps:
        active_sims = []
        for decoy_fp in decoys_fps:
            active_sims.append(DataStructs.TanimotoSimilarity(active_fp, decoy_fp))
        closest_sims.append(max(active_sims))
        closest_sims_id.append(np.argmax(active_sims))

    return closest_sims, closest_sims_id

def dg_score_rev(actives, decoys):
    # Similar to DEKOIS
    # Lower is better (less like actives), higher is worse (more like actives)
    active_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi),3,useFeatures=True) for smi in actives] # Roughly FCFP_6
    decoys_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi),3,useFeatures=True) for smi in decoys] # Roughly FCFP_6

    closest_sims = []
    closest_sims_id = []
    for decoy_fp in decoys_fps:
        active_sims = []
        for active_fp in active_fps:
            active_sims.append(DataStructs.TanimotoSimilarity(active_fp, decoy_fp))
        closest_sims.append(max(active_sims))
        closest_sims_id.append(np.argmax(active_sims))

    return closest_sims, closest_sims_id

# LOAD ZINC DATA
def read_file(filename):
    '''Reads .smi file '''
    '''Returns array containing smiles strings of molecules'''
    smiles, names = [], []
    with open(filename, 'r') as f:
        for line in f:
            if line:
                smiles.append(line.strip().split(' ')[0])
    return smiles

def read_paired_file(filename):
    '''Reads .smi file '''
    '''Returns array containing smiles strings of molecules'''
    smiles, names = [], []
    with open(filename, 'r') as f:
        for line in f:
            if line:
                smiles.append(line.strip().split(' ')[0:2])
    return smiles

def read_paired_dude_file(filename):
    '''Reads .smi file '''
    '''Returns array containing smiles strings of molecules'''
    smiles, names = [], []
    with open(filename, 'r') as f:
        for line in f:
            if line:
                toks = line.strip().split(' ')
                smiles.append([toks[0], toks[-1]])
    return smiles

# Calculate number of macrocycles generated
def num_macro(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        # Cyles with >7 atoms
        ri = mol.GetRingInfo()
        nMacrocycles = 0
        for x in ri.AtomRings():
            if len(x) > 7:
                nMacrocycles += 1
        return nMacrocycles
    except:
        print(smi)
        return 0

# Calculate number of unique
def unique(test_data):
    original_num = len(test_data)
    test_data = set(test_data)
    new_num = len(test_data)
    return new_num/float(original_num)

def calc_xval_performance(active_props, decoy_props, random_state=42, n_jobs=-1):
    # Scale properties based on active props
    active_min_all = []
    active_max_all = []
    active_scale_all = []

    # Exclude errors from min/max calc
    act_prop = np.array(active_props)

    active_maxes = np.amax(act_prop, axis=0)
    active_mins = np.amin(act_prop, axis=0)

    active_max_all.append(active_maxes)
    active_min_all.append(active_mins)

    scale = []
    for (a_max, a_min) in zip(active_maxes,active_mins):
        if a_max != a_min:
            scale.append(a_max - a_min)
        else:
            scale.append(a_min)
    scale = np.array(scale)
    scale[scale == 0.0] = 1.0
    active_scale_all.append(scale)

    # Scale
    in_props_scaled = [(active_feat - active_min_all) / active_scale_all for active_feat in active_props]
    gen_props_scaled = [(gen_feat - active_min_all) / active_scale_all for gen_feat in decoy_props]

    # Features
    X = np.squeeze(np.concatenate((np.array(in_props_scaled), np.array(gen_props_scaled))), axis=1)
    # Labels
    y = np.array([1]*len(active_props) + [0]*len(decoy_props))

    # Construct x-val folds
    strat_kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=42)

    # Loop over K folds
    perf_1nn = []
    perf_rf = []
    for train_idx, test_idx in strat_kfold.split(X, y):
        # Train and test model
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Perform 1NN analysis
        neigh = KNeighborsClassifier(n_neighbors=1)
        neigh.fit(X_train, y_train)
        test_probs = neigh.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, test_probs)
        perf_1nn.append(auc(fpr, tpr))

        # Perform RF analysis
        rf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=n_jobs)
        rf.fit(X_train, y_train)
        test_probs = rf.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, test_probs)
        perf_rf.append(auc(fpr, tpr))

    return (np.mean(perf_1nn), np.mean(perf_rf))

### Dataset info #####
def dataset_info(dataset): #qm9, zinc, cep
    if dataset=='qm9':
        return { 'atom_types': ["H", "C", "N", "O", "F"],
                 'maximum_valence': {0: 1, 1: 4, 2: 3, 3: 2, 4: 1},
                 'number_to_atom': {0: "H", 1: "C", 2: "N", 3: "O", 4: "F"},
                 'bucket_sizes': np.array(list(range(4, 28, 2)) + [29])
               }
    elif dataset=='zinc':
        return { 'atom_types': ['Br1(0)', 'C4(0)', 'Cl1(0)', 'F1(0)', 'H1(0)', 'I1(0)',
                'N2(-1)', 'N3(0)', 'N4(1)', 'O1(-1)', 'O2(0)', 'S2(0)','S4(0)', 'S6(0)'],
                 'maximum_valence': {0: 1, 1: 4, 2: 1, 3: 1, 4: 1, 5:1, 6:2, 7:3, 8:4, 9:1, 10:2, 11:2, 12:4, 13:6, 14:3},
                 'number_to_atom': {0: 'Br', 1: 'C', 2: 'Cl', 3: 'F', 4: 'H', 5:'I', 6:'N', 7:'N', 8:'N', 9:'O', 10:'O', 11:'S', 12:'S', 13:'S'},
                 'bucket_sizes': np.array([28,31,33,35,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,53,55,58,84])
               }

    elif dataset=="cep":
        return { 'atom_types': ["C", "S", "N", "O", "Se", "Si"],
                 'maximum_valence': {0: 4, 1: 2, 2: 3, 3: 2, 4: 2, 5: 4},
                 'number_to_atom': {0: "C", 1: "S", 2: "N", 3: "O", 4: "Se", 5: "Si"},
                 'bucket_sizes': np.array([25,28,29,30, 32, 33,34,35,36,37,38,39,43,46])
               }
    else:
        print("the datasets in use are qm9|zinc|cep")
        exit(1)

##### Check data #####
def check_smi_atom_types(smi, dataset='zinc', verbose=False):
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        valence = atom.GetTotalValence()
        charge = atom.GetFormalCharge()
        atom_str = "%s%i(%i)" % (symbol, valence, charge)

        if atom_str not in dataset_info(dataset)['atom_types']:
            if "*" in atom_str:
                continue
            else:
                if verbose:
                    print('unrecognized atom type %s' % atom_str)
                return False
    return True

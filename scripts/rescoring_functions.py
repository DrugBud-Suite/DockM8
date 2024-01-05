import os
import subprocess
import time
import warnings
from pathlib import Path
from subprocess import DEVNULL, STDOUT
from typing import List

import pandas as pd
from pandas import DataFrame
from rdkit.Chem import PandasTools
from rdkit import RDLogger
from tqdm import tqdm

from scripts.utilities import delete_files, parallel_executor, printlog, split_sdf, split_sdf_str, convert_molecules

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def gnina_rescoring(sdf: str, ncpus: int, column_name: str, **kwargs):
    """
    Performs rescoring of ligand poses using the gnina software package. The function splits the input SDF file into
    smaller files, and then runs gnina on each of these files in parallel. The results are then combined into a single
    Pandas dataframe and saved to a CSV file.

    Args:
        sdf (str): The path to the input SDF file.
        ncpus (int): The number of CPUs to use for parallel execution.
        column_name (str): The name of the column in the output dataframe that will contain the rescoring results.

    Returns:
        A Pandas dataframe containing the rescoring results.
    """
    tic = time.perf_counter()
    rescoring_folder = kwargs.get('rescoring_folder')
    software = kwargs.get('software')
    protein_file = kwargs.get('protein_file')
    pocket_definition = kwargs.get('pocket_definition')
    cnn = 'crossdock_default2018'
    split_files_folder = split_sdf_str(rescoring_folder / f'{column_name}_rescoring', sdf, ncpus)
    split_files_sdfs = [split_files_folder / f for f in os.listdir(split_files_folder) if f.endswith('.sdf')]

    global gnina_rescoring_splitted

    def gnina_rescoring_splitted(split_file, protein_file, pocket_definition):
        gnina_folder = rescoring_folder / f'{column_name}_rescoring'
        results = gnina_folder / f'{Path(split_file).stem}_{column_name}.sdf'
        gnina_cmd = (
            f'{software}/gnina'
            f' --receptor {protein_file}'
            f' --ligand {split_file}'
            f' --out {results}'
            f' --center_x {pocket_definition["center"][0]}'
            f' --center_y {pocket_definition["center"][1]}'
            f' --center_z {pocket_definition["center"][2]}'
            f' --size_x {pocket_definition["size"][0]}'
            f' --size_y {pocket_definition["size"][1]}'
            f' --size_z {pocket_definition["size"][2]}'
            ' --cpu 1'
            ' --score_only'
            f' --cnn {cnn} --no_gpu'
        )
        try:
            subprocess.call(gnina_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
        except Exception as e:
            printlog(f'{column_name} rescoring failed: ' + e)
        return

    parallel_executor(gnina_rescoring_splitted, split_files_sdfs, ncpus, protein_file=protein_file, pocket_definition=pocket_definition)
    
    try:
        gnina_dataframes = [PandasTools.LoadSDF(str(rescoring_folder / f'{column_name}_rescoring' / file),  idName='Pose ID', molColName=None, includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True) for file in os.listdir(rescoring_folder / f'{column_name}_rescoring') if file.startswith('split') and file.endswith('.sdf')]
    except Exception as e:
        printlog(f'ERROR: Failed to Load {column_name} rescoring SDF file!')
        printlog(e)
    try:
        gnina_rescoring_results = pd.concat(gnina_dataframes)
    except Exception as e:
        printlog(f'ERROR: Could not combine {column_name} rescored poses')
        printlog(e)
    gnina_rescoring_results.rename(columns={'minimizedAffinity': 'GNINA-Affinity',
                                            'CNNscore': 'CNN-Score',
                                            'CNNaffinity': 'CNN-Affinity'},
                                            inplace=True)
    gnina_rescoring_results = gnina_rescoring_results[['Pose ID', column_name]]
    gnina_scores_path = rescoring_folder / f'{column_name}_rescoring' / f'{column_name}_scores.csv'
    gnina_rescoring_results.to_csv(gnina_scores_path, index=False)
    delete_files(rescoring_folder / f'{column_name}_rescoring', f'{column_name}_scores.csv')
    toc = time.perf_counter()
    printlog(f'Rescoring with {column_name} complete in {toc - tic:0.4f}!')
    return gnina_rescoring_results

def vinardo_rescoring(sdf: str, ncpus: int, column_name: str, **kwargs) -> DataFrame:
    """
    Performs rescoring of poses using the Vinardo scoring function.

    Args:
        sdf (str): The path to the input SDF file containing the poses to be rescored.
        ncpus (int): The number of CPUs to be used for the rescoring process.
        column_name (str): The name of the column in the output dataframe to store the Vinardo scores.
        **kwargs: Additional keyword arguments for rescoring.

    Keyword Args:
        rescoring_folder (str): The path to the folder for storing the Vinardo rescoring results.
        software (str): The path to the gnina software.
        protein_file (str): The path to the protein file.
        pocket_definition (dict): The pocket definition.

    Returns:
        DataFrame: A dataframe containing the 'Pose ID' and Vinardo score columns for the rescored poses.
    """
    tic = time.perf_counter()

    rescoring_folder = kwargs.get('rescoring_folder')
    software = kwargs.get('software')
    protein_file = kwargs.get('protein_file')
    pocket_definition = kwargs.get('pocket_definition')
    
    split_files_folder = split_sdf_str(rescoring_folder / f'{column_name}_rescoring', sdf, ncpus)
    split_files_sdfs = [split_files_folder / f for f in os.listdir(split_files_folder) if f.endswith('.sdf')]

    vinardo_rescoring_folder = rescoring_folder / f'{column_name}_rescoring'
    vinardo_rescoring_folder.mkdir(parents=True, exist_ok=True)

    global vinardo_rescoring_splitted
    def vinardo_rescoring_splitted(split_file, protein_file, pocket_definition):
        vinardo_rescoring_folder = rescoring_folder / f'{column_name}_rescoring'
        results = vinardo_rescoring_folder / f'{Path(split_file).stem}_{column_name}.sdf'
        vinardo_cmd = (
            f"{software}/gnina"
            f" --receptor {protein_file}"
            f" --ligand {split_file}"
            f" --out {results}"
            f" --center_x {pocket_definition['center'][0]}"
            f" --center_y {pocket_definition['center'][1]}"
            f" --center_z {pocket_definition['center'][2]}"
            f" --size_x {pocket_definition['size'][0]}"
            f" --size_y {pocket_definition['size'][1]}"
            f" --size_z {pocket_definition['size'][2]}"
            " --score_only"
            " --scoring vinardo"
            " --cnn_scoring none"
        )
        try:
            subprocess.call(vinardo_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
        except Exception as e:
            printlog(f'{column_name} rescoring failed: ' + e)
        return

    parallel_executor(vinardo_rescoring_splitted, split_files_sdfs, ncpus, protein_file=protein_file, pocket_definition=pocket_definition)
    
    try:
        vinardo_dataframes = [PandasTools.LoadSDF(str(rescoring_folder / f'{column_name}_rescoring' / file),  idName='Pose ID', molColName=None, includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True) for file in os.listdir(rescoring_folder / f'{column_name}_rescoring') if file.startswith('split') and file.endswith('.sdf')]
    except Exception as e:
        printlog(f'ERROR: Failed to Load {column_name} rescoring SDF file!')
        printlog(e)
    try:
        vinardo_rescoring_results = pd.concat(vinardo_dataframes)
    except Exception as e:
        printlog(f'ERROR: Could not combine {column_name} rescored poses')
        printlog(e)
    vinardo_rescoring_results.rename(columns={'minimizedAffinity': column_name}, inplace=True)
    vinardo_rescoring_results = vinardo_rescoring_results[['Pose ID', column_name]]
    vinardo_scores_path = vinardo_rescoring_folder / f'{column_name}_scores.csv'
    vinardo_rescoring_results.to_csv(vinardo_scores_path, index=False)
    delete_files(vinardo_rescoring_folder, f'{column_name}_scores.csv')
    toc = time.perf_counter()
    printlog(f'Rescoring with Vinardo complete in {toc - tic:0.4f}!')
    return vinardo_rescoring_results

def AD4_rescoring(sdf: str, ncpus: int, column_name: str, **kwargs) -> DataFrame:
    """
    Performs rescoring of poses using the AutoDock4 (AD4) scoring function.

    Args:
        sdf (str): The path to the input SDF file containing the poses to be rescored.
        ncpus (int): The number of CPUs to be used for the rescoring process.
        column_name (str): The name of the column in the output dataframe to store the AD4 scores.
        kwargs: Additional keyword arguments including rescoring_folder, software, protein_file, and pocket_de.

    Returns:
        DataFrame: A dataframe containing the 'Pose ID' and AD4 score columns for the rescored poses.
    """
    tic = time.perf_counter()

    rescoring_folder = kwargs.get('rescoring_folder')
    software = kwargs.get('software')
    protein_file = kwargs.get('protein_file')
    pocket_definition = kwargs.get('pocket_definition')
    
    split_files_folder = split_sdf_str(rescoring_folder / f'{column_name}_rescoring', sdf, ncpus)
    split_files_sdfs = [split_files_folder / f for f in os.listdir(split_files_folder) if f.endswith('.sdf')]

    AD4_rescoring_folder = rescoring_folder / f'{column_name}_rescoring'
    AD4_rescoring_folder.mkdir(parents=True, exist_ok=True)

    global AD4_rescoring_splitted
    def AD4_rescoring_splitted(split_file, protein_file, pocket_definition):
        AD4_rescoring_folder = rescoring_folder / f'{column_name}_rescoring'
        results = AD4_rescoring_folder / f'{Path(split_file).stem}_{column_name}.sdf'
        AD4_cmd = (
            f"{software}/gnina"
            f" --receptor {protein_file}"
            f" --ligand {split_file}"
            f" --out {results}"
            f" --center_x {pocket_definition['center'][0]}"
            f" --center_y {pocket_definition['center'][1]}"
            f" --center_z {pocket_definition['center'][2]}"
            f" --size_x {pocket_definition['size'][0]}"
            f" --size_y {pocket_definition['size'][1]}"
            f" --size_z {pocket_definition['size'][2]}"
            " --score_only"
            " --scoring ad4_scoring"
            " --cnn_scoring none"
        )
        try:
            subprocess.call(AD4_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
        except Exception as e:
            printlog(f'{column_name} rescoring failed: ' + e)
        return

    parallel_executor(AD4_rescoring_splitted, split_files_sdfs, ncpus, protein_file=protein_file, pocket_definition=pocket_definition)
    
    try:
        AD4_dataframes = [PandasTools.LoadSDF(str(rescoring_folder / f'{column_name}_rescoring' / file),  idName='Pose ID', molColName=None, includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True) for file in os.listdir(rescoring_folder / f'{column_name}_rescoring') if file.startswith('split') and file.endswith('.sdf')]
    except Exception as e:
        printlog(f'ERROR: Failed to Load {column_name} rescoring SDF file!')
        printlog(e)
    try:
        AD4_rescoring_results = pd.concat(AD4_dataframes)
    except Exception as e:
        printlog(f'ERROR: Could not combine {column_name} rescored poses')
        printlog(e)

    AD4_rescoring_results.rename(columns={'minimizedAffinity': column_name}, inplace=True)
    AD4_rescoring_results = AD4_rescoring_results[['Pose ID', column_name]]
    AD4_scores_file = AD4_rescoring_folder / f'{column_name}_scores.csv'
    AD4_rescoring_results.to_csv(AD4_scores_file, index=False)
    delete_files(AD4_rescoring_folder, f'{column_name}_scores.csv')
    toc = time.perf_counter()
    printlog(f'Rescoring with AD4 complete in {toc-tic:0.4f}!')
    return AD4_rescoring_results

#def rfscorevs_rescoring(sdf : str, ncpus : int, column_name : str, **kwargs):
    """
    Rescores poses in an SDF file using RFScoreVS and returns the results as a pandas DataFrame.

    Args:
        sdf (str): Path to the SDF file containing the poses to be rescored.
        ncpus (int): Number of CPUs to use for the RFScoreVS calculation.
        column_name (str): Name of the column to be used for the RFScoreVS scores in the output DataFrame.
        kwargs: Additional keyword arguments.

    Keyword Args:
        rescoring_folder (str): Path to the folder for storing the RFScoreVS rescoring results.
        software (str): Path to the RFScoreVS software.
        protein_file (str): Path to the receptor protein file.
        pocket_de (dict): Dictionary containing pocket definitions.

    Returns:
        pandas.DataFrame: DataFrame containing the RFScoreVS scores for each pose in the input SDF file.
    """
    rescoring_folder = kwargs.get('rescoring_folder')
    software = kwargs.get('software')
    protein_file = kwargs.get('protein_file')
    
    tic = time.perf_counter()

    rfscorevs_rescoring_folder = rescoring_folder / f'{column_name}_rescoring'
    rfscorevs_rescoring_folder.mkdir(parents=True, exist_ok=True)
    
    split_files_folder = split_sdf_str(rescoring_folder / f'{column_name}_rescoring', sdf, ncpus)
    split_files_sdfs = [split_files_folder / f for f in os.listdir(split_files_folder) if f.endswith('.sdf')]
    global rf_score_vs_splitted
    def rf_score_vs_splitted(split_file, protein_file):
        rfscorevs_cmd = f'{software}/rf-score-vs --receptor {protein_file} {split_file} -O {rfscorevs_rescoring_folder / Path(split_file).stem}_RFScoreVS_scores.csv -n 1'
        subprocess.call(rfscorevs_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
        return
    
    parallel_executor(rf_score_vs_splitted, split_files_sdfs, ncpus, protein_file=protein_file)
    
    try:
        rfscorevs_dataframes = [pd.read_csv(rfscorevs_rescoring_folder / file, delimiter=',', header=0) for file in os.listdir(rfscorevs_rescoring_folder) if file.startswith('split') and file.endswith('.csv')]
        rfscorevs_results = pd.concat(rfscorevs_dataframes)
        rfscorevs_results.rename(columns={'name': 'Pose ID', 'RFScoreVS_v2': column_name}, inplace=True)
    except Exception as e:
        printlog('ERROR: Failed to process RFScoreVS results!')
        printlog(e)
    rfscorevs_results.to_csv(rescoring_folder / f'{column_name}_rescoring' / f'{column_name}_scores.csv', index=False)
    delete_files(rescoring_folder / f'{column_name}_rescoring', f'{column_name}_scores.csv')
    toc = time.perf_counter()
    printlog(f'Rescoring with RFScoreVS complete in {toc-tic:0.4f}!')
    return rfscorevs_results

def rfscorevs_rescoring(sdf : str, ncpus : int, column_name : str, **kwargs):
    """
    Rescores poses in an SDF file using RFScoreVS and returns the results as a pandas DataFrame.

    Args:
        sdf (str): Path to the SDF file containing the poses to be rescored.
        ncpus (int): Number of CPUs to use for the RFScoreVS calculation.
        column_name (str): Name of the column to be used for the RFScoreVS scores in the output DataFrame.
        kwargs: Additional keyword arguments.

    Keyword Args:
        rescoring_folder (str): Path to the folder for storing the RFScoreVS rescoring results.
        software (str): Path to the RFScoreVS software.
        protein_file (str): Path to the receptor protein file.
        pocket_de (dict): Dictionary containing pocket definitions.

    Returns:
        pandas.DataFrame: DataFrame containing the RFScoreVS scores for each pose in the input SDF file.
    """
    rescoring_folder = kwargs.get('rescoring_folder')
    software = kwargs.get('software')
    protein_file = kwargs.get('protein_file')
    
    tic = time.perf_counter()

    rfscorevs_rescoring_folder = rescoring_folder / f'{column_name}_rescoring'
    rfscorevs_rescoring_folder.mkdir(parents=True, exist_ok=True)
    
    rfscorevs_cmd = f'{software}/rf-score-vs --receptor {protein_file} {sdf} -O {rfscorevs_rescoring_folder / f"{column_name}_scores.csv -n {ncpus}"}'
    subprocess.call(rfscorevs_cmd, shell=True)
    
    try:
        rfscorevs_results = pd.read_csv(rfscorevs_rescoring_folder / f"{column_name}_scores.csv", delimiter=',', header=0)
        rfscorevs_results.rename(columns={'name': 'Pose ID', 'RFScoreVS_v2': column_name}, inplace=True)
    except Exception as e:
        printlog('ERROR: Failed to process RFScoreVS results!')
        printlog(e)
    rfscorevs_results.to_csv(rescoring_folder / f'{column_name}_rescoring' / f'{column_name}_scores.csv', index=False)
    delete_files(rescoring_folder / f'{column_name}_rescoring', f'{column_name}_scores.csv')
    toc = time.perf_counter()
    printlog(f'Rescoring with RFScoreVS complete in {toc-tic:0.4f}!')
    return rfscorevs_results

def plp_rescoring(sdf : str, ncpus : int, column_name : str, **kwargs):
    """
    Rescores ligands using PLP scoring function.

    Args:
    sdf (str): Path to the input SDF file.
    ncpus (int): Number of CPUs to use for docking.
    column_name (str): Name of the column to store the PLP scores.
    kwargs: Additional keyword arguments.

    Returns:
    pandas.DataFrame: DataFrame containing the Pose ID and PLP scores.
    """
    rescoring_folder = kwargs.get('rescoring_folder')
    software = kwargs.get('software')
    protein_file = kwargs.get('protein_file')
    pocket_definition = kwargs.get('pocket_definition')

    tic = time.perf_counter()

    plants_search_speed = 'speed1'
    ants = '20'
    plp_rescoring_folder = Path(rescoring_folder) / f'{column_name}_rescoring'
    plp_rescoring_folder.mkdir(parents=True, exist_ok=True)
    # Convert protein file to .mol2 using open babel
    plants_protein_mol2 = plp_rescoring_folder / 'protein.mol2'
    convert_molecules(protein_file, plants_protein_mol2, 'pdb', 'mol2')
    # Convert prepared ligand file to .mol2 using open babel
    plants_ligands_mol2 = plp_rescoring_folder / 'ligands.mol2'
    convert_molecules(sdf, plants_ligands_mol2, 'sdf', 'mol2')

    # Generate plants config file
    plp_rescoring_config_path_txt = plp_rescoring_folder / 'config.txt'
    plp_config = ['# search algorithm\n',
                    'search_speed ' + plants_search_speed + '\n',
                    'aco_ants ' + ants + '\n',
                    'flip_amide_bonds 0\n',
                    'flip_planar_n 1\n',
                    'force_flipped_bonds_planarity 0\n',
                    'force_planar_bond_rotation 1\n',
                    'rescore_mode simplex\n',
                    'flip_ring_corners 0\n',
                    '# scoring functions\n',
                    '# Intermolecular (protein-ligand interaction scoring)\n',
                    'scoring_function plp\n',
                    'outside_binding_site_penalty 50.0\n',
                    'enable_sulphur_acceptors 1\n',
                    '# Intramolecular ligand scoring\n',
                    'ligand_intra_score clash2\n',
                    'chemplp_clash_include_14 1\n',
                    'chemplp_clash_include_HH 0\n',

                    '# input\n',
                    'protein_file ' + str(plants_protein_mol2) + '\n',
                    'ligand_file ' + str(plants_ligands_mol2) + '\n',

                    '# output\n',
                    'output_dir ' + str(plp_rescoring_folder / 'results') + '\n',

                    '# write single mol2 files (e.g. for RMSD calculation)\n',
                    'write_multi_mol2 1\n',

                    '# binding site definition\n',
                    'bindingsite_center ' + str(pocket_definition["center"][0]) + ' ' + str(pocket_definition["center"][1]) + ' ' + str(pocket_definition["center"][2]) + '\n',
                    'bindingsite_radius ' + str(pocket_definition["size"][0] / 2) + '\n',

                    '# cluster algorithm\n',
                    'cluster_structures 10\n',
                    'cluster_rmsd 2.0\n',

                    '# write\n',
                    'write_ranking_links 0\n',
                    'write_protein_bindingsite 1\n',
                    'write_protein_conformations 1\n',
                    'write_protein_splitted 1\n',
                    'write_merged_protein 0\n',
                    '####\n']
    plp_rescoring_config_path_config = plp_rescoring_config_path_txt.with_suffix('.config')
    with plp_rescoring_config_path_config.open('w') as configwriter:
        configwriter.writelines(plp_config)

    # Run PLANTS docking
    plp_rescoring_command = f'{software}/PLANTS --mode rescore {plp_rescoring_config_path_config}'
    subprocess.call(plp_rescoring_command, shell=True, stdout=DEVNULL, stderr=STDOUT)

    # Fetch results
    results_csv_location = plp_rescoring_folder / 'results' / 'ranking.csv'
    plp_results = pd.read_csv(results_csv_location, sep=',', header=0)
    plp_results.rename(columns={'TOTAL_SCORE': column_name}, inplace=True)
    for i, row in plp_results.iterrows():
        split = row['LIGAND_ENTRY'].split('_')
        plp_results.loc[i, ['Pose ID']] = f'{split[0]}_{split[1]}_{split[2]}'
    plp_rescoring_output = plp_results[['Pose ID', column_name]]
    plp_rescoring_output.to_csv(rescoring_folder / f'{column_name}_rescoring' / f'{column_name}_scores.csv', index=False)

    # Remove files
    plants_ligands_mol2.unlink()
    delete_files(rescoring_folder / f'{column_name}_rescoring', f'{column_name}_scores.csv')
    toc = time.perf_counter()
    printlog(f'Rescoring with PLP complete in {toc-tic:0.4f}!')
    return plp_rescoring_output

def chemplp_rescoring(sdf : str, ncpus : int, column_name : str, **kwargs):
    rescoring_folder = kwargs.get('rescoring_folder')
    software = kwargs.get('software')
    protein_file = kwargs.get('protein_file')
    
    tic = time.perf_counter()

    plants_search_speed = 'speed1'
    ants = '20'
    chemplp_rescoring_folder = kwargs.get('rescoring_folder') / f'{column_name}_rescoring'
    chemplp_rescoring_folder.mkdir(parents=True, exist_ok=True)
    # Convert protein file to .mol2 using open babel
    plants_protein_mol2 = chemplp_rescoring_folder / 'protein.mol2'
    convert_molecules(protein_file, plants_protein_mol2, 'pdb', 'mol2')
    # Convert prepared ligand file to .mol2 using open babel
    plants_ligands_mol2 = chemplp_rescoring_folder / 'ligands.mol2'
    convert_molecules(sdf, plants_ligands_mol2, 'sdf', 'mol2')
    
    chemplp_rescoring_config_path_txt = chemplp_rescoring_folder / 'config.txt'
    chemplp_config = ['# search algorithm\n',
                        'search_speed ' + plants_search_speed + '\n',
                        'aco_ants ' + ants + '\n',
                        'flip_amide_bonds 0\n',
                        'flip_planar_n 1\n',
                        'force_flipped_bonds_planarity 0\n',
                        'force_planar_bond_rotation 1\n',
                        'rescore_mode simplex\n',
                        'flip_ring_corners 0\n',
                        '# scoring functions\n',
                        '# Intermolecular (protein-ligand interaction scoring)\n',
                        'scoring_function chemplp\n',
                        'outside_binding_site_penalty 50.0\n',
                        'enable_sulphur_acceptors 1\n',
                        '# Intramolecular ligand scoring\n',
                        'ligand_intra_score clash2\n',
                        'chemplp_clash_include_14 1\n',
                        'chemplp_clash_include_HH 0\n',

                        '# input\n',
                        'protein_file ' + str(plants_protein_mol2) + '\n',
                        'ligand_file ' + str(plants_ligands_mol2) + '\n',

                        '# output\n',
                        'output_dir ' + str(chemplp_rescoring_folder / 'results') + '\n',

                        '# write single mol2 files (e.g. for RMSD calculation)\n',
                        'write_multi_mol2 1\n',

                        '# binding site definition\n',
                        'bindingsite_center ' + str(kwargs.get('pocket_definition')["center"][0]) + ' ' + str(kwargs.get('pocket_definition')["center"][1]) + ' ' + str(kwargs.get('pocket_definition')["center"][2]) + '\n',
                        'bindingsite_radius ' + str(kwargs.get('pocket_definition')["size"][0] / 2) + '\n',

                        '# cluster algorithm\n',
                        'cluster_structures 10\n',
                        'cluster_rmsd 2.0\n',

                        '# write\n',
                        'write_ranking_links 0\n',
                        'write_protein_bindingsite 1\n',
                        'write_protein_conformations 1\n',
                        'write_protein_splitted 1\n',
                        'write_merged_protein 0\n',
                        '####\n']
    # Write config file
    chemplp_rescoring_config_path_config = chemplp_rescoring_config_path_txt.with_suffix(
        '.config')
    with chemplp_rescoring_config_path_config.open('w') as configwriter:
        configwriter.writelines(chemplp_config)

    # Run PLANTS docking
    chemplp_rescoring_command = f'{software}/PLANTS --mode rescore {chemplp_rescoring_config_path_config}'
    subprocess.call(chemplp_rescoring_command, shell=True, stdout=DEVNULL, stderr=STDOUT)

    # Fetch results
    results_csv_location = chemplp_rescoring_folder / 'results' / 'ranking.csv'
    chemplp_results = pd.read_csv(results_csv_location, sep=',', header=0)
    chemplp_results.rename(columns={'TOTAL_SCORE': column_name}, inplace=True)
    for i, row in chemplp_results.iterrows():
        split = row['LIGAND_ENTRY'].split('_')
        chemplp_results.loc[i, ['Pose ID']] = f'{split[0]}_{split[1]}_{split[2]}'
    chemplp_rescoring_output = chemplp_results[['Pose ID', column_name]]
    chemplp_rescoring_output.to_csv(rescoring_folder / f'{column_name}_rescoring' / f'{column_name}_scores.csv', index=False)

    # Remove files
    plants_ligands_mol2.unlink()
    delete_files(rescoring_folder / f'{column_name}_rescoring', f'{column_name}_scores.csv')

    toc = time.perf_counter()
    printlog(f'Rescoring with CHEMPLP complete in {toc-tic:0.4f}!')
    return chemplp_rescoring_output

def oddt_nnscore_rescoring(sdf : str, ncpus : int, column_name : str, **kwargs):
    """
    Rescores the input SDF file using the NNscore algorithm and returns a Pandas dataframe with the rescored values.

    Args:
    sdf (str): Path to the input SDF file.
    ncpus (int): Number of CPUs to use for the rescoring.
    column_name (str): Name of the column to store the rescored values in the output dataframe.
    **kwargs: Additional keyword arguments.

    Returns:
    df (Pandas dataframe): Dataframe with the rescored values and the corresponding pose IDs.
    """
    rescoring_folder = kwargs.get('rescoring_folder')
    software = kwargs.get('software')
    protein_file = kwargs.get('protein_file')

    tic = time.perf_counter()

    nnscore_rescoring_folder = rescoring_folder / f'{column_name}_rescoring'
    nnscore_rescoring_folder.mkdir(parents=True, exist_ok=True)
    pickle_path = f'{software}/models/NNScore_pdbbind2016.pickle'
    results = nnscore_rescoring_folder / 'rescored_NNscore.sdf'
    nnscore_rescoring_command = ('oddt_cli ' + str(sdf) + ' --receptor ' + str(protein_file) + ' -n ' + str(ncpus) + ' --score_file ' + str(pickle_path) + ' -O ' + str(results))
    subprocess.call(nnscore_rescoring_command, shell=True, stdout=DEVNULL, stderr=STDOUT)
    df = PandasTools.LoadSDF(str(results), idName='Pose ID', molColName=None, includeFingerprints=False, removeHs=False)
    df.rename(columns={'nnscore': column_name}, inplace=True)
    df = df[['Pose ID', column_name]]
    df.to_csv(nnscore_rescoring_folder / f'{column_name}_scores.csv', index=False)
    toc = time.perf_counter()
    printlog(f'Rescoring with NNscore complete in {toc-tic:0.4f}!')
    delete_files(nnscore_rescoring_folder, f'{column_name}_scores.csv')
    return df

def oddt_plecscore_rescoring(sdf : str, ncpus : int, column_name : str, **kwargs):
    """
    Rescores the input SDF file using the PLECscore rescoring method.

    Args:
    - sdf (str): the path to the input SDF file
    - ncpus (int): the number of CPUs to use for the rescoring calculation
    - column_name (str): the name of the column to use for the rescoring results
    **kwargs: Additional keyword arguments.

    Returns:
    - df (pandas.DataFrame): a DataFrame containing the rescoring results, with columns 'Pose ID' and 'column_name'
    """
    rescoring_folder = kwargs.get('rescoring_folder')
    software = kwargs.get('software')
    protein_file = kwargs.get('protein_file')

    tic = time.perf_counter()

    plecscore_rescoring_folder = rescoring_folder / f'{column_name}_rescoring'
    plecscore_rescoring_folder.mkdir(parents=True, exist_ok=True)
    pickle_path = f'{software}/models/PLECnn_p5_l1_pdbbind2016_s65536.pickle'
    results = plecscore_rescoring_folder / 'rescored_PLECnn.sdf'
    plecscore_rescoring_command = ('oddt_cli ' + str(sdf) + ' --receptor ' + str(protein_file) + ' -n ' + str(ncpus) + ' --score_file ' + str(pickle_path) + ' -O ' + str(results)        )
    subprocess.call(plecscore_rescoring_command, shell=True)
    df = PandasTools.LoadSDF(str(results), idName='Pose ID', molColName=None, includeFingerprints=False, removeHs=False)
    df.rename(columns={'PLECnn_p5_l1_s65536': column_name}, inplace=True)
    df = df[['Pose ID', column_name]]
    df.to_csv(plecscore_rescoring_folder / f'{column_name}_scores.csv', index=False)
    toc = time.perf_counter()
    printlog(f'Rescoring with PLECScore complete in {toc-tic:0.4f}!')
    delete_files(plecscore_rescoring_folder, f'{column_name}_scores.csv')
    return df

def SCORCH_rescoring(sdf : str, ncpus : int, column_name : str, **kwargs):
    """
    Rescores ligands in an SDF file using SCORCH and saves the results in a CSV file.

    Args:
        sdf (str): Path to the SDF file containing the ligands to be rescored.
        ncpus (int): Number of CPUs to use for parallel processing.
        column_name (str): Name of the column to store the SCORCH scores in the output CSV file.
        **kwargs: Additional keyword arguments.

    Returns:
        None
    """
    rescoring_folder = kwargs.get('rescoring_folder')
    software = kwargs.get('software')
    protein_file = kwargs.get('protein_file')

    tic = time.perf_counter()
    SCORCH_rescoring_folder = rescoring_folder / f'{column_name}_rescoring'
    SCORCH_rescoring_folder.mkdir(parents=True, exist_ok=True)
    SCORCH_protein = SCORCH_rescoring_folder / "protein.pdbqt"
    convert_molecules(str(protein_file).replace('.pdb', '_pocket.pdb'), SCORCH_protein, 'pdb', 'pdbqt')
    # Convert ligands to pdbqt
    split_files_folder = SCORCH_rescoring_folder / f'split_{sdf.stem}'
    split_files_folder.mkdir(exist_ok=True)
    convert_molecules(sdf, split_files_folder, 'sdf', 'pdbqt')
    # Run SCORCH

    SCORCH_command = f'python {software}/SCORCH-1.0.0/scorch.py --receptor {SCORCH_protein} --ligand {split_files_folder} --out {SCORCH_rescoring_folder}/scoring_results.csv --threads {ncpus} --return_pose_scores'
    subprocess.call(SCORCH_command, shell=True, stdout=DEVNULL, stderr=STDOUT)
    # Clean data
    SCORCH_scores = pd.read_csv(SCORCH_rescoring_folder / 'scoring_results.csv')
    SCORCH_scores = SCORCH_scores.rename(columns={'Ligand_ID': 'Pose ID',
                                                    'SCORCH_pose_score': column_name})
    SCORCH_scores = SCORCH_scores[[column_name, 'Pose ID']]
    SCORCH_scores.to_csv(SCORCH_rescoring_folder / f'{column_name}_scores.csv', index=False)
    delete_files(SCORCH_rescoring_folder, f'{column_name}_scores.csv')
    toc = time.perf_counter()
    printlog(f'Rescoring with SCORCH complete in {toc-tic:0.4f}!')
    return

def RTMScore_rescoring(sdf: str, ncpus: int, column_name: str, **kwargs):
    """
    Rescores poses in an SDF file using RTMScore.

    Args:
    - sdf (str): Path to the SDF file containing the poses to be rescored.
    - ncpus (int): Number of CPUs to use for parallel execution.
    - column_name (str): Name of the column in the output CSV file that will contain the RTMScore scores.
    - **kwargs: Additional keyword arguments.

    Keyword Args:
    - rescoring_folder (str): Path to the folder where the rescoring results will be saved.
    - software (str): Path to the RTMScore software.
    - protein_file (str): Path to the protein file.
    - pocket_definition (str): Path to the pocket definition file.

    Returns:
    - None
    """
    rescoring_folder = kwargs.get('rescoring_folder')
    software = kwargs.get('software')
    protein_file = kwargs.get('protein_file')

    tic = time.perf_counter()
    (rescoring_folder / f'{column_name}_rescoring').mkdir(parents=True, exist_ok=True)
    output_file = str(rescoring_folder / f'{column_name}_rescoring' / f'{column_name}_scores.csv')
    
    RTMScore_command = (f'cd {rescoring_folder / "RTMScore_rescoring"} && python {software}/RTMScore-main/example/rtmscore.py' +
                        f' -p {str(protein_file).replace(".pdb", "_pocket.pdb")}' +
                        f' -l {sdf}' +
                        ' -o RTMScore_scores' +
                        ' -pl' 
                        f' -m {software}/RTMScore-main/trained_models/rtmscore_model1.pth')
    subprocess.call(RTMScore_command, shell=True, stdout=DEVNULL, stderr=STDOUT)
    df = pd.read_csv(output_file)
    df = df.rename(columns={'id': 'Pose ID', 'score': f'{column_name}'})
    df['Pose ID'] = df['Pose ID'].str.rsplit('-', n=1).str[0]
    df.to_csv(output_file, index=False)
    delete_files(rescoring_folder / f'{column_name}_rescoring', f'{column_name}_scores.csv')
    toc = time.perf_counter()
    printlog(f'Rescoring with RTMScore complete in {toc-tic:0.4f}!')
    return

def LinF9_rescoring(sdf : str, ncpus : int, column_name : str, **kwargs):
    """
    Performs rescoring of poses in an SDF file using the LinF9 scoring function.

    Args:
    sdf (str): The path to the SDF file containing the poses to be rescored.
    ncpus (int): The number of CPUs to use for parallel execution.
    column_name (str): The name of the column to store the rescoring results.
    **kwargs: Additional keyword arguments.

    Keyword Args:
    rescoring_folder (str): Path to the folder where the rescoring results will be saved.
    software (str): Path to the software.
    protein_file (str): Path to the protein file.
    pocket_definition (str): Path to the pocket definition file.

    Returns:
    pandas.DataFrame: A DataFrame containing the rescoring results, with columns 'Pose ID' and the specified column name.
    """
    rescoring_folder = kwargs.get('rescoring_folder')
    software = kwargs.get('software')
    protein_file = kwargs.get('protein_file')
    pocket_definition = kwargs.get('pocket_definition')

    tic = time.perf_counter()
    (rescoring_folder / f'{column_name}_rescoring').mkdir(parents=True, exist_ok=True)
    split_files_folder = split_sdf_str(rescoring_folder / f'{column_name}_rescoring', sdf, ncpus)
    split_files_sdfs = [Path(split_files_folder) / f for f in os.listdir(split_files_folder) if f.endswith('.sdf')]
    
    global LinF9_rescoring_splitted

    def LinF9_rescoring_splitted(split_file, protein_file, pocket_definition):
        LinF9_folder = rescoring_folder / 'LinF9_rescoring'
        results = LinF9_folder / f'{split_file.stem}_LinF9.sdf'
        LinF9_cmd = (
            f'{software}/smina.static' +
            f' --receptor {protein_file}' +
            f' --ligand {split_file}' +
            f' --out {results}' +
            f' --center_x {pocket_definition["center"][0]}' +
            f' --center_y {pocket_definition["center"][1]}' +
            f' --center_z {pocket_definition["center"][2]}' +
            f' --size_x {pocket_definition["size"][0]}' +
            f' --size_y {pocket_definition["size"][1]}' +
            f' --size_z {pocket_definition["size"][2]}' +
            ' --cpu 1' +
            ' --scoring Lin_F9 --score_only')
        try:
            subprocess.call(LinF9_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
        except Exception as e:
            printlog(f'LinF9 rescoring failed: {e}')
        return

    parallel_executor(LinF9_rescoring_splitted, split_files_sdfs, ncpus, protein_file=protein_file, pocket_definition=pocket_definition)
    
    try:
        LinF9_dataframes = [PandasTools.LoadSDF(str(rescoring_folder / 'LinF9_rescoring' / file),
                                                idName='Pose ID',
                                                molColName=None,
                                                includeFingerprints=False,
                                                embedProps=False,
                                                removeHs=False,
                                                strictParsing=True) for file in os.listdir(
                                                rescoring_folder /
                                                'LinF9_rescoring') if file.startswith('split') and file.endswith('_LinF9.sdf')
                            ]
    except Exception as e:
        printlog('ERROR: Failed to Load LinF9 rescoring SDF file!')
        printlog(e)

    try:
        LinF9_rescoring_results = pd.concat(LinF9_dataframes)
    except Exception as e:
        printlog('ERROR: Could not combine LinF9 rescored poses')
        printlog(e)

    LinF9_rescoring_results.rename(columns={'minimizedAffinity': column_name},inplace=True)
    LinF9_rescoring_results = LinF9_rescoring_results[['Pose ID', column_name]]
    LinF9_rescoring_results.to_csv(rescoring_folder / 'LinF9_rescoring' / 'LinF9_scores.csv', index=False)
    delete_files(rescoring_folder / 'LinF9_rescoring', 'LinF9_scores.csv')
    toc = time.perf_counter()
    printlog(f'Rescoring with LinF9 complete in {toc-tic:0.4f}!')
    return LinF9_rescoring_results

def AAScore_rescoring(sdf: str, ncpus: int, column_name: str, **kwargs) -> DataFrame:
    """
    Rescores poses in an SDF file using the AA-Score tool.

    Args:
    - sdf (str): The path to the SDF file containing the poses to be rescored.
    - ncpus (int): The number of CPUs to use for parallel processing.
    - column_name (str): The name of the column to be used for the rescored scores.
    - kwargs: Additional keyword arguments.

    Keyword Args:
    - rescoring_folder (str): The path to the folder where the rescored results will be saved.
    - software (str): The path to the AA-Score software.
    - protein_file (str): The path to the protein file.
    - pocket_de (str): The path to the pocket definitions file.

    Returns:
    - A pandas DataFrame containing the rescored poses and their scores.
    """
    tic = time.perf_counter()
    rescoring_folder = kwargs.get('rescoring_folder')
    software = kwargs.get('software')
    protein_file = kwargs.get('protein_file')

    (rescoring_folder / f'{column_name}_rescoring').mkdir(parents=True, exist_ok=True)
    pocket = str(protein_file).replace('.pdb', '_pocket.pdb')

    if ncpus == 1:
        results = rescoring_folder / f'{column_name}_rescoring' / 'rescored_AAScore.csv'
        AAscore_cmd = f'python {software}/AA-Score-Tool-main/AA_Score.py --Rec {pocket} --Lig {sdf} --Out {results}'
        subprocess.call(AAscore_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
        AAScore_rescoring_results = pd.read_csv(results, delimiter='\t', header=None, names=['Pose ID', column_name])
    else:
        split_files_folder = split_sdf_str(rescoring_folder / f'{column_name}_rescoring', sdf, ncpus)
        split_files_sdfs = [Path(split_files_folder) / f for f in os.listdir(split_files_folder) if f.endswith('.sdf')]
        global AAScore_rescoring_splitted

        def AAScore_rescoring_splitted(split_file):
            AAScore_folder = rescoring_folder / 'AAScore_rescoring'
            results = AAScore_folder / f'{split_file.stem}_AAScore.csv'
            AAScore_cmd = f'python {software}/AA-Score-Tool-main/AA_Score.py --Rec {pocket} --Lig {split_file} --Out {results}'
            try:
                subprocess.call(AAScore_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
            except Exception as e:
                printlog('AAScore rescoring failed: ' + str(e))

        parallel_executor(AAScore_rescoring_splitted, split_files_sdfs, ncpus)

        try:
            AAScore_dataframes = [pd.read_csv(rescoring_folder / 'AAScore_rescoring' / file,
                                              delimiter='\t',
                                              header=None,
                                              names=['Pose ID', column_name])
                                  for file in os.listdir(rescoring_folder / 'AAScore_rescoring') if
                                  file.startswith('split') and file.endswith('.csv')
                                  ]
        except Exception as e:
            printlog('ERROR: Failed to Load AAScore rescoring SDF file!')
            printlog(e)
        else:
            try:
                AAScore_rescoring_results = pd.concat(AAScore_dataframes)
            except Exception as e:
                printlog('ERROR: Could not combine AAScore rescored poses')
                printlog(e)
            else:
                delete_files(rescoring_folder / 'AAScore_rescoring', 'AAScore_scores.csv')
        AAScore_rescoring_results.to_csv(rescoring_folder / 'AAScore_rescoring' / 'AAScore_scores.csv', index=False)
        toc = time.perf_counter()
        printlog(f'Rescoring with AAScore complete in {toc - tic:0.4f}!')
        return AAScore_rescoring_results

def KORPL_rescoring(sdf : str, ncpus : int, column_name : str, **kwargs):
    """
    Rescores a given SDF file using KORP-PL software and saves the results to a CSV file.

    Args:
    - sdf (str): The path to the SDF file to be rescored.
    - ncpus (int): The number of CPUs to use for parallel processing.
    - column_name (str): The name of the column to store the rescored values in.
    - rescoring_folder (str): The path to the folder to store the rescored results.
    - software (str): The path to the KORP-PL software.
    - protein_file (str): The path to the protein file.
    - pocket_definition (str): The path to the pocket definition file.

    Returns:
    - None
    """
    rescoring_folder = kwargs.get('rescoring_folder')
    software = kwargs.get('software')
    protein_file = kwargs.get('protein_file')

    tic = time.perf_counter()
    (rescoring_folder / f'{column_name}_rescoring').mkdir(parents=True, exist_ok=True)
    split_files_folder = split_sdf_str((rescoring_folder / f'{column_name}_rescoring'), sdf, ncpus)
    split_files_sdfs = [Path(split_files_folder) / f for f in os.listdir(split_files_folder) if f.endswith('.sdf')]
    global KORPL_rescoring_splitted

    def KORPL_rescoring_splitted(split_file, protein_file):
        df = PandasTools.LoadSDF(str(split_file), idName='Pose ID', molColName=None)
        df = df[['Pose ID']]
        korpl_command = (f'{software}/KORP-PL' +
                        ' --receptor ' + str(protein_file) +
                        ' --ligand ' + str(split_file) +
                        ' --sdf')
        process = subprocess.Popen(korpl_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = process.communicate()
        energies = []
        output = stdout.decode().splitlines()
        for line in output:
            if line.startswith('model'):
                parts = line.split(',')
                energy = round(float(parts[1].split('=')[1]), 2)
                energies.append(energy)
        df[column_name] = energies
        output_csv = str(rescoring_folder / f'{column_name}_rescoring' / (str(split_file.stem) + '_scores.csv'))
        df.to_csv(output_csv, index=False)
        return
        
    parallel_executor(KORPL_rescoring_splitted, split_files_sdfs, ncpus, protein_file=protein_file)
    
    print('Combining KORPL scores')
    scores_folder = rescoring_folder / f'{column_name}_rescoring'
    # Get a list of all files with names ending in "_scores.csv"
    score_files = list(scores_folder.glob('*_scores.csv'))
    if not score_files:
        print("No CSV files found with names ending in '_scores.csv' in the specified folder.")
    else:
        # Read and concatenate the CSV files into a single DataFrame
        combined_scores_df = pd.concat([pd.read_csv(file) for file in score_files], ignore_index=True)
        # Save the combined scores to a single CSV file
        combined_scores_csv = scores_folder / f'{column_name}_scores.csv'
        combined_scores_df.to_csv(combined_scores_csv, index=False)
    delete_files(rescoring_folder / f'{column_name}_rescoring', f'{column_name}_scores.csv')
    toc = time.perf_counter()
    printlog(f'Rescoring with KORPL complete in {toc-tic:0.4f}!')
    return

def ConvexPLR_rescoring(sdf : str, ncpus : int, column_name : str, **kwargs):
    """
    Rescores the given SDF file using Convex-PLR software and saves the results in a CSV file.

    Args:
        - sdf (str): path to the input SDF file
        - ncpus (int): number of CPUs to use for parallel processing
        - column_name (str): name of the column to store the scores in the output CSV file
        - rescoring_folder (str): path to the folder to store the rescoring results
        - software (str): path to the Convex-PLR software
        - protein_file (str): path to the protein file
        - pocket_definition (str): path to the pocket definition file

    Returns:
        - None
    """

    tic = time.perf_counter()
    rescoring_folder = kwargs.get('rescoring_folder')
    software = kwargs.get('software')
    protein_file = kwargs.get('protein_file')
    
    (rescoring_folder / f'{column_name}_rescoring').mkdir(parents=True, exist_ok=True)
    split_files_folder = split_sdf_str((rescoring_folder / f'{column_name}_rescoring'), sdf, ncpus)
    split_files_sdfs = [Path(split_files_folder) / f for f in os.listdir(split_files_folder) if f.endswith('.sdf')]
    global ConvexPLR_rescoring_splitted
    def ConvexPLR_rescoring_splitted(split_file, protein_file):
        df = PandasTools.LoadSDF(str(split_file), idName='Pose ID', molColName=None)
        df = df[['Pose ID']]
        ConvexPLR_command = (f'{software}/Convex-PL' +
                            f' --receptor {protein_file}' +
                            f' --ligand {split_file}' +
                            ' --sdf --regscore')
        process = subprocess.Popen(ConvexPLR_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = process.communicate()
        energies = []
        output = stdout.decode().splitlines()
        for line in output:
            if line.startswith('model'):
                parts = line.split(',')
                energy = round(float(parts[1].split('=')[1]), 2)
                energies.append(energy)
        df[column_name] = energies
        output_csv = str(rescoring_folder / f'{column_name}_rescoring' / (str(split_file.stem) + '_scores.csv'))
        df.to_csv(output_csv, index=False)
        return
    
    parallel_executor(ConvexPLR_rescoring_splitted, split_files_sdfs, ncpus, protein_file=protein_file)
    
    # Get a list of all files with names ending in "_scores.csv"
    score_files = list((rescoring_folder / f'{column_name}_rescoring').glob('*_scores.csv'))
    # Read and concatenate the CSV files into a single DataFrame
    combined_scores_df = pd.concat([pd.read_csv(file) for file in score_files], ignore_index=True)
    # Save the combined scores to a single CSV file
    combined_scores_csv = rescoring_folder / f'{column_name}_rescoring' / f'{column_name}_scores.csv'
    combined_scores_df.to_csv(combined_scores_csv, index=False)
    delete_files(rescoring_folder / f'{column_name}_rescoring', f'{column_name}_scores.csv')
    toc = time.perf_counter()
    printlog(f'Rescoring with ConvexPLR complete in {toc-tic:0.4f}!')
    return

#add new scoring functions here!
# Dict key: (function, column_name, min or max ordering, min value for scaled standardisation, max value for scaled standardisation)
RESCORING_FUNCTIONS = {
    'GNINA-Affinity':   {'function': gnina_rescoring,         'column_name': 'GNINA-Affinity', 'best_value': 'min', 'range': (100, -100)},
    'CNN-Score':        {'function': gnina_rescoring,         'column_name': 'CNN-Score',      'best_value': 'max', 'range': (0, 1)},
    'CNN-Affinity':     {'function': gnina_rescoring,         'column_name': 'CNN-Affinity',   'best_value': 'max', 'range': (0, 20)},
    'Vinardo':          {'function': vinardo_rescoring,       'column_name': 'Vinardo',        'best_value': 'min', 'range': (200, 20)},
    'AD4':              {'function': AD4_rescoring,           'column_name': 'AD4',            'best_value': 'min', 'range': (100, -100)},
    'RFScoreVS':        {'function': rfscorevs_rescoring,     'column_name': 'RFScoreVS',      'best_value': 'max', 'range': (5, 10)},
    #'RFScoreVS2':       {'function': rfscorevs_rescoring2,    'column_name': 'RFScoreVS',      'best_value': 'max', 'range': (5, 10)},
    'PLP':              {'function': plp_rescoring,           'column_name': 'PLP',            'best_value': 'min', 'range': (200, -200)},
    'CHEMPLP':          {'function': chemplp_rescoring,       'column_name': 'CHEMPLP',        'best_value': 'min', 'range': (200, -200)},
    'NNScore':          {'function': oddt_nnscore_rescoring,  'column_name': 'NNScore',        'best_value': 'max', 'range': (0, 20)},
    'PLECScore':        {'function': oddt_plecscore_rescoring,'column_name': 'PLECScore',      'best_value': 'max', 'range': (0, 20)},
    'LinF9':            {'function': LinF9_rescoring,         'column_name': 'LinF9',          'best_value': 'min', 'range': (100, -100)},
    'AAScore':          {'function': AAScore_rescoring,       'column_name': 'AAScore',        'best_value': 'max', 'range': (100, -100)},
    'SCORCH':           {'function': SCORCH_rescoring,        'column_name': 'SCORCH',         'best_value': 'max', 'range': (0, 1)},
    'RTMScore':         {'function': RTMScore_rescoring,      'column_name': 'RTMScore',       'best_value': 'max', 'range': (0, 100)},
    'KORP-PL':          {'function': KORPL_rescoring,         'column_name': 'KORP-PL',        'best_value': 'min', 'range': (200, -1000)},
    'ConvexPLR':        {'function': ConvexPLR_rescoring,     'column_name': 'ConvexPLR',      'best_value': 'max', 'range': (-10, 10)}
}


def rescore_poses(w_dir: Path, protein_file: Path, pocket_definition: dict, software: Path, clustered_sdf: Path, functions: List[str], ncpus: int) -> None:
    """
    Rescores ligand poses using the specified software and scoring functions. The function splits the input SDF file into
    smaller files, and then runs the specified software on each of these files in parallel. The results are then combined into a single
    Pandas dataframe and saved to a CSV file.

    Args:
        w_dir (str): The working directory.
        protein_file (str): The path to the protein file.
        pocket_definition (dict): A dictionary containing the pocket center and size.
        software (str): The path to the software to be used for rescoring.
        clustered_sdf (str): The path to the input SDF file containing the clustered poses.
        functions (List[str]): A list of the scoring functions to be used.
        ncpus (int): The number of CPUs to use for parallel execution.

    Returns:
        None
    """
    RDLogger.DisableLog('rdApp.*') 
    tic = time.perf_counter()
    rescoring_folder_name = Path(clustered_sdf).stem
    rescoring_folder = w_dir / f'rescoring_{rescoring_folder_name}'
    (rescoring_folder).mkdir(parents=True, exist_ok=True)

    skipped_functions = []
    for function in functions:
        function_info = RESCORING_FUNCTIONS.get(function)
        if not (rescoring_folder / f'{function}_rescoring' / f'{function}_scores.csv').is_file():
            try:
                function_info['function'](clustered_sdf, ncpus, function_info['column_name'], protein_file=protein_file, pocket_definition=pocket_definition, software=software, rescoring_folder=rescoring_folder)
            except Exception as e:
                printlog(e)
                printlog(f'Failed for {function}')
        else:
            skipped_functions.append(function)
    if skipped_functions:
        printlog(f'Skipping functions: {", ".join(skipped_functions)}')


    score_files = [f'{function}_scores.csv' for function in functions]
    csv_files = [file for file in (rescoring_folder.rglob('*.csv')) if file.name in score_files]
    csv_dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        csv_dfs.append(df)
    if len(csv_dfs) == 1:
        combined_dfs = csv_dfs[0]
    if len(csv_dfs) > 1:
        combined_dfs = csv_dfs[0]
        for df in tqdm(csv_dfs[1:], desc='Combining scores', unit='files'):
            combined_dfs = pd.merge(combined_dfs, df, on='Pose ID', how='inner')
    first_column = combined_dfs.pop('Pose ID')
    combined_dfs.insert(0, 'Pose ID', first_column)
    columns = combined_dfs.columns
    col = columns[1:]
    for c in col.tolist():
        if c == 'Pose ID':
            pass
        if combined_dfs[c].dtypes is not float:
            combined_dfs[c] = combined_dfs[c].apply(pd.to_numeric, errors='coerce')
        else:
            pass
    combined_dfs.to_csv(rescoring_folder / 'allposes_rescored.csv', index=False)
    #delete_files(rescoring_folder, 'allposes_rescored.csv')
    toc = time.perf_counter()
    printlog(f'Rescoring complete in {toc - tic:0.4f}!')
    return

def rescore_docking(w_dir: Path, protein_file: Path, pocket_definition: dict, software: Path, function: str, ncpus: int) -> None:
    """
    Rescores ligand poses using the specified software and scoring functions. The function splits the input SDF file into
    smaller files, and then runs the specified software on each of these files in parallel. The results are then combined into a single
    Pandas dataframe and saved to a CSV file.

    Args:
        w_dir (str): The working directory.
        protein_file (str): The path to the protein file.
        pocket_definition (dict): A dictionary containing the pocket center and size.
        software (str): The path to the software to be used for rescoring.
        all_poses (str): The path to the input SDF file containing the clustered poses.
        functions (List[str]): A list of the scoring functions to be used.
        ncpus (int): The number of CPUs to use for parallel execution.

    Returns:
        None
    """    
    RDLogger.DisableLog('rdApp.*') 
    tic = time.perf_counter()
    
    all_poses = Path(f"{w_dir}/allposes.sdf")
    
    function_info = RESCORING_FUNCTIONS.get(function)

    function_info['function'](all_poses, ncpus, function_info['column_name'], protein_file=protein_file, pocket_definition=pocket_definition, software=software, rescoring_folder=w_dir)
    
    score_file = f'{w_dir}/{function}_rescoring/{function}_scores.csv'
    
    score_df = pd.read_csv(score_file)
    if 'Unnamed: 0' in score_df.columns:
        score_df = score_df.drop(columns=['Unnamed: 0'])
        
    score_df['Pose_Number'] = score_df['Pose ID'].str.split('_').str[2].astype(int)
    score_df['Docking_program'] = score_df['Pose ID'].str.split('_').str[1].astype(str)
    score_df['ID'] = score_df['Pose ID'].str.split('_').str[0].astype(str)
    
    if function_info['best_value'] == 'min':
        best_pose_indices = score_df.groupby('ID')[function_info['column_name']].idxmin()
    else:
        best_pose_indices = score_df.groupby('ID')[function_info['column_name']].idxmax()
    
    if os.path.exists(score_file):
        os.remove(score_file)
    if os.path.exists(f'{w_dir}/{function}_rescoring'):
        os.rmdir(f'{w_dir}/{function}_rescoring')
        
    best_poses = pd.DataFrame(score_df.loc[best_pose_indices, 'Pose ID'])
    toc = time.perf_counter()
    printlog(f'Rescoring complete in {toc - tic:0.4f}!')
    return best_poses

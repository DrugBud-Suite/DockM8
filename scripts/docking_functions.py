import os
import shutil
import subprocess
from subprocess import DEVNULL, STDOUT
import shutil
import pandas as pd
from rdkit.Chem import PandasTools
from IPython.display import display
import time
from scripts.utilities import *
import multiprocessing
from tqdm import tqdm
from pathlib import Path


def smina_docking(protein_file, ref_file, software, exhaustiveness, n_poses):
    '''
    Perform docking using the SMINA software on a protein and a reference ligand, and return the path to the results.

    Args:
    protein_file (str): path to the protein file in PDB format
    ref_file (str): path to the reference ligand file in SDF format
    software (str): path to the software folder
    exhaustiveness (int): level of exhaustiveness for the docking search, ranges from 0-8
    n_poses (int): number of poses to be generated

    Returns:
    results_path (str): the path to the results file in SDF format
    '''
    tic = time.perf_counter()
    w_dir = os.path.dirname(protein_file)
    library = w_dir+'/temp/final_library.sdf'
    smina_folder = w_dir+'/temp/smina/'
    try:
        os.mkdir(smina_folder, mode = 0o777)
    except:
        print('SMINA folder already exists')
    results_path = smina_folder+'docked.sdf'
    log = smina_folder+'log.txt'
    smina_cmd = 'cd '+software+' && ./gnina -r '+protein_file+' -l '+library+' --autobox_ligand '+ref_file+' -o '+results_path+' --exhaustiveness ' +str(exhaustiveness)+' --num_modes '+str(n_poses)+' --cnn_scoring none'+' --log '+log
    try:
        subprocess.call(smina_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
    except Exception as e:
        print('SMINA docking failed: '+e)
    toc = time.perf_counter()
    print(f'Docking with SMINA complete in {toc-tic:0.4f}!')
    return results_path

def gnina_docking(protein_file, ref_file, software, exhaustiveness, n_poses):
    '''
    Perform docking using the GNINA software on a protein and a reference ligand, and return the path to the results.

    Args:
    protein_file (str): path to the protein file in PDB format
    ref_file (str): path to the reference ligand file in SDF format
    software (str): path to the software folder
    exhaustiveness (int): level of exhaustiveness for the docking search, ranges from 0-8
    n_poses (int): number of poses to be generated

    Returns:
    results_path (str): the path to the results file in SDF format
    '''
    tic = time.perf_counter()
    w_dir = os.path.dirname(protein_file)
    library = w_dir+'/temp/final_library.sdf'
    gnina_folder = w_dir+'/temp/gnina/'
    try:
        os.mkdir(gnina_folder, mode = 0o777)
    except:
        print('GNINA folder already exists')
    results_path = gnina_folder+'/docked.sdf'
    log = gnina_folder+'log.txt'
    gnina_cmd = 'cd '+software+' && ./gnina -r '+protein_file+' -l '+library+' --autobox_ligand '+ref_file+' -o '+results_path+' --exhaustiveness ' +str(exhaustiveness)+' --num_modes '+str(n_poses)+' --cnn_scoring rescore --cnn crossdock_default2018 '+' --log '+log
    try:
        subprocess.call(gnina_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
    except Exception as e:
        print('GNINA docking failed: '+e)
    toc = time.perf_counter()
    print(f'Docking with GNINA complete in {toc-tic:0.4f}!')
    return results_path

def plants_docking(protein_file, ref_file, software, n_poses):
    '''
    Perform docking using the PLANTS software on a protein and a reference ligand, and return the path to the results.

    Args:
    protein_file (str): path to the protein file in PDB format
    ref_file (str): path to the reference ligand file in SDF format
    software (str): path to the software folder
    exhaustiveness (int): level of exhaustiveness for the docking search, ranges from 0-8
    n_poses (int): number of poses to be generated

    Returns:
    results_path (str): the path to the results file in SDF format
    '''
    tic = time.perf_counter()
    w_dir = os.path.dirname(protein_file)
    # Define initial variables
    plants_search_speed = 'speed1'
    ants = '20'
    plants_docking_scoring = 'chemplp'
    plants_docking_dir = w_dir+'/temp/plants'
    plants_docking_results_dir = w_dir+'/temp/plants/results'
    #Create plants docking folder
    if os.path.isdir(plants_docking_dir) == True:
        print('Plants docking folder already exists')
    else:
        os.mkdir(plants_docking_dir)
    #Convert protein file to .mol2 using open babel
    plants_protein_mol2 = w_dir+'/temp/plants/protein.mol2'
    try:
        print('Converting protein file to .mol2 format for PLANTS docking...')
        obabel_command = 'obabel -ipdb '+protein_file+' -O '+plants_protein_mol2
        subprocess.call(obabel_command, shell=True, stdout=DEVNULL, stderr=STDOUT)
    except Exception as e:
        print('ERROR: Failed to convert protein file to .mol2!')
        print(e)
    #Convert protein file to .mol2 using open babel
    plants_ref_mol2 = w_dir+'/temp/plants/ref.mol2'
    try:
        if ref_file.endswith('.mol2'):
            shutil.copy(ref_file, plants_docking_dir)
            os.rename(plants_docking_dir+'/'+os.path.basename(ref_file), plants_ref_mol2)
        else:
            print(f'Converting reference file from .{ref_file.split(".")[-1]} to .mol2 format for PLANTS docking...')
            obabel_command = f'obabel -i{ref_file.split(".")[-1]} {ref_file} -O {plants_ref_mol2}'
            os.system(obabel_command)
    except Exception as e:
        print('ERROR: Failed to convert reference file to .mol2 for PLANTS docking!')
        print(e)
    #Convert prepared ligand file to .mol2 using open babel
    final_library = w_dir+'/temp/final_library.sdf'
    plants_library_mol2 = w_dir+'/temp/plants/ligands.mol2'
    try:
        obabel_command = 'obabel -isdf '+final_library+' -O '+plants_library_mol2
        os.system(obabel_command)
    except Exception as e:
        print('ERROR: Failed to convert docking library file to .mol2!')
        print(e)
    #Determine binding site coordinates
    try:
        print('Determining binding site coordinates using PLANTS...')
        plants_binding_site_command = 'cd '+software+' && ./PLANTS --mode bind '+plants_ref_mol2+' 8'
        run_plants_binding_site = os.popen(plants_binding_site_command)
        output_plants_binding_site = run_plants_binding_site.readlines()
        keep = []
        for l in output_plants_binding_site:
            if l.startswith('binding'):
                keep.append(l)
            else:
                pass
        binding_site_center = keep[0].split()
        binding_site_radius = keep[1].split()
        binding_site_radius = binding_site_radius[1]
        binding_site_x = binding_site_center[1]
        binding_site_y = binding_site_center[2]
        binding_site_z = binding_site_center[3].replace('+', '')
    except Exception as e:
        print('ERROR: Could not determine binding site coordinates using PLANTS')
        print(e)
    #Generate plants config file
    plants_docking_config_path_txt = plants_docking_dir+"/config.txt"
    plants_config = ['# search algorithm\n',
    'search_speed '+plants_search_speed+'\n',
    'aco_ants '+ants+'\n',
    'flip_amide_bonds 0\n',
    'flip_planar_n 1\n',
    'force_flipped_bonds_planarity 0\n',
    'force_planar_bond_rotation 1\n',
    'rescore_mode simplex\n',
    'flip_ring_corners 0\n',
    '# scoring functions\n',
    '# Intermolecular (protein-ligand interaction scoring)\n',
    'scoring_function '+plants_docking_scoring+'\n',
    'outside_binding_site_penalty 50.0\n',
    'enable_sulphur_acceptors 1\n',
    '# Intramolecular ligand scoring\n',
    'ligand_intra_score clash2\n',
    'chemplp_clash_include_14 1\n',
    'chemplp_clash_include_HH 0\n',

    '# input\n',
    'protein_file '+plants_protein_mol2+'\n',
    'ligand_file '+plants_library_mol2+'\n',

    '# output\n',
    'output_dir '+plants_docking_results_dir+'\n',

    '# write single mol2 files (e.g. for RMSD calculation)\n',
    'write_multi_mol2 1\n',

    '# binding site definition\n',
    'bindingsite_center '+binding_site_x+' '+binding_site_y+' '+binding_site_z+'+\n',
    'bindingsite_radius '+binding_site_radius+'\n',

    '# cluster algorithm\n',
    'cluster_structures '+str(n_poses)+'\n',
    'cluster_rmsd 2.0\n',

    '# write\n',
    'write_ranking_links 0\n',
    'write_protein_bindingsite 0\n',
    'write_protein_conformations 0\n',
    'write_protein_splitted 0\n',
    'write_merged_protein 0\n',
    '####\n']
    #Write config file
    print('Writing PLANTS config file...')
    plants_docking_config_path_config = plants_docking_config_path_txt.replace(".txt", ".config")
    with open(plants_docking_config_path_config, 'w') as configwriter:
        configwriter.writelines(plants_config)
    configwriter.close()
    #Run PLANTS docking
    try:
        print('Starting PLANTS docking...')
        plants_docking_command = "cd "+software+" && ./PLANTS --mode screen "+plants_docking_config_path_config
        subprocess.call(plants_docking_command, shell=True, stdout=DEVNULL, stderr=STDOUT)
    except Exception as e:
        print('ERROR: PLANTS docking command failed...')
        print(e)
    plants_docking_results_mol2 = w_dir+"/temp/plants/results/docked_ligands.mol2"
    plants_docking_results_sdf = plants_docking_results_mol2.replace(".mol2", ".sdf")
    # Convert PLANTS poses to sdf
    try:
        print('Converting PLANTS poses to .sdf format...')
        obabel_command = 'obabel -imol2 '+plants_docking_results_mol2+' -O '+plants_docking_results_sdf 
        subprocess.call(obabel_command, shell=True, stdout=DEVNULL, stderr=STDOUT)
    except Exception as e:
        print('ERROR: Failed to convert PLANTS poses file to .sdf!')
        print(e)
    toc = time.perf_counter()
    print(f'Docking with PLANTS complete in {toc-tic:0.4f}!')
    return plants_docking_results_sdf

def fetch_poses(w_dir, n_poses, split_files_folder):
    '''
    This function is used to fetch the poses from different docking results (SMINA, GNINA, PLANTS) and create a new dataframe with the poses and their corresponding scores.
    It takes two input parameters:

    protein_file: the path of the protein file
    n_poses: number of poses to be fetched
    The function uses the PandasTools library to load the SDF files and creates a new dataframe with the poses and scores. It also renames the columns and modifies the ID column to include the source of the pose (SMINA, GNINA, PLANTS). In case of an error, the function will print an error message.
    '''
    tic = time.perf_counter()
    smina_docking_results = w_dir+"/temp/smina/docked.sdf"
    gnina_docking_results = w_dir+"/temp/gnina/docked.sdf"
    plants_poses_results_sdf = w_dir+"/temp/plants/results/docked_ligands.sdf"
    plants_scoring_results_sdf = w_dir+"/temp/plants/results/ranking.csv"
    all_poses = pd.DataFrame()
    #Fetch PLANTS poses
    print('Fetching PLANTS poses...')
    try:
        plants_poses = PandasTools.LoadSDF(plants_poses_results_sdf, idName='ID', molColName='Molecule',includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True)
        plants_scores = pd.read_csv(plants_scoring_results_sdf, usecols=['LIGAND_ENTRY', 'TOTAL_SCORE'])
        plants_scores = plants_scores.rename(columns={'LIGAND_ENTRY':'ID', 'TOTAL_SCORE':'CHEMPLP'})
        plants_scores = plants_scores[['ID', 'CHEMPLP']]
        plants_df = pd.merge(plants_scores, plants_poses, on='ID')
        plants_df['Pose ID'] = plants_df['ID'].str.split("_").str[0] + "_PLANTS_" + plants_df['ID'].str.split("_").str[4]
        plants_df['ID'] = plants_df['ID'].str.split("_").str[0]
        all_poses = pd.concat([all_poses, plants_df])
    except Exception as e:
        print('ERROR: Failed to Load PLANTS poses SDF file!')
        print(e)
    #Fetch SMINA poses
    print('Fetching SMINA poses...')
    try:
        smina_df = PandasTools.LoadSDF(smina_docking_results, idName='ID', molColName='Molecule',includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True)
        list_ = [*range(1, int(n_poses)+1, 1)]
        ser = list_ * (len(smina_df) // len(list_))
        smina_df['Pose ID'] = [f"{row['ID']}_SMINA_{num}" for num, (_, row) in zip(ser + list_[:len(smina_df)-len(ser)], smina_df.iterrows())]
        smina_df.rename(columns={'minimizedAffinity':'SMINA_Affinity'}, inplace=True)
        all_poses = pd.concat([all_poses, smina_df])
    except Exception as e:
        print('ERROR: Failed to Load SMINA poses SDF file!')
        print(e)
    #Fetch GNINA poses
    print('Fetching GNINA poses...')
    try:
        gnina_df = PandasTools.LoadSDF(gnina_docking_results, idName='ID', molColName='Molecule',includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True)
        list_ = [*range(1, int(n_poses)+1, 1)]
        ser = list_ * (len(gnina_df) // len(list_))
        gnina_df['Pose ID'] = [f"{row['ID']}_GNINA_{num}" for num, (_, row) in zip(ser + list_[:len(gnina_df)-len(ser)], gnina_df.iterrows())]
        gnina_df.rename(columns={'minimizedAffinity':'GNINA_Affinity'}, inplace=True)
        all_poses = pd.concat([all_poses, gnina_df])
        del gnina_df
    except Exception as e:
        print('ERROR: Failed to Load SMINA poses SDF file!')
        print(e)
    PandasTools.WriteSDF(all_poses, w_dir+"/temp/allposes.sdf", molColName='Molecule', idName='Pose ID', properties=list(all_poses.columns))
    toc = time.perf_counter()
    print(f'Combined all docking poses in {toc-tic:0.4f}!')
    return all_poses

def docking(w_dir, protein_file, ref_file, software, docking_programs, exhaustiveness, n_poses):
    tic = time.perf_counter()
    if 'SMINA' in docking_programs:
        smina_docking_results_path = smina_docking(protein_file, ref_file, software, exhaustiveness, n_poses)
    if 'GNINA' in docking_programs:
        gnina_docking_results_path = gnina_docking(protein_file, ref_file, software, exhaustiveness, n_poses)
    if 'PLANTS' in docking_programs:
        plants_docking_results_path = plants_docking(protein_file, ref_file, software, n_poses)
    toc = time.perf_counter()
    print(f'Finished docking in {toc-tic:0.4f}!')
    return

def smina_docking_splitted(split_file, protein_file, ref_file, software, exhaustiveness, n_poses):
    w_dir = os.path.dirname(protein_file)
    smina_folder = w_dir+'/temp/smina/'
    os.makedirs(smina_folder, exist_ok=True)
    results_path = smina_folder+os.path.basename(split_file).split('.')[0]+'_smina.sdf'
    smina_cmd = 'cd '+software+' && ./gnina -r '+protein_file+' -l '+split_file+' --autobox_ligand '+ref_file+' -o '+results_path+' --exhaustiveness ' +str(exhaustiveness)+' --num_modes '+str(n_poses)+' --cnn_scoring none'
    try:
        subprocess.call(smina_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
    except Exception as e:
        print('SMINA docking failed: '+e)
    return

def gnina_docking_splitted(split_file, protein_file, ref_file, software, exhaustiveness, n_poses):
    w_dir = os.path.dirname(protein_file)
    gnina_folder = w_dir+'/temp/gnina/'
    os.makedirs(gnina_folder, exist_ok=True)
    results_path = gnina_folder+os.path.basename(split_file).split('.')[0]+'_gnina.sdf'
    gnina_cmd = 'cd '+software+' && ./gnina -r '+protein_file+' -l '+split_file+' --autobox_ligand '+ref_file+' -o '+results_path+' --exhaustiveness ' +str(exhaustiveness)+' --num_modes '+str(n_poses)+' --cnn_scoring rescore --cnn crossdock_default2018'
    try:
        subprocess.call(gnina_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
    except Exception as e:
        print('GNINA docking failed: '+e)
    return

def plants_docking_splitted(split_file, w_dir, software, n_poses, binding_site_x, binding_site_y, binding_site_z, binding_site_radius):
    plants_docking_results_dir = w_dir+'/temp/plants/results_'+os.path.basename(split_file).replace('.sdf', '')
    #Generate plants config file
    plants_docking_config_path_txt = w_dir+'/temp/plants/config_'+os.path.basename(split_file).replace('.sdf', '.txt')
    plants_config = ['# search algorithm\n',
    'search_speed 1\n',
    'aco_ants 20\n',
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
    'protein_file '+w_dir+'/temp/plants/protein.mol2'+'\n',
    'ligand_file '+w_dir+'/temp/plants/'+os.path.basename(split_file).replace('.sdf', '.mol2')+'\n',

    '# output\n',
    'output_dir '+plants_docking_results_dir+'\n',

    '# write single mol2 files (e.g. for RMSD calculation)\n',
    'write_multi_mol2 1\n',

    '# binding site definition\n',
    'bindingsite_center '+binding_site_x+' '+binding_site_y+' '+binding_site_z+'\n',
    'bindingsite_radius '+binding_site_radius+'\n',

    '# cluster algorithm\n',
    'cluster_structures '+str(n_poses)+'\n',
    'cluster_rmsd 2.0\n',

    '# write\n',
    'write_ranking_links 0\n',
    'write_protein_bindingsite 0\n',
    'write_protein_conformations 0\n',
    'write_protein_splitted 0\n',
    'write_merged_protein 0\n',
    '####\n']
    #Write config file
    plants_docking_config_path_config = plants_docking_config_path_txt.replace('.txt', '.config')
    with open(plants_docking_config_path_config, 'w') as configwriter:
        configwriter.writelines(plants_config)
    configwriter.close()
    #Run PLANTS docking
    try:
        plants_docking_command = 'cd '+software+' && ./PLANTS --mode screen '+plants_docking_config_path_config
        subprocess.call(plants_docking_command, shell=True, stdout=DEVNULL, stderr=STDOUT)
    except Exception as e:
        print('ERROR: PLANTS docking command failed...')
        print(e)
    return

def plants_preparation(protein_file, ref_file, software):
    w_dir = os.path.dirname(protein_file)
    # Define initial variables
    plants_docking_dir = w_dir+'/temp/plants'
    #Create plants docking folder
    create_temp_folder(plants_docking_dir)
    plants_protein_mol2 = w_dir+'/temp/plants/protein.mol2'
    try:
        print('Converting protein file to .mol2 format for PLANTS docking...')
        obabel_command = 'obabel -ipdb '+protein_file+' -O '+plants_protein_mol2
        subprocess.call(obabel_command, shell=True, stdout=DEVNULL, stderr=STDOUT)
    except Exception as e:
        print('ERROR: Failed to convert protein file to .mol2!')
        print(e)
    plants_ref_mol2 = w_dir+'/temp/plants/ref.mol2'
    try:
        if ref_file.endswith('.mol2'):
            shutil.copy(ref_file, plants_docking_dir)
            os.rename(plants_docking_dir+'/'+os.path.basename(ref_file), plants_ref_mol2)
        else:
            print(f'Converting reference file from .{ref_file.split(".")[-1]} to .mol2 format for PLANTS docking...')
            obabel_command = f'obabel -i{ref_file.split(".")[-1]} {ref_file} -O {plants_ref_mol2}'
            subprocess.call(obabel_command, shell=True, stdout=DEVNULL, stderr=STDOUT)
    except Exception as e:
        print('ERROR: Failed to convert reference file to .mol2!')
        print(e)
    #Determine binding site coordinates
    try:
        print('Determining binding site coordinates using PLANTS...')
        plants_binding_site_command = 'cd '+software+' && ./PLANTS --mode bind '+plants_ref_mol2+' 8'
        run_plants_binding_site = os.popen(plants_binding_site_command)
        output_plants_binding_site = run_plants_binding_site.readlines()
        keep = []
        for l in output_plants_binding_site:
            if l.startswith('binding'):
                keep.append(l)
            else:
                pass
        binding_site_center = keep[0].split()
        binding_site_radius = keep[1].split()
        binding_site_radius = binding_site_radius[1]
        binding_site_x = binding_site_center[1]
        binding_site_y = binding_site_center[2]
        binding_site_z = binding_site_center[3].replace('+', '')
    except Exception as e:
        print('ERROR: Could not determine binding site coordinates using PLANTS')
        print(e)
    return binding_site_x, binding_site_y, binding_site_z, binding_site_radius

def fetch_poses_splitted(w_dir, n_poses, split_files_folder):
    '''
    This function is used to fetch the poses from different docking results (SMINA, GNINA, PLANTS) and create a new dataframe with the poses and their corresponding scores.
    It takes two input parameters:

    w_dir: the path of the working_directory
    n_poses: number of poses to be fetched
    The function uses the PandasTools library to load the SDF files and creates a new dataframe with the poses and scores. It also renames the columns and modifies the ID column to include the source of the pose (SMINA, GNINA, PLANTS). In case of an error, the function will print an error message.
    '''
    tic = time.perf_counter()
    print('Fetching docking poses...')
    #Fetch PLANTS poses
    plants_dataframes = []
    results_folders = [item for item in os.listdir(w_dir+'/temp/plants')]
    for item in tqdm(results_folders):
        if item.startswith('results'):
            file_path = os.path.join(w_dir+'/temp/plants', item, 'docked_ligands.mol2')
            if os.path.isfile(file_path):
                try:
                    obabel_command = f'obabel -imol2 {file_path} -O {file_path.replace(".mol2",".sdf")}'
                    subprocess.call(obabel_command, shell=True, stdout=DEVNULL, stderr=STDOUT)
                    plants_poses = PandasTools.LoadSDF(file_path.replace('.mol2','.sdf'), idName='ID', molColName='Molecule',includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True)
                    plants_scores = pd.read_csv(file_path.replace('docked_ligands.mol2','ranking.csv')).rename(columns={'LIGAND_ENTRY':'ID', 'TOTAL_SCORE':'CHEMPLP'})[['ID', 'CHEMPLP']]
                    plants_df = pd.merge(plants_scores, plants_poses, on='ID')
                    plants_df['Pose ID'] = plants_df['ID'].str.split("_").str[0] + "_PLANTS_" + plants_df['ID'].str.split("_").str[4]
                    plants_df['ID'] = plants_df['ID'].str.split("_").str[0]
                    plants_dataframes.append(plants_df)
                except Exception as e:
                    print('ERROR: Failed to convert PLANTS docking results file to .sdf!')
                    print(e)
        elif item in ['protein.mol2', 'ref.mol2']:
            pass
        else:
            Path(os.path.join(w_dir+'/temp/plants', item)).unlink(missing_ok=True)
    try:
        plants_df = pd.concat(plants_dataframes)
        PandasTools.WriteSDF(plants_df, w_dir+'/temp/plants/plants_poses.sdf', molColName='Molecule', idName='Pose ID', properties=list(plants_df.columns))
    except Exception as e:
        print('Could not combine PLANTS docking poses')
        print(e)
    else:
        for file in os.listdir(w_dir+'/temp/plants'):
            if file.startswith('results'):
                shutil.rmtree(os.path.join(w_dir+'/temp/plants', file))
    #Fetch SMINA poses
    try:
        smina_dataframes = [PandasTools.LoadSDF(w_dir+'/temp/smina/'+file, idName='ID', molColName='Molecule',includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True) for file in os.listdir(w_dir+'/temp/smina/') if file.startswith('split')]
        smina_df = pd.concat(smina_dataframes)
        list_ = [*range(1, int(n_poses)+1, 1)]
        ser = list_ * (len(smina_df) // len(list_))
        smina_df['Pose ID'] = [f"{row['ID']}_SMINA_{num}" for num, (_, row) in zip(ser + list_[:len(smina_df)-len(ser)], smina_df.iterrows())]
        smina_df.rename(columns={'minimizedAffinity':'SMINA_Affinity'}, inplace=True)
    except Exception as e:
        print('ERROR: Failed to Load SMINA poses SDF file!')
        print(e)
    try:
        PandasTools.WriteSDF(smina_df, w_dir+'/temp/smina/smina_poses.sdf', molColName='Molecule', idName='Pose ID', properties=list(smina_df.columns))
    except Exception as e:
        print('ERROR: Could not combine SMINA poses SDF file!')
        print(e)
    else:
        for file in os.listdir(w_dir+'/temp/smina'):
            if file.startswith('split'):
                os.remove(os.path.join(w_dir+'/temp/smina', file))
    #Fetch GNINA poses
    try:
        gnina_dataframes = [PandasTools.LoadSDF(w_dir+'/temp/gnina/'+file, idName='ID', molColName='Molecule',includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True) for file in os.listdir(w_dir+'/temp/gnina/') if file.startswith('split')]
        gnina_df = pd.concat(gnina_dataframes)
        list_ = [*range(1, int(n_poses)+1, 1)]
        ser = list_ * (len(gnina_df) // len(list_))
        gnina_df['Pose ID'] = [f"{row['ID']}_GNINA_{num}" for num, (_, row) in zip(ser + list_[:len(gnina_df)-len(ser)], gnina_df.iterrows())]
        gnina_df.rename(columns={'minimizedAffinity':'GNINA_Affinity'}, inplace=True)
    except Exception as e:
        print('ERROR: Failed to Load GNINA poses SDF file!')
        print(e)
    try:
        PandasTools.WriteSDF(gnina_df, w_dir+'/temp/gnina/gnina_poses.sdf', molColName='Molecule', idName='Pose ID', properties=list(gnina_df.columns))
    except Exception as e:
        print('ERROR: Could not combine GNINA docking poses')
        print(e)
    else:
        for file in os.listdir(w_dir+'/temp/gnina'):
            if file.startswith('split'):
                os.remove(os.path.join(w_dir+'/temp/gnina', file))
    #Combine all poses
    try:
        all_poses = pd.concat([plants_df, smina_df, gnina_df]) 
        PandasTools.WriteSDF(all_poses, w_dir+'/temp/allposes.sdf', molColName='Molecule', idName='Pose ID', properties=list(all_poses.columns))
    except Exception as e:
        print('Could not combine all docking poses')
        print(e)
    else:
        shutil.rmtree(os.path.join(split_files_folder))
    toc = time.perf_counter()
    print(f'Combined all docking poses in {toc-tic:0.4f}!')
    return all_poses

def docking_splitted(w_dir, protein_file, ref_file, software, docking_programs, exhaustiveness, n_poses):
    if os.path.isdir(w_dir+'/temp/split_final_library') == False :
        split_files_folder = split_sdf(w_dir, w_dir+'/temp/final_library.sdf')
    else:
        print('Split final library folder already exists...')
        split_files_folder = w_dir+'/temp/split_final_library'
    split_files_sdfs = [os.path.join(split_files_folder, f) for f in os.listdir(split_files_folder) if f.endswith('.sdf')]
    if 'PLANTS' in docking_programs:
        tic = time.perf_counter()
        binding_site_x, binding_site_y, binding_site_z, binding_site_radius = plants_preparation(protein_file, ref_file, software)
        #Convert prepared ligand file to .mol2 using open babel
        for file in os.listdir(split_files_folder):
            if file.endswith('.sdf'):
                try:
                    obabel_command = 'obabel -isdf '+split_files_folder+'/'+file+' -O '+w_dir+'/temp/plants/'+os.path.basename(file).replace('.sdf', '.mol2')
                    subprocess.call(obabel_command, shell=True, stdout=DEVNULL, stderr=STDOUT)
                except Exception as e:
                    print(f'ERROR: Failed to convert {file} to .mol2!')
                    print(e)
        print('Docking split files using PLANTS...')
        with multiprocessing.Pool(processes=(multiprocessing.cpu_count()-2)) as pool:
            pool.starmap(plants_docking_splitted, [(split_file, w_dir, software, n_poses, binding_site_x, binding_site_y, binding_site_z, binding_site_radius) for split_file in split_files_sdfs])
        toc = time.perf_counter()
        print(f'Docking with PLANTS complete in {toc-tic:0.4f}!')
    if 'SMINA' in docking_programs:
        print('Docking split files using SMINA...')
        tic = time.perf_counter()
        with multiprocessing.Pool(processes=(multiprocessing.cpu_count()-2)) as pool:
            pool.starmap(smina_docking_splitted, [(split_file, protein_file, ref_file, software, exhaustiveness, n_poses) for split_file in split_files_sdfs])
        toc = time.perf_counter()
        print(f'Docking with SMINA complete in {toc-tic:0.4f}!')
    if 'GNINA' in docking_programs:
        print('Docking split files using GNINA...')
        tic = time.perf_counter()
        with multiprocessing.Pool(processes=(multiprocessing.cpu_count()-2)) as pool:
            pool.starmap(gnina_docking_splitted, [(split_file, protein_file, ref_file, software, exhaustiveness, n_poses) for split_file in split_files_sdfs])
        toc = time.perf_counter()
        print(f'Docking with GNINA complete in {toc-tic:0.4f}!')
    return

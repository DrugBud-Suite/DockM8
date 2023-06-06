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
import concurrent.futures
from scripts.utilities import load_molecule
import glob

def qvinaw_docking(protein_file, pocket_definition, software, exhaustiveness, n_poses):
    printlog('Docking library using QVINAW...')
    tic = time.perf_counter()
    w_dir = os.path.dirname(protein_file)
    library = w_dir+'/temp/final_library.sdf'
    qvinaw_folder = os.path.join(w_dir, 'temp', 'qvinaw')
    pdbqt_files_folder = os.path.join(qvinaw_folder, 'pdbqt_files')
    try:
        os.makedirs(pdbqt_files_folder, mode=0o777, exist_ok=True)
    except Exception as e:
        print(f"Failed to create directories: {e}")

    try:
        print('Converting molecules to .pdbqt using Meeko')
        meeko_to_pdbqt(library, qvinaw_folder+'/pdbqt_files')
    except Exception as e:
        print('Failed to convert sdf file to .pdbqt')
        print(e)
        
    results_path = os.path.join(qvinaw_folder, 'docked')
    try:
        os.makedirs(results_path, mode=0o777, exist_ok=True)
    except Exception as e:
        print(f"Failed to create directories: {e}")
        
    pdbqt_files = glob.glob(os.path.join(pdbqt_files_folder, '*.pdbqt'))
    
    protein_file_pdbqt = convert_pdb_to_pdbqt(protein_file)

    for pdbqt_file in tqdm(pdbqt_files):
        qvina_cmd = (
            'cd ' + software +
            ' && ./qvina-w' +
            ' --receptor ' + protein_file_pdbqt +
            ' --ligand ' + pdbqt_file +
            ' --out ' + pdbqt_file.replace('pdbqt_files', 'docked') + 
            ' --center_x ' + str(pocket_definition["center"][0]) +
            ' --center_y ' + str(pocket_definition["center"][1]) +
            ' --center_z ' + str(pocket_definition["center"][2]) +
            ' --size_x ' + str(pocket_definition["size"][0]) +
            ' --size_y ' + str(pocket_definition["size"][1]) +
            ' --size_z ' + str(pocket_definition["size"][2]) +
            ' --exhaustiveness ' + str(exhaustiveness) +
            ' --cpu 1' +
            ' --num_modes ' + str(n_poses)
        )
        try:
            subprocess.call(qvina_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
        except Exception as e:
            printlog('QVINAW docking failed: '+e)
    toc = time.perf_counter()
    printlog(f'Docking with QVINAW complete in {toc-tic:0.4f}!')
    tic = time.perf_counter()
    qvinaw_docking_results = w_dir+"/temp/qvinaw/qvinaw_poses.sdf"
    printlog('Fetching QVINAW poses...')
    
    results_pdbqt_files = glob.glob(results_path + '/*.pdbqt')

    for pdbqt_file in results_pdbqt_files:
        # Convert to sdf
        sdf_file = pdbqt_file.replace('.pdbqt', '.sdf')
        obabel_command = f'obabel {pdbqt_file} -O {sdf_file}'
        try:
            subprocess.call(obabel_command, shell=True, stdout=DEVNULL, stderr=STDOUT)
        except Exception as e:
            print(f'Conversion from PDBQT to SDF failed: {e}')
    try:
        sdf_files = glob.glob(results_path + '/*.sdf')
        qvinaw_poses = []
        for sdf in sdf_files:
            df = PandasTools.LoadSDF(sdf, idName='ID', molColName='Molecule', includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=False)
            list_ = [*range(1, int(n_poses)+1, 1)]
            ser = list_ * (len(df) // len(list_))
            df['Pose ID'] = [f"{sdf.replace(results_path+'/', '').replace('.sdf', '')}_QVINAW_{num}" for num, (_, row) in zip(ser + list_[:len(df)-len(ser)], df.iterrows())]
            df = df.rename(columns={'REMARK': 'QVINAW_Affinity'})[['Molecule', 'QVINAW_Affinity', 'Pose ID']]
            df['QVINAW_Affinity'] = df['QVINAW_Affinity'].str.split().str[2]
            qvinaw_poses.append(df)
        qvinaw_poses = pd.concat(qvinaw_poses)
        PandasTools.WriteSDF(qvinaw_poses, qvinaw_docking_results, molColName='Molecule', idName='Pose ID', properties=list(qvinaw_poses.columns))
    except Exception as e:
        printlog('ERROR: Failed to combine QVINAW SDF file!')
        printlog(e)
    else:
        shutil.rmtree(os.path.join(pdbqt_files_folder), ignore_errors=True)
        shutil.rmtree(os.path.join(results_path), ignore_errors=True)
    return w_dir+"/temp/qvinaw/qvinaw_poses.sdf"

def qvina2_docking(protein_file, pocket_definition, software, exhaustiveness, n_poses):
    printlog('Docking library using QVINA2...')
    tic = time.perf_counter()
    w_dir = os.path.dirname(protein_file)
    library = w_dir+'/temp/final_library.sdf'
    qvina_folder = os.path.join(w_dir, 'temp', 'qvina2')
    pdbqt_files_folder = os.path.join(qvina_folder, 'pdbqt_files')
    try:
        os.makedirs(pdbqt_files_folder, mode=0o777, exist_ok=True)
    except Exception as e:
        print(f"Failed to create directories: {e}")

    try:
        print('Converting molecules to .pdbqt using Meeko')
        meeko_to_pdbqt(library, qvina_folder+'/pdbqt_files')
    except Exception as e:
        print('Failed to convert sdf file to .pdbqt')
        print(e)
        
    results_path = os.path.join(qvina_folder, 'docked')
    try:
        os.makedirs(results_path, mode=0o777, exist_ok=True)
    except Exception as e:
        print(f"Failed to create directories: {e}")
        
    pdbqt_files = glob.glob(os.path.join(pdbqt_files_folder, '*.pdbqt'))
    
    protein_file_pdbqt = convert_pdb_to_pdbqt(protein_file)

    for pdbqt_file in tqdm(pdbqt_files):
        qvina_cmd = (
            'cd ' + software +
            ' && ./qvina2.1' +
            ' --receptor ' + protein_file_pdbqt +
            ' --ligand ' + pdbqt_file +
            ' --out ' + pdbqt_file.replace('pdbqt_files', 'docked') + 
            ' --center_x ' + str(pocket_definition["center"][0]) +
            ' --center_y ' + str(pocket_definition["center"][1]) +
            ' --center_z ' + str(pocket_definition["center"][2]) +
            ' --size_x ' + str(pocket_definition["size"][0]) +
            ' --size_y ' + str(pocket_definition["size"][1]) +
            ' --size_z ' + str(pocket_definition["size"][2]) +
            ' --exhaustiveness ' + str(exhaustiveness) +
            ' --cpu 1' +
            ' --num_modes ' + str(n_poses)
        )
        try:
            subprocess.call(qvina_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
        except Exception as e:
            printlog('QVINA2 docking failed: '+e)
    toc = time.perf_counter()
    printlog(f'Docking with QVINA2 complete in {toc-tic:0.4f}!')
    tic = time.perf_counter()
    printlog('Fetching QVINA2 poses...')
    
    results_pdbqt_files = glob.glob(results_path + '/*.pdbqt')

    for pdbqt_file in results_pdbqt_files:
        # Convert to sdf
        sdf_file = pdbqt_file.replace('.pdbqt', '.sdf')
        obabel_command = f'obabel {pdbqt_file} -O {sdf_file}'
        try:
            subprocess.call(obabel_command, shell=True, stdout=DEVNULL, stderr=STDOUT)
        except Exception as e:
            print(f'Conversion from PDBQT to SDF failed: {e}')
    try:
        sdf_files = glob.glob(results_path + '/*.sdf')
        qvina2_poses = []
        for sdf in sdf_files:
            df = PandasTools.LoadSDF(sdf, idName='ID', molColName='Molecule', includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=False)
            list_ = [*range(1, int(n_poses)+1, 1)]
            ser = list_ * (len(df) // len(list_))
            df['Pose ID'] = [f"{sdf.replace(results_path+'/', '').replace('.sdf', '')}_QVINA2_{num}" for num, (_, row) in zip(ser + list_[:len(df)-len(ser)], df.iterrows())]
            df = df.rename(columns={'REMARK': 'QVINA2_Affinity'})[['Molecule', 'QVINA2_Affinity', 'Pose ID']]
            df['QVINA2_Affinity'] = df['QVINA2_Affinity'].str.split().str[2]
            qvina2_poses.append(df)
        qvina2_poses = pd.concat(qvina2_poses)
        PandasTools.WriteSDF(qvina2_poses, w_dir+"/temp/qvina2/qvina2_poses.sdf", molColName='Molecule', idName='Pose ID', properties=list(qvina2_poses.columns))
    except Exception as e:
        printlog('ERROR: Failed to combine QVINA2 SDF file!')
        printlog(e)
    else:
        shutil.rmtree(os.path.join(pdbqt_files_folder), ignore_errors=True)
        shutil.rmtree(os.path.join(results_path), ignore_errors=True)
    return w_dir+"/temp/qvina2/qvina2_poses.sdf"

def smina_docking(protein_file, pocket_definition, software, exhaustiveness, n_poses):
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
    printlog('Docking library using SMINA...')
    tic = time.perf_counter()
    w_dir = os.path.dirname(protein_file)
    library = w_dir+'/temp/final_library.sdf'
    smina_folder = w_dir+'/temp/smina/'
    try:
        os.mkdir(smina_folder, mode = 0o777)
    except:
        printlog('SMINA folder already exists')
    results_path = smina_folder+'docked.sdf'
    log = smina_folder+'log.txt'
    smina_cmd = (
            'cd ' + software +
            ' && ./gnina' +
            ' --receptor ' + protein_file +
            ' --ligand ' + library +
            ' --out ' + results_path + 
            ' --center_x ' + str(pocket_definition["center"][0]) +
            ' --center_y ' + str(pocket_definition["center"][1]) +
            ' --center_z ' + str(pocket_definition["center"][2]) +
            ' --size_x ' + str(pocket_definition["size"][0]) +
            ' --size_y ' + str(pocket_definition["size"][1]) +
            ' --size_z ' + str(pocket_definition["size"][2]) +
            ' --exhaustiveness ' + str(exhaustiveness) +
            ' --cpu 1' +
            ' --num_modes ' + str(n_poses) +
            ' --log ' + log +
            ' --cnn_scoring none --no_gpu'
        )
    try:
        subprocess.call(smina_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
    except Exception as e:
        printlog('SMINA docking failed: '+e)
    toc = time.perf_counter()
    printlog(f'Docking with SMINA complete in {toc-tic:0.4f}!')
    
    tic = time.perf_counter()
    smina_docking_results = w_dir+"/temp/smina/docked.sdf"
    printlog('Fetching SMINA poses...')
    try:
        smina_df = PandasTools.LoadSDF(smina_docking_results, idName='ID', molColName='Molecule',includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True)
        list_ = [*range(1, int(n_poses)+1, 1)]
        ser = list_ * (len(smina_df) // len(list_))
        smina_df['Pose ID'] = [f"{row['ID']}_SMINA_{num}" for num, (_, row) in zip(ser + list_[:len(smina_df)-len(ser)], smina_df.iterrows())]
        smina_df.rename(columns={'minimizedAffinity':'SMINA_Affinity'}, inplace=True)
    except Exception as e:
        printlog('ERROR: Failed to Load SMINA poses SDF file!')
        printlog(e)
    try:
        PandasTools.WriteSDF(smina_df, w_dir+'/temp/smina/smina_poses.sdf', molColName='Molecule', idName='Pose ID', properties=list(smina_df.columns))
        toc = time.perf_counter()
        printlog(f'Cleaned up SMINA poses in {toc-tic:0.4f}!')
    except Exception as e:
        printlog('ERROR: Failed to Write SMINA poses SDF file!')
        printlog(e)
    return w_dir+'/temp/smina/smina_poses.sdf'

def gnina_docking(protein_file, pocket_definition, software, exhaustiveness, n_poses):
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
    printlog('Docking library using GNINA...')
    tic = time.perf_counter()
    w_dir = os.path.dirname(protein_file)
    library = w_dir+'/temp/final_library.sdf'
    gnina_folder = w_dir+'/temp/gnina/'
    try:
        os.mkdir(gnina_folder, mode = 0o777)
    except:
        printlog('GNINA folder already exists')
    results_path = gnina_folder+'docked.sdf'
    log = gnina_folder+'log.txt'
    gnina_cmd = (
            'cd ' + software +
            ' && ./gnina' +
            ' --receptor ' + protein_file +
            ' --ligand ' + library +
            ' --out ' + results_path + 
            ' --center_x ' + str(pocket_definition["center"][0]) +
            ' --center_y ' + str(pocket_definition["center"][1]) +
            ' --center_z ' + str(pocket_definition["center"][2]) +
            ' --size_x ' + str(pocket_definition["size"][0]) +
            ' --size_y ' + str(pocket_definition["size"][1]) +
            ' --size_z ' + str(pocket_definition["size"][2]) +
            ' --exhaustiveness ' + str(exhaustiveness) +
            ' --cpu 1' +
            ' --num_modes ' + str(n_poses) +
            ' --log ' + log +
            ' --cnn_scoring rescore --cnn crossdock_default2018 --no_gpu'
        )
    try:
        subprocess.call(gnina_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
    except Exception as e:
        printlog('GNINA docking failed: '+e)
    toc = time.perf_counter()
    printlog(f'Docking with GNINA complete in {toc-tic:0.4f}!')
    
    tic = time.perf_counter()
    gnina_docking_results = w_dir+"/temp/gnina/docked.sdf"
    printlog('Fetching GNINA poses...')
    try:
        gnina_df = PandasTools.LoadSDF(gnina_docking_results, idName='ID', molColName='Molecule',includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True)
        list_ = [*range(1, int(n_poses)+1, 1)]
        ser = list_ * (len(gnina_df) // len(list_))
        gnina_df['Pose ID'] = [f"{row['ID']}_GNINA_{num}" for num, (_, row) in zip(ser + list_[:len(gnina_df)-len(ser)], gnina_df.iterrows())]
        gnina_df.rename(columns={'minimizedAffinity':'GNINA_Affinity'}, inplace=True)
    except Exception as e:
        printlog('ERROR: Failed to Load GNINA poses SDF file!')
        printlog(e)
    try:
        PandasTools.WriteSDF(gnina_df, w_dir+'/temp/gnina/gnina_poses.sdf', molColName='Molecule', idName='Pose ID', properties=list(gnina_df.columns))
        toc = time.perf_counter()
        printlog(f'Cleaned up GNINA poses in {toc-tic:0.4f}!')
    except Exception as e:
        printlog('ERROR: Failed to Write GNINA poses SDF file!')
        printlog(e)
    return w_dir+'/temp/gnina/gnina_poses.sdf'

def plants_docking(protein_file, pocket_definition, software, n_poses):
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
    printlog('Docking library using PLANTS...')
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
        printlog('Plants docking folder already exists')
    else:
        os.mkdir(plants_docking_dir)
    #Convert protein file to .mol2 using open babel
    plants_protein_mol2 = w_dir+'/temp/plants/protein.mol2'
    try:
        printlog('Converting protein file to .mol2 format for PLANTS docking...')
        obabel_command = 'obabel -ipdb '+protein_file+' -O '+plants_protein_mol2
        subprocess.call(obabel_command, shell=True, stdout=DEVNULL, stderr=STDOUT)
    except Exception as e:
        printlog('ERROR: Failed to convert protein file to .mol2!')
        printlog(e)
    #Convert prepared ligand file to .mol2 using open babel
    final_library = w_dir+'/temp/final_library.sdf'
    plants_library_mol2 = w_dir+'/temp/plants/ligands.mol2'
    try:
        obabel_command = 'obabel -isdf '+final_library+' -O '+plants_library_mol2
        os.system(obabel_command)
    except Exception as e:
        printlog('ERROR: Failed to convert docking library file to .mol2!')
        printlog(e)
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
    'bindingsite_center '+str(pocket_definition["center"][0])+' '+str(pocket_definition["center"][1])+' '+(pocket_definition["center"][2])+'+\n',
    'bindingsite_radius '+str(pocket_definition["size"][0]/2)+'\n',

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
    printlog('Writing PLANTS config file...')
    plants_docking_config_path_config = plants_docking_config_path_txt.replace(".txt", ".config")
    with open(plants_docking_config_path_config, 'w') as configwriter:
        configwriter.writelines(plants_config)
    configwriter.close()
    #Run PLANTS docking
    try:
        printlog('Starting PLANTS docking...')
        plants_docking_command = "cd "+software+" && ./PLANTS --mode screen "+plants_docking_config_path_config
        subprocess.call(plants_docking_command, shell=True, stdout=DEVNULL, stderr=STDOUT)
    except Exception as e:
        printlog('ERROR: PLANTS docking command failed...')
        printlog(e)
    plants_docking_results_mol2 = w_dir+"/temp/plants/results/docked_ligands.mol2"
    plants_docking_results_sdf = plants_docking_results_mol2.replace(".mol2", ".sdf")
    # Convert PLANTS poses to sdf
    try:
        printlog('Converting PLANTS poses to .sdf format...')
        obabel_command = 'obabel -imol2 '+plants_docking_results_mol2+' -O '+plants_docking_results_sdf 
        subprocess.call(obabel_command, shell=True, stdout=DEVNULL, stderr=STDOUT)
    except Exception as e:
        printlog('ERROR: Failed to convert PLANTS poses file to .sdf!')
        printlog(e)
    toc = time.perf_counter()
    printlog(f'Docking with PLANTS complete in {toc-tic:0.4f}!')
    plants_poses_results_sdf = w_dir+"/temp/plants/results/docked_ligands.sdf"
    plants_scoring_results_sdf = w_dir+"/temp/plants/results/ranking.csv"
    #Fetch PLANTS poses
    printlog('Fetching PLANTS poses...')
    try:
        plants_poses = PandasTools.LoadSDF(plants_poses_results_sdf, idName='ID', molColName='Molecule',includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True)
        plants_scores = pd.read_csv(plants_scoring_results_sdf, usecols=['LIGAND_ENTRY', 'TOTAL_SCORE'])
        plants_scores = plants_scores.rename(columns={'LIGAND_ENTRY':'ID', 'TOTAL_SCORE':'CHEMPLP'})
        plants_scores = plants_scores[['ID', 'CHEMPLP']]
        plants_df = pd.merge(plants_scores, plants_poses, on='ID')
        plants_df['Pose ID'] = plants_df['ID'].str.split("_").str[0] + "_PLANTS_" + plants_df['ID'].str.split("_").str[4]
        plants_df['ID'] = plants_df['ID'].str.split("_").str[0]
    except Exception as e:
        printlog('ERROR: Failed to Load PLANTS poses SDF file!')
        printlog(e)
    try:
        PandasTools.WriteSDF(plants_df, w_dir+'/temp/plants/plants_poses.sdf', molColName='Molecule', idName='Pose ID', properties=list(plants_df.columns))
        toc = time.perf_counter()
        printlog(f'Cleaned up PLANTS poses in {toc-tic:0.4f}!')
    except Exception as e:
        printlog('ERROR: Failed to Write PLANTS poses SDF file!')
        printlog(e)
    return w_dir+'/temp/plants/plants_poses.sdf'

def docking(w_dir, protein_file, ref_file, software, docking_programs, exhaustiveness, n_poses, ncpus, pocket_definition):
    if ncpus == 1:
        tic = time.perf_counter()
        if 'SMINA' in docking_programs and os.path.isdir(w_dir+'/temp/smina') == False:
            smina_docking_results_path = smina_docking(protein_file, pocket_definition, software, exhaustiveness, n_poses)
        if 'GNINA' in docking_programs and os.path.isdir(w_dir+'/temp/gnina') == False:
            gnina_docking_results_path = gnina_docking(protein_file, pocket_definition, software, exhaustiveness, n_poses)
        if 'PLANTS' in docking_programs and os.path.isdir(w_dir+'/temp/plants') == False:
            plants_docking_results_path = plants_docking(protein_file, pocket_definition, software, n_poses)
        if 'QVINAW' in docking_programs and os.path.isdir(w_dir+'/temp/qvinaw') == False:
            qvinaw_docking_results_path = qvinaw_docking(protein_file, pocket_definition, software, exhaustiveness, n_poses)
        if 'QVINA2' in docking_programs and os.path.isdir(w_dir+'/temp/qvina2') == False:
            qvina2_docking_results_path = qvina2_docking(protein_file, pocket_definition, software, exhaustiveness, n_poses)
        toc = time.perf_counter()
        printlog(f'Finished docking in {toc-tic:0.4f}!')
        #Load all poses
        if os.path.isfile(w_dir+"/temp/allposes.sdf") == False:
            try:
                all_poses_list = []
                if 'GNINA' in docking_programs and os.path.isdir(w_dir+'/temp/gnina') == True:
                    gnina_df = PandasTools.LoadSDF(w_dir+'/temp/gnina/gnina_poses.sdf', idName='Pose ID', molColName='Molecule',includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True)
                    all_poses_list.append(gnina_df)
                if 'SMINA' in docking_programs and os.path.isdir(w_dir+'/temp/smina') == True:
                    smina_df = PandasTools.LoadSDF(w_dir+'/temp/smina/smina_poses.sdf', idName='Pose ID', molColName='Molecule',includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True)
                    all_poses_list.append(smina_df)
                if 'PLANTS' in docking_programs and os.path.isdir(w_dir+'/temp/plants') == True:
                    plants_df = PandasTools.LoadSDF(w_dir+'/temp/plants/plants_poses.sdf', idName='Pose ID', molColName='Molecule',includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True)
                    all_poses_list.append(plants_df)
                if 'QVINAW' in docking_programs and os.path.isdir(w_dir+'/temp/qvinaw') == True:
                    qvinaw_df = PandasTools.LoadSDF(w_dir+'/temp/qvinaw/qvinaw_poses.sdf', idName='Pose ID', molColName='Molecule',includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True)
                    all_poses_list.append(qvinaw_df)
                if 'QVINA2' in docking_programs and os.path.isdir(w_dir+'/temp/qvina2') == True:
                    qvina2_df = PandasTools.LoadSDF(w_dir+'/temp/qvina2/qvina2_poses.sdf', idName='Pose ID', molColName='Molecule',includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True)
                    all_poses_list.append(qvina2_df)
                all_poses = pd.concat(all_poses_list)
                PandasTools.WriteSDF(all_poses, w_dir+"/temp/allposes.sdf", molColName='Molecule', idName='Pose ID', properties=list(all_poses.columns))
            except Exception as e:
                printlog('ERROR: Failed to combine all poses!')
                printlog(e)
        else:
            pass
    else:
            if os.path.isdir(w_dir+'/temp/split_final_library') == False :
                split_files_folder = split_sdf(w_dir+'/temp', w_dir+'/temp/final_library.sdf', ncpus)
            else:
                printlog('Split final library folder already exists...')
                split_files_folder = w_dir+'/temp/split_final_library'
            split_files_sdfs = [os.path.join(split_files_folder, f) for f in os.listdir(split_files_folder) if f.endswith('.sdf')]
            if 'PLANTS' in docking_programs and os.path.isdir(w_dir+'/temp/plants') == False:
                tic = time.perf_counter()
                #Convert prepared ligand file to .mol2 using open babel
                for file in os.listdir(split_files_folder):
                    if file.endswith('.sdf'):
                        try:
                            obabel_command = 'obabel -isdf '+split_files_folder+'/'+file+' -O '+w_dir+'/temp/plants/'+os.path.basename(file).replace('.sdf', '.mol2')
                            subprocess.call(obabel_command, shell=True, stdout=DEVNULL, stderr=STDOUT)
                        except Exception as e:
                            printlog(f'ERROR: Failed to convert {file} to .mol2!')
                            printlog(e)
                printlog('Docking split files using PLANTS...')
                with concurrent.futures.ProcessPoolExecutor(max_workers=ncpus) as executor:
                    jobs = []
                    for split_file in tqdm(split_files_sdfs, desc = 'Submitting PLANTS jobs', unit='Jobs'):
                        try:
                            job = executor.submit(plants_docking_splitted, split_file, w_dir, software, n_poses, pocket_definition)
                            jobs.append(job)
                        except Exception as e:
                            printlog("Error in concurrent futures job creation: "+ str(e))
                    for job in tqdm(concurrent.futures.as_completed(jobs), desc="Docking with PLANTS", total=len(jobs)):
                        try:
                            _ = job.result()
                        except Exception as e:
                            printlog("Error in concurrent futures job execution: "+ str(e))
                toc = time.perf_counter()
                printlog(f'Docking with PLANTS complete in {toc-tic:0.4f}!')
            #Fetch PLANTS poses
            if 'PLANTS' in docking_programs and os.path.isdir(w_dir+'/temp/plants') == True and os.path.isfile(w_dir+'/temp/plants/plants_poses.sdf') == False:
                plants_dataframes = []
                results_folders = [item for item in os.listdir(w_dir+'/temp/plants')]
                for item in tqdm(results_folders, desc='Fetching PLANTS docking poses'):
                    if item.startswith('results'):
                        file_path = os.path.join(w_dir+'/temp/plants', item, 'docked_ligands.mol2')
                        if os.path.isfile(file_path):
                            try:
                                obabel_command = f'obabel -imol2 {file_path} -O {file_path.replace(".mol2",".sdf")}'
                                subprocess.call(obabel_command, shell=True, stdout=DEVNULL, stderr=STDOUT)
                                plants_poses = PandasTools.LoadSDF(file_path.replace('.mol2','.sdf'), idName='ID', molColName='Molecule',includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True)
                                plants_scores = pd.read_csv(file_path.replace('docked_ligands.mol2','ranking.csv')).rename(columns={'LIGAND_ENTRY':'ID', 'TOTAL_SCORE':'CHEMPLP'})[['ID', 'CHEMPLP']]
                                plants_df = pd.merge(plants_scores, plants_poses, on='ID')
                                plants_df['ID'] = plants_df['ID'].str.split("_").str[0]
                                list_ = [*range(1, int(n_poses)+1, 1)]
                                ser = list_ * (len(plants_df) // len(list_))
                                plants_df['Pose ID'] = [f"{row['ID']}_PLANTS_{num}" for num, (_, row) in zip(ser + list_[:len(plants_df)-len(ser)], plants_df.iterrows())]
                                plants_dataframes.append(plants_df)
                            except Exception as e:
                                printlog('ERROR: Failed to convert PLANTS docking results file to .sdf!')
                                printlog(e)
                    elif item in ['protein.mol2', 'ref.mol2']:
                        pass
                    else:
                        Path(os.path.join(w_dir+'/temp/plants', item)).unlink(missing_ok=True)
                try:
                    plants_df = pd.concat(plants_dataframes)
                    PandasTools.WriteSDF(plants_df, w_dir+'/temp/plants/plants_poses.sdf', molColName='Molecule', idName='Pose ID', properties=list(plants_df.columns))
                except Exception as e:
                    printlog('ERROR: Failed to write combined PLANTS docking poses')
                    printlog(e)
                else:
                    for file in os.listdir(w_dir+'/temp/plants'):
                        if file.startswith('results'):
                            shutil.rmtree(os.path.join(w_dir+'/temp/plants', file))
            if 'SMINA' in docking_programs and os.path.isdir(w_dir+'/temp/smina') == False:
                printlog('Docking split files using SMINA...')
                tic = time.perf_counter()
                with concurrent.futures.ProcessPoolExecutor(max_workers=ncpus) as executor:
                    jobs = []
                    for split_file in tqdm(split_files_sdfs, desc = 'Submitting SMINA jobs', unit='Jobs'):
                        try:
                            job = executor.submit(smina_docking_splitted, split_file, protein_file, pocket_definition, software, exhaustiveness, n_poses)
                            jobs.append(job)
                        except Exception as e:
                            printlog("Error in concurrent futures job creation: "+ str(e))
                    for job in tqdm(concurrent.futures.as_completed(jobs), desc="Docking with SMINA", total=len(jobs)):
                        try:
                            _ = job.result()
                        except Exception as e:
                            printlog("Error in concurrent futures job execution: "+ str(e))
                toc = time.perf_counter()
                printlog(f'Docking with SMINA complete in {toc-tic:0.4f}!')
            #Fetch SMINA poses
            if 'SMINA' in docking_programs and os.path.isdir(w_dir+'/temp/smina') == True and os.path.isfile(w_dir+'/temp/smina/smina_poses.sdf') == False:
                try:
                    smina_dataframes = []
                    for file in tqdm(os.listdir(w_dir+'/temp/smina/'), desc="Loading SMINA poses"):
                        if file.startswith('split'):
                            df = PandasTools.LoadSDF(w_dir+'/temp/smina/'+file, idName='ID', molColName='Molecule',includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True)
                            smina_dataframes.append(df)
                    smina_df = pd.concat(smina_dataframes)
                    list_ = [*range(1, int(n_poses)+1, 1)]
                    ser = list_ * (len(smina_df) // len(list_))
                    smina_df['Pose ID'] = [f"{row['ID']}_SMINA_{num}" for num, (_, row) in zip(ser + list_[:len(smina_df)-len(ser)], smina_df.iterrows())]
                    smina_df.rename(columns={'minimizedAffinity':'SMINA_Affinity'}, inplace=True)
                except Exception as e:
                    printlog('ERROR: Failed to Load SMINA poses SDF file!')
                    printlog(e)
                try:
                    PandasTools.WriteSDF(smina_df, w_dir+'/temp/smina/smina_poses.sdf', molColName='Molecule', idName='Pose ID', properties=list(smina_df.columns))
                except Exception as e:
                    printlog('ERROR: Failed to write combined SMINA poses SDF file!')
                    printlog(e)
                else:
                    for file in os.listdir(w_dir+'/temp/smina'):
                        if file.startswith('split'):
                            os.remove(os.path.join(w_dir+'/temp/smina', file))
            if 'GNINA' in docking_programs and os.path.isdir(w_dir+'/temp/gnina') == False:
                printlog('Docking split files using GNINA...')
                tic = time.perf_counter()
                with concurrent.futures.ProcessPoolExecutor(max_workers=ncpus) as executor:
                    jobs = []
                    for split_file in tqdm(split_files_sdfs, desc = 'Submitting GNINA jobs', unit='Jobs'):
                        try:
                            job = executor.submit(gnina_docking_splitted, split_file, protein_file, pocket_definition, software, exhaustiveness, n_poses)
                            jobs.append(job)
                        except Exception as e:
                            printlog("Error in concurrent futures job creation: "+ str(e))
                    for job in tqdm(concurrent.futures.as_completed(jobs), desc="Docking with GNINA", total=len(jobs)):
                        try:
                            _ = job.result()
                        except Exception as e:
                            printlog("Error in concurrent futures job execution: "+ str(e))
                toc = time.perf_counter()
                printlog(f'Docking with GNINA complete in {toc-tic:0.4f}!')
            #Fetch GNINA poses
            if 'GNINA' in docking_programs and os.path.isdir(w_dir+'/temp/gnina') == True and os.path.isfile(w_dir+'/temp/gnina/gnina_poses.sdf') == False:
                try:
                    gnina_dataframes = []
                    for file in tqdm(os.listdir(w_dir+'/temp/gnina/'), desc="Loading GNINA poses"):
                        if file.startswith('split'):
                            df = PandasTools.LoadSDF(w_dir+'/temp/gnina/'+file, idName='ID', molColName='Molecule',includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True)
                            gnina_dataframes.append(df)
                    gnina_df = pd.concat(gnina_dataframes)
                    list_ = [*range(1, int(n_poses)+1, 1)]
                    ser = list_ * (len(gnina_df) // len(list_))
                    gnina_df['Pose ID'] = [f"{row['ID']}_GNINA_{num}" for num, (_, row) in zip(ser + list_[:len(gnina_df)-len(ser)], gnina_df.iterrows())]
                    gnina_df.rename(columns={'minimizedAffinity':'GNINA_Affinity'}, inplace=True)
                except Exception as e:
                    printlog('ERROR: Failed to Load GNINA poses SDF file!')
                    printlog(e)
                try:
                    PandasTools.WriteSDF(gnina_df, w_dir+'/temp/gnina/gnina_poses.sdf', molColName='Molecule', idName='Pose ID', properties=list(gnina_df.columns))
                except Exception as e:
                    printlog('ERROR: Failed to write combined GNINA docking poses')
                    printlog(e)
                else:
                    for file in os.listdir(w_dir+'/temp/gnina'):
                        if file.startswith('split'):
                            os.remove(os.path.join(w_dir+'/temp/gnina', file))
            if 'QVINAW' in docking_programs and os.path.isdir(w_dir+'/temp/qvinaw') == False:
                printlog('Docking split files using QVINAW...')
                tic = time.perf_counter()
                protein_file_pdbqt = convert_pdb_to_pdbqt(protein_file)
                with concurrent.futures.ProcessPoolExecutor(max_workers=ncpus) as executor:
                    jobs = []
                    for split_file in tqdm(split_files_sdfs, desc = 'Submitting QVINAW jobs', unit='Jobs'):
                        try:
                            job = executor.submit(qvinaw_docking_splitted, split_file, protein_file_pdbqt, pocket_definition, software, exhaustiveness, n_poses)
                            jobs.append(job)
                        except Exception as e:
                            printlog("Error in concurrent futures job creation: "+ str(e))
                    for job in tqdm(concurrent.futures.as_completed(jobs), desc="Docking with QVINAW", total=len(jobs)):
                        try:
                            _ = job.result()
                        except Exception as e:
                            printlog("Error in concurrent futures job execution: "+ str(e))
                toc = time.perf_counter()
                printlog(f'Docking with QVINAW complete in {toc-tic:0.4f}!')
            #Fetch QVINAW poses
            if 'QVINAW' in docking_programs and os.path.isdir(w_dir+'/temp/qvinaw') == True and os.path.isfile(w_dir+'/temp/qvinaw/qvinaw_poses.sdf') == False:
                try:
                    qvinaw_dataframes = []
                    for file in tqdm(os.listdir(w_dir+'/temp/qvinaw/'), desc="Loading QVINAW poses"):
                        if file.startswith('split'):
                            df = PandasTools.LoadSDF(w_dir+'/temp/qvinaw/'+file, idName='Pose ID', molColName='Molecule',includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True)
                            qvinaw_dataframes.append(df)
                    qvinaw_df = pd.concat(qvinaw_dataframes)
                except Exception as e:
                    printlog('ERROR: Failed to Load QVINAW poses SDF file!')
                    printlog(e)
                try:
                    PandasTools.WriteSDF(qvinaw_df, w_dir+'/temp/qvinaw/qvinaw_poses.sdf', molColName='Molecule', idName='Pose ID', properties=list(qvinaw_df.columns))
                except Exception as e:
                    printlog('ERROR: Failed to write combined QVINAW poses SDF file!')
                    printlog(e)
                else:
                    for file in os.listdir(w_dir+'/temp/qvinaw'):
                        if file.startswith('split'):
                            os.remove(os.path.join(w_dir+'/temp/qvinaw', file))
            if 'QVINA2' in docking_programs and os.path.isdir(w_dir+'/temp/qvina2') == False:
                printlog('Docking split files using QVINA2...')
                tic = time.perf_counter()
                protein_file_pdbqt = convert_pdb_to_pdbqt(protein_file)
                with concurrent.futures.ProcessPoolExecutor(max_workers=ncpus) as executor:
                    jobs = []
                    for split_file in tqdm(split_files_sdfs, desc = 'Submitting QVINA2 jobs', unit='Jobs'):
                        try:
                            job = executor.submit(qvina2_docking_splitted, split_file, protein_file_pdbqt, pocket_definition, software, exhaustiveness, n_poses)
                            jobs.append(job)
                        except Exception as e:
                            printlog("Error in concurrent futures job creation: "+ str(e))
                    for job in tqdm(concurrent.futures.as_completed(jobs), desc="Docking with QVINA2", total=len(jobs)):
                        try:
                            _ = job.result()
                        except Exception as e:
                            printlog("Error in concurrent futures job execution: "+ str(e))
                toc = time.perf_counter()
                printlog(f'Docking with QVINA2 complete in {toc-tic:0.4f}!')
            #Fetch QVINa2 poses
            if 'QVINA2' in docking_programs and os.path.isdir(w_dir+'/temp/qvina2') == True and os.path.isfile(w_dir+'/temp/qvina2/qvina2_poses.sdf') == False:
                try:
                    qvina2_dataframes = []
                    for file in tqdm(os.listdir(w_dir+'/temp/qvina2/'), desc="Loading QVINA2 poses"):
                        if file.startswith('split'):
                            df = PandasTools.LoadSDF(w_dir+'/temp/qvina2/'+file, idName='Pose ID', molColName='Molecule',includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True)
                            qvina2_dataframes.append(df)
                    qvina2_df = pd.concat(qvina2_dataframes)
                except Exception as e:
                    printlog('ERROR: Failed to Load QVINa2 poses SDF file!')
                    printlog(e)
                try:
                    PandasTools.WriteSDF(qvina2_df, w_dir+'/temp/qvina2/qvina2_poses.sdf', molColName='Molecule', idName='Pose ID', properties=list(qvina2_df.columns))
                except Exception as e:
                    printlog('ERROR: Failed to write combined QVINA2 poses SDF file!')
                    printlog(e)
                else:
                    for file in os.listdir(w_dir+'/temp/qvina2'):
                        if file.startswith('split'):
                            os.remove(os.path.join(w_dir+'/temp/qvina2', file))
            #Load all poses
            printlog('Combining all poses...')
            all_poses = pd.DataFrame()
            if 'GNINA' in docking_programs and os.path.isdir(w_dir+'/temp/gnina') == True and os.path.isfile(w_dir+'/temp/gnina/gnina_poses.sdf') == True:
                try:
                    gnina_df = PandasTools.LoadSDF(w_dir+'/temp/gnina/gnina_poses.sdf', idName='Pose ID', molColName='Molecule',includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True)
                    all_poses = pd.concat([all_poses, gnina_df])
                except Exception as e:
                    printlog('ERROR: Failed to load GNINA poses for combining!')
                    printlog(e)
            if 'SMINA' in docking_programs and os.path.isdir(w_dir+'/temp/smina') == True and os.path.isfile(w_dir+'/temp/smina/smina_poses.sdf') == True:
                try:
                    smina_df = PandasTools.LoadSDF(w_dir+'/temp/smina/smina_poses.sdf', idName='Pose ID', molColName='Molecule',includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True)
                    all_poses = pd.concat([all_poses, smina_df])
                except Exception as e:
                    printlog('ERROR: Failed to load SMINA poses for combining!')
                    printlog(e)
            if 'PLANTS' in docking_programs and os.path.isdir(w_dir+'/temp/plants') == True and os.path.isfile(w_dir+'/temp/plants/plants_poses.sdf') == True:
                try:
                    plants_df = PandasTools.LoadSDF(w_dir+'/temp/plants/plants_poses.sdf', idName='Pose ID', molColName='Molecule',includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True)
                    all_poses = pd.concat([all_poses, plants_df])
                except Exception as e:
                    printlog('ERROR: Failed to load PLANTS poses for combining!')
                    printlog(e)
            if 'QVINAW' in docking_programs and os.path.isdir(w_dir+'/temp/qvinaw') == True and os.path.isfile(w_dir+'/temp/qvinaw/qvinaw_poses.sdf') == True:
                try:
                    qvinaw_df = PandasTools.LoadSDF(w_dir+'/temp/qvinaw/qvinaw_poses.sdf', idName='Pose ID', molColName='Molecule',includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True)
                    all_poses = pd.concat([all_poses, qvinaw_df])
                except Exception as e:
                    printlog('ERROR: Failed to load QVINAW poses for combining!')
                    printlog(e)
            if 'QVINA2' in docking_programs and os.path.isdir(w_dir+'/temp/qvina2') == True and os.path.isfile(w_dir+'/temp/qvina2/qvina2_poses.sdf') == True:
                try:
                    qvina2_df = PandasTools.LoadSDF(w_dir+'/temp/qvina2/qvina2_poses.sdf', idName='Pose ID', molColName='Molecule',includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True)
                    all_poses = pd.concat([all_poses, qvina2_df])
                except Exception as e:
                    printlog('ERROR: Failed to load QVINA2 poses for combining!')
                    printlog(e)
            try:
                PandasTools.WriteSDF(all_poses, w_dir+"/temp/allposes.sdf", molColName='Molecule', idName='Pose ID', properties=list(all_poses.columns))
                printlog('All poses succesfully combined!')
            except Exception as e:
                printlog('ERROR: Failed to write all_poses SDF file!')
                printlog(e)
            else:
                shutil.rmtree(os.path.join(split_files_folder), ignore_errors=True)
    return

def smina_docking_splitted(split_file, protein_file, pocket_definition, software, exhaustiveness, n_poses):
    w_dir = os.path.dirname(protein_file)
    smina_folder = w_dir+'/temp/smina/'
    os.makedirs(smina_folder, exist_ok=True)
    results_path = smina_folder+os.path.basename(split_file).split('.')[0]+'_smina.sdf'
    smina_cmd = (
            'cd ' + software +
            ' && ./gnina' +
            ' --receptor ' + protein_file +
            ' --ligand ' + split_file +
            ' --out ' + results_path + 
            ' --center_x ' + str(pocket_definition["center"][0]) +
            ' --center_y ' + str(pocket_definition["center"][1]) +
            ' --center_z ' + str(pocket_definition["center"][2]) +
            ' --size_x ' + str(pocket_definition["size"][0]) +
            ' --size_y ' + str(pocket_definition["size"][1]) +
            ' --size_z ' + str(pocket_definition["size"][2]) +
            ' --exhaustiveness ' + str(exhaustiveness) +
            ' --cpu 1' +
            ' --num_modes ' + str(n_poses) +
            ' --cnn_scoring none --no_gpu'
        )
    try:
        subprocess.call(smina_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
    except Exception as e:
        printlog('SMINA docking failed: '+e)
    return

def gnina_docking_splitted(split_file, protein_file, pocket_definition, software, exhaustiveness, n_poses):
    w_dir = os.path.dirname(protein_file)
    gnina_folder = w_dir+'/temp/gnina/'
    os.makedirs(gnina_folder, exist_ok=True)
    results_path = gnina_folder+os.path.basename(split_file).split('.')[0]+'_gnina.sdf'
    gnina_cmd = (
            'cd ' + software +
            ' && ./gnina' +
            ' --receptor ' + protein_file +
            ' --ligand ' + split_file +
            ' --out ' + results_path + 
            ' --center_x ' + str(pocket_definition["center"][0]) +
            ' --center_y ' + str(pocket_definition["center"][1]) +
            ' --center_z ' + str(pocket_definition["center"][2]) +
            ' --size_x ' + str(pocket_definition["size"][0]) +
            ' --size_y ' + str(pocket_definition["size"][1]) +
            ' --size_z ' + str(pocket_definition["size"][2]) +
            ' --exhaustiveness ' + str(exhaustiveness) +
            ' --cpu 1' +
            ' --num_modes ' + str(n_poses) +
            ' --cnn_scoring rescore --cnn crossdock_default2018 --no_gpu'
        )
    try:
        subprocess.call(gnina_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
    except Exception as e:
        printlog('GNINA docking failed: '+e)
    return

def plants_docking_splitted(split_file, w_dir, software, n_poses, pocket_definition):
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
    'enable_sulphur_acceptors 0\n',
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
    'bindingsite_center '+str(pocket_definition["center"][0])+' '+str(pocket_definition["center"][1])+' '+(pocket_definition["center"][2])+'+\n',
    'bindingsite_radius '+str(pocket_definition["size"][0]/2)+'\n',

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
        printlog('ERROR: PLANTS docking command failed...')
        printlog(e)
    return

def qvinaw_docking_splitted(split_file, protein_file_pdbqt, pocket_definition, software, exhaustiveness, n_poses):
    w_dir = os.path.dirname(protein_file_pdbqt)
    qvinaw_folder = os.path.join(w_dir, 'temp', 'qvinaw')
    pdbqt_files_folder = os.path.join(qvinaw_folder, os.path.basename(split_file).split('.')[0], 'pdbqt_files')
    try:
        os.makedirs(pdbqt_files_folder, mode=0o777, exist_ok=True)
    except Exception as e:
        print(f"Failed to create directories: {e}")

    try:
        meeko_to_pdbqt(split_file, pdbqt_files_folder)
    except Exception as e:
        print('Failed to convert sdf file to .pdbqt')
        print(e)
        
    results_path = os.path.join(qvinaw_folder, os.path.basename(split_file).split('.')[0], 'docked')
    try:
        os.makedirs(results_path, mode=0o777, exist_ok=True)
    except Exception as e:
        print(f"Failed to create directories: {e}")
        
    pdbqt_files = glob.glob(os.path.join(pdbqt_files_folder, '*.pdbqt'))

    for pdbqt_file in pdbqt_files:
        qvina_cmd = (
            'cd ' + software +
            ' && ./qvina-w' +
            ' --receptor ' + protein_file_pdbqt +
            ' --ligand ' + pdbqt_file +
            ' --out ' + pdbqt_file.replace('pdbqt_files', 'docked') + 
            ' --center_x ' + str(pocket_definition["center"][0]) +
            ' --center_y ' + str(pocket_definition["center"][1]) +
            ' --center_z ' + str(pocket_definition["center"][2]) +
            ' --size_x ' + str(pocket_definition["size"][0]) +
            ' --size_y ' + str(pocket_definition["size"][1]*2) +
            ' --size_z ' + str(pocket_definition["size"][2]*2) +
            ' --exhaustiveness ' + str(exhaustiveness) +
            ' --cpu 1' +
            ' --num_modes ' + str(n_poses)
        )
        try:
            subprocess.call(qvina_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
        except Exception as e:
            printlog('QVINAW docking failed: '+e)
            
    qvinaw_docking_results = os.path.join(qvinaw_folder, os.path.basename(split_file).split('.')[0]+'_qvinaw.sdf')
    
    results_pdbqt_files = glob.glob(results_path + '/*.pdbqt')

    for pdbqt_file in results_pdbqt_files:
        # Convert to sdf
        sdf_file = pdbqt_file.replace('.pdbqt', '.sdf')
        obabel_command = f'obabel {pdbqt_file} -O {sdf_file}'
        try:
            subprocess.call(obabel_command, shell=True, stdout=DEVNULL, stderr=STDOUT)
        except Exception as e:
            print(f'Conversion from PDBQT to SDF failed: {e}')
    try:
        sdf_files = glob.glob(results_path + '/*.sdf')
        qvinaw_poses = []
        for sdf in sdf_files:
            df = PandasTools.LoadSDF(sdf, idName='ID', molColName='Molecule', includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=False)
            list_ = [*range(1, int(n_poses)+1, 1)]
            ser = list_ * (len(df) // len(list_))
            df['Pose ID'] = [f"{sdf.replace(results_path+'/', '').replace('.sdf', '')}_QVINAW_{num}" for num, (_, row) in zip(ser + list_[:len(df)-len(ser)], df.iterrows())]
            df = df.rename(columns={'REMARK': 'QVINAW_Affinity'})[['Molecule', 'QVINAW_Affinity', 'Pose ID']]
            df['QVINAW_Affinity'] = df['QVINAW_Affinity'].str.split().str[2]
            qvinaw_poses.append(df)
        qvinaw_poses = pd.concat(qvinaw_poses)
        PandasTools.WriteSDF(qvinaw_poses, qvinaw_docking_results, molColName='Molecule', idName='Pose ID', properties=list(qvinaw_poses.columns))
    except Exception as e:
        printlog('ERROR: Failed to combine QVINAW SDF file!')
        printlog(e)
    else:
        shutil.rmtree(os.path.join(qvinaw_folder, os.path.basename(split_file).split('.')[0]), ignore_errors=True)
    return qvinaw_docking_results

def qvina2_docking_splitted(split_file, protein_file_pdbqt, pocket_definition, software, exhaustiveness, n_poses):
    w_dir = os.path.dirname(protein_file_pdbqt)
    qvina2_folder = os.path.join(w_dir, 'temp', 'qvina2')
    pdbqt_files_folder = os.path.join(qvina2_folder, os.path.basename(split_file).split('.')[0], 'pdbqt_files')
    try:
        os.makedirs(pdbqt_files_folder, mode=0o777, exist_ok=True)
    except Exception as e:
        print(f"Failed to create directories: {e}")

    try:
        meeko_to_pdbqt(split_file, pdbqt_files_folder)
    except Exception as e:
        print('Failed to convert sdf file to .pdbqt')
        print(e)
        
    results_path = os.path.join(qvina2_folder, os.path.basename(split_file).split('.')[0], 'docked')
    try:
        os.makedirs(results_path, mode=0o777, exist_ok=True)
    except Exception as e:
        print(f"Failed to create directories: {e}")
        
    pdbqt_files = glob.glob(os.path.join(pdbqt_files_folder, '*.pdbqt'))

    for pdbqt_file in pdbqt_files:
        qvina_cmd = (
            'cd ' + software +
            ' && ./qvina-w' +
            ' --receptor ' + protein_file_pdbqt +
            ' --ligand ' + pdbqt_file +
            ' --out ' + pdbqt_file.replace('pdbqt_files', 'docked') + 
            ' --center_x ' + str(pocket_definition["center"][0]) +
            ' --center_y ' + str(pocket_definition["center"][1]) +
            ' --center_z ' + str(pocket_definition["center"][2]) +
            ' --size_x ' + str(pocket_definition["size"][0]*2) +
            ' --size_y ' + str(pocket_definition["size"][1]*2) +
            ' --size_z ' + str(pocket_definition["size"][2]*2) +
            ' --exhaustiveness ' + str(exhaustiveness) +
            ' --cpu 1' +
            ' --num_modes ' + str(n_poses)
        )
        try:
            subprocess.call(qvina_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
        except Exception as e:
            printlog('QVINA2 docking failed: '+e)
            
    qvina2_docking_results = os.path.join(qvina2_folder, os.path.basename(split_file).split('.')[0]+'_qvina2.sdf')
    
    results_pdbqt_files = glob.glob(results_path + '/*.pdbqt')

    for pdbqt_file in results_pdbqt_files:
        # Convert to sdf
        sdf_file = pdbqt_file.replace('.pdbqt', '.sdf')
        obabel_command = f'obabel {pdbqt_file} -O {sdf_file}'
        try:
            subprocess.call(obabel_command, shell=True, stdout=DEVNULL, stderr=STDOUT)
        except Exception as e:
            print(f'Conversion from PDBQT to SDF failed: {e}')
    try:
        sdf_files = glob.glob(results_path + '/*.sdf')
        qvina2_poses = []
        for sdf in sdf_files:
            df = PandasTools.LoadSDF(sdf, idName='ID', molColName='Molecule', includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=False)
            list_ = [*range(1, int(n_poses)+1, 1)]
            ser = list_ * (len(df) // len(list_))
            df['Pose ID'] = [f"{sdf.replace(results_path+'/', '').replace('.sdf', '')}_QVINA2_{num}" for num, (_, row) in zip(ser + list_[:len(df)-len(ser)], df.iterrows())]
            df = df.rename(columns={'REMARK': 'QVINA2_Affinity'})[['Molecule', 'QVINA2_Affinity', 'Pose ID']]
            df['QVINA2_Affinity'] = df['QVINA2_Affinity'].str.split().str[2]
            qvina2_poses.append(df)
        qvina2_poses = pd.concat(qvina2_poses)
        PandasTools.WriteSDF(qvina2_poses, qvina2_docking_results, molColName='Molecule', idName='Pose ID', properties=list(qvina2_poses.columns))
    except Exception as e:
        printlog('ERROR: Failed to combine QVINA2 SDF file!')
        printlog(e)
    else:
        shutil.rmtree(os.path.join(qvina2_folder, os.path.basename(split_file).split('.')[0]), ignore_errors=True)
    return qvina2_docking_results
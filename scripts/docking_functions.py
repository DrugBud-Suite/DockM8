import os
import subprocess
import shutil
import pandas as pd
from rdkit.Chem import PandasTools
from IPython.display import display

def smina_docking(protein_file, ref_file, software, exhaustiveness, n_poses):
    w_dir = os.path.dirname(protein_file)
    library = w_dir+'/temp/final_library.sdf'
    smina_folder = w_dir+'/temp/smina/'
    try:
        os.mkdir(smina_folder, mode = 0o777)
    except:
        print('Smina folder already exists')
    results = smina_folder+'docked.sdf'
    log = smina_folder+'log.txt'
    smina_cmd = 'cd '+software+' && ./gnina -r '+protein_file+' -l '+library+' --autobox_ligand '+ref_file+' -o '+results+' --exhaustiveness ' +str(exhaustiveness)+' --num_modes '+str(n_poses)+' --cnn_scoring none'+' --log '+log
    subprocess.call(smina_cmd, shell=True)
    smina_poses = PandasTools.LoadSDF(results, idName='ID', molColName='Molecule',includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True)
    return results

def gnina_docking(protein_file, ref_file, software, exhaustiveness, n_poses):
    w_dir = os.path.dirname(protein_file)
    library = w_dir+'/temp/final_library.sdf'
    gnina_folder = w_dir+'/temp/gnina/'
    try:
        os.mkdir(gnina_folder, mode = 0o777)
    except:
        print('Gnina folder already exists')
    results = gnina_folder+'/docked.sdf'
    log = gnina_folder+'log.txt'
    gnina_cmd = 'cd '+software+' && ./gnina -r '+protein_file+' -l '+library+' --autobox_ligand '+ref_file+' -o '+results+' --exhaustiveness ' +str(exhaustiveness)+' --num_modes '+str(n_poses)+' --cnn_scoring rescore --cnn crossdock_default2018 '+' --log '+log
    subprocess.call(gnina_cmd, shell=True)
    gnina_poses = PandasTools.LoadSDF(results, idName='ID', molColName='Molecule',includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True)
    return results

def plants_docking(protein_file, ref_file, software):
    w_dir = os.path.dirname(protein_file)
    # Define initial variables
    plants_search_speed = "speed1"
    ants = "20"
    plants_docking_scoring = "chemplp"
    plants_docking_dir = w_dir+"/temp/plants"
    plants_docking_results_dir = w_dir+"/temp/plants/results"
    #Create plants docking folder
    if os.path.isdir(plants_docking_dir) == True:
        print('Plants docking folder already exists')
    else:
        os.mkdir(plants_docking_dir)
    #Convert protein file to .mol2 using open babel
    plants_protein_mol2 = w_dir+"/temp/plants/protein.mol2"
    try:
        obabel_command = 'obabel -ipdb '+protein_file+' -O '+plants_protein_mol2
        os.system(obabel_command)
    except:
        print('ERROR: Failed to convert protein file to .mol2!')
    #Convert protein file to .mol2 using open babel
    plants_ref_mol2 = w_dir+"/temp/plants/ref.mol2"
    if ref_file.endswith(".mol2"):
        shutil.copy(ref_file, plants_docking_dir)
        os.rename(plants_docking_dir+"/"+os.path.basename(ref_file), plants_ref_mol2)
    if ref_file.endswith(".sdf"):
        try:
            obabel_command = 'obabel -isdf '+ref_file+' -O '+plants_ref_mol2
            os.system(obabel_command)
        except:
            print('ERROR: Failed to convert reference ligand file to .mol2!')
    if ref_file.endswith(".pdb"):
        try:
            obabel_command = 'obabel -ipdb '+ref_file+' -O '+plants_ref_mol2
            os.system(obabel_command)
        except:
            print('ERROR: Failed to convert reference ligand file to .mol2!')
    else:
        print('ERROR: Reference ligand file not in a readable format!')
    #Convert prepared ligand file to .mol2 using open babel
    final_library = w_dir+"/temp/final_library.sdf"
    plants_library_mol2 = w_dir+"/temp/plants/ligands.mol2"
    try:
        obabel_command = 'obabel -isdf '+final_library+' -O '+plants_library_mol2
        os.system(obabel_command)
    except:
        print('ERROR: Failed to convert library file to .mol2!')
    #Determine binding site coordinates
    plants_binding_site_command = "cd "+software+" && ./PLANTS --mode bind "+plants_ref_mol2+" 6"
    run_plants_binding_site = os.popen(plants_binding_site_command)
    output_plants_binding_site = run_plants_binding_site.readlines()
    keep = []
    for l in output_plants_binding_site:
        if l.startswith("binding"):
            keep.append(l)
        else:
            pass
    binding_site_center = keep[0].split()
    binding_site_radius = keep[1].split()
    binding_site_radius = binding_site_radius[1]
    binding_site_x = binding_site_center[1]
    binding_site_y = binding_site_center[2]
    binding_site_z = binding_site_center[3].replace('+', '')
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
    'cluster_structures 10\n',
    'cluster_rmsd 2.0\n',

    '# write\n',
    'write_ranking_links 0\n',
    'write_protein_bindingsite 0\n',
    'write_protein_conformations 0\n',
    'write_protein_splitted 0\n',
    'write_merged_protein 0\n',
    '####\n']
    #Write config file
    plants_docking_config_path_config = plants_docking_config_path_txt.replace(".txt", ".config")
    with open(plants_docking_config_path_config, 'w') as configwriter:
        configwriter.writelines(plants_config)
    configwriter.close()
    # os.rename(plants_docking_config_path_txt, plants_docking_config_path_config)
    #Run PLANTS docking
    plants_docking_command = "cd "+software+" && ./PLANTS --mode screen "+plants_docking_config_path_config
    os.system(plants_docking_command)

    plants_docking_results_mol2 = w_dir+"/temp/plants/results/docked_ligands.mol2"
    plants_docking_results_sdf = plants_docking_results_mol2.replace(".mol2", ".sdf")
    # Convert PLANTS poses to sdf
    try:
        obabel_command = 'obabel -imol2 '+plants_docking_results_mol2+' -O '+plants_docking_results_sdf 
        os.system(obabel_command)
    except:
        print('ERROR: Failed to convert PLANTS poses file to .sdf!')
    return plants_docking_results_sdf

def fetch_poses(protein_file, n_poses):
    w_dir = os.path.dirname(protein_file)
    smina_docking_results = w_dir+"/temp/smina/docked.sdf"
    gnina_docking_results = w_dir+"/temp/gnina/docked.sdf"
    plants_poses_results_sdf = w_dir+"/temp/plants/results/docked_ligands.sdf"
    plants_scoring_results_sdf = w_dir+"/temp/plants/results/ranking.csv"
    #Fetch PLANTS poses
    try:
        plants_poses = PandasTools.LoadSDF(plants_poses_results_sdf, idName='ID', molColName='Molecule',includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True)
        plants_scores = pd.read_csv(plants_scoring_results_sdf)
        plants_scores = plants_scores.rename(columns={'LIGAND_ENTRY':'ID', 'TOTAL_SCORE':'CHEMPLP'})
        plants_scores = plants_scores[['ID', 'CHEMPLP']]
        plants_df = pd.merge(plants_scores, plants_poses, on='ID')
        for i, row in plants_df.iterrows():
            split = row['ID'].split("_")
            conformer_id = str(split[4])
            plants_df.loc[i, ['Pose ID']] = split[0]+"_PLANTS_"+conformer_id
            plants_df.loc[i, ['ID']] = split[0]
    except:
        print('ERROR: Failed to Load PLANTS poses SDF file!')
    #Fetch SMINA poses
    try:
        smina_df = PandasTools.LoadSDF(smina_docking_results, idName='ID', molColName='Molecule',includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True)
        #smina_df = smina_df[['ID', 'Molecule']]
        list_ = [*range(1, int(n_poses)+1, 1)]
        ser = list_ * int(len(smina_df)/len(list_))
        smina_df['number'] = ser + list_[:len(smina_df)-len(ser)]
        for i, row in smina_df.iterrows():
            smina_df.loc[i, ['Pose ID']] = row['ID']+"_SMINA_"+str(row['number'])
        smina_df.drop('number', axis=1, inplace=True)
        smina_df = smina_df.rename(columns={'minimizedAffinity':'SMINA_Affinity'})
    except:
        print('ERROR: Failed to Load SMINA poses SDF file!')
    #Fetch GNINA poses
    try:
        gnina_df = PandasTools.LoadSDF(gnina_docking_results, idName='ID', molColName='Molecule',includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True)
        #gnina_df = gnina_df[['ID', 'Molecule']]
        list_ = [*range(1, int(n_poses)+1, 1)]
        ser = list_ * int(len(gnina_df)/len(list_))
        gnina_df['number'] = ser + list_[:len(gnina_df)-len(ser)]
        for i, row in gnina_df.iterrows():
            gnina_df.loc[i, ['Pose ID']] = row['ID']+"_GNINA_"+str(row['number'])
        gnina_df.drop('number', axis=1, inplace=True)
        gnina_df = gnina_df.rename(columns={'minimizedAffinity':'GNINA_Affinity'})
    except:
        print('ERROR: Failed to Load GNINA poses SDF file!')

    all_poses = pd.concat([plants_df, smina_df, gnina_df]) 
    PandasTools.WriteSDF(all_poses, w_dir+"/temp/allposes.sdf", molColName='Molecule', idName='Pose ID', properties=list(all_poses.columns))
    return all_poses

def docking(protein_file, ref_file, software, exhaustiveness, n_poses):
    smina_docking_results_sdf = smina_docking(protein_file, ref_file, software, exhaustiveness, n_poses)
    gnina_docking_results_sdf = gnina_docking(protein_file, ref_file, software, exhaustiveness, n_poses)
    plants_docking_results_sdf = plants_docking(protein_file, ref_file, software)
    all_poses = fetch_poses(protein_file, n_poses)
    return all_poses
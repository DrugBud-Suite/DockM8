import os
import shutil
import subprocess
from subprocess import DEVNULL, STDOUT, PIPE
import pandas as pd
import functools
from rdkit import Chem
from rdkit.Chem import PandasTools
import oddt
from oddt.scoring.functions import NNScore
from oddt.scoring.functions import RFScore
from oddt.scoring.functions.PLECscore import PLECscore
from tqdm import tqdm
import multiprocessing
import time
from scripts.utilities import *
from tqdm import tqdm

def rescore_all(w_dir, protein_file, ref_file, software, clustered_sdf):
    tic = time.perf_counter()
    rescoring_folder_name = os.path.basename(clustered_sdf).split('/')[-1]
    rescoring_folder_name = rescoring_folder_name.replace('.sdf', '')
    rescoring_folder = w_dir+'/temp/rescoring_'+rescoring_folder_name
    if os.path.isdir(rescoring_folder) == True:
        print('Deleting existing rescoring folder')
        shutil.rmtree(rescoring_folder)
    else:
        os.mkdir(rescoring_folder)
    def gnina_rescoring(sdf):
        tic = time.perf_counter()
        print('Rescoring with GNINA')
        create_temp_folder(rescoring_folder+'/gnina_rescoring/')
        cnn = 'crossdock_default2018'
        results = rescoring_folder+'/gnina_rescoring/'+'rescored_'+cnn+'.sdf'
        log_file = rescoring_folder+'/gnina_rescoring/'+'log_'+cnn+'.txt'
        gnina_cmd = 'cd '+software+' && ./gnina -r '+protein_file+' -l '+sdf+' --autobox_ligand '+ref_file+' -o '+results+' --cnn '+cnn+' --log '+log_file+' --score_only'
        subprocess.call(gnina_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
        gnina_rescoring_results = PandasTools.LoadSDF(results, idName='Pose ID', molColName=None, includeFingerprints=False, removeHs=False)
        gnina_rescoring_results.rename(columns = {'minimizedAffinity':'GNINA_Affinity', 'CNNscore':'GNINA_CNN_Score', 'CNNaffinity':'GNINA_CNN_Affinity'}, inplace = True)
        gnina_rescoring_results = gnina_rescoring_results[['Pose ID', 'GNINA_Affinity', 'GNINA_CNN_Score', 'GNINA_CNN_Affinity']]
        toc = time.perf_counter()
        print(f'Rescoring with GNINA complete in {toc-tic:0.4f}!')
        return gnina_rescoring_results
    def vinardo_rescoring(sdf):
        tic = time.perf_counter()
        print('Rescoring with Vinardo')
        create_temp_folder(rescoring_folder+'/vinardo_rescoring/')
        results = rescoring_folder+'/vinardo_rescoring/'+'rescored_vinardo.sdf'
        log_file = rescoring_folder+'/vinardo_rescoring/'+'log.txt'
        vinardo_cmd = 'cd '+software+' && ./gnina -r '+protein_file+' -l '+sdf+' --autobox_ligand '+ref_file+' -o '+results+' --log '+log_file+' --score_only --scoring vinardo --cnn_scoring none'
        subprocess.call(vinardo_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
        vinardo_rescoring_results = PandasTools.LoadSDF(results, idName='Pose ID', molColName=None, includeFingerprints=False, removeHs=False)
        vinardo_rescoring_results.rename(columns = {'minimizedAffinity':'Vinardo_Affinity'}, inplace = True)
        vinardo_rescoring_results = vinardo_rescoring_results[['Pose ID', 'Vinardo_Affinity']]
        toc = time.perf_counter()
        print(f'Rescoring with Vinardo complete in {toc-tic:0.4f}!')
        return vinardo_rescoring_results
    def AD4_rescoring(sdf):
        tic = time.perf_counter()
        print('Rescoring with AD4')
        create_temp_folder(rescoring_folder+'/AD4_rescoring/')
        results = rescoring_folder+'/AD4_rescoring/'+'rescored_AD4.sdf'
        log_file = rescoring_folder+'/AD4_rescoring/'+'log.txt'
        AD4_cmd = 'cd '+software+' && ./gnina -r '+protein_file+' -l '+sdf+' --autobox_ligand '+ref_file+' -o '+results+' --log '+log_file+' --score_only --scoring ad4_scoring --cnn_scoring none'
        subprocess.call(AD4_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
        AD4_rescoring_results = PandasTools.LoadSDF(results, idName='Pose ID', molColName='None', includeFingerprints=False, removeHs=False)
        AD4_rescoring_results.rename(columns = {'minimizedAffinity':'AD4_Affinity'}, inplace = True)
        AD4_rescoring_results = AD4_rescoring_results[['Pose ID', 'AD4_Affinity']]
        toc = time.perf_counter()
        print(f'Rescoring with AD4 complete in {toc-tic:0.4f}!')
        return AD4_rescoring_results
    def oddt_rfscoreV1_rescoring(sdf):
        tic = time.perf_counter()
        print('Rescoring with RFScoreV1')
        rescorers = {'rfscore':RFScore.rfscore()}
        create_temp_folder(rescoring_folder+'/rfscoreV1_rescoring/')
        scorer = rescorers['rfscore']
        pickle = software+'/RFScore_v1_pdbbind2016.pickle'
        scorer = scorer.load(pickle)
        oddt_prot=next(oddt.toolkit.readfile('pdb', protein_file))
        oddt_prot.protein = True
        scorer.set_protein(oddt_prot)
        re_scores = []
        df = PandasTools.LoadSDF(sdf, idName='Pose ID', molColName='Molecule', removeHs=False)
        for mol in tqdm(df['Molecule']):
            oddt_mol = oddt.toolkit.Molecule(mol)
            scored_mol = scorer.predict_ligand(oddt_mol)
            re_scores.append(float(scored_mol.data['rfscore_v1']))
        df['RFScoreV1']=re_scores
        df = df[['Pose ID', 'RFScoreV1']]
        toc = time.perf_counter()
        print(f'Rescoring with RFScoreV1 complete in {toc-tic:0.4f}!')
        return df
    def oddt_rfscoreV1_rescoring_multiprocessing(sdf):
        tic = time.perf_counter()
        print('Rescoring with RFScoreV1')
        rescorers = {'rfscore':RFScore.rfscore()}
        create_temp_folder(rescoring_folder+'/rfscoreV1_rescoring/')
        scorer = rescorers['rfscore']
        pickle = software+'/RFScore_v1_pdbbind2016.pickle'
        scorer = scorer.load(pickle)
        oddt_prot=next(oddt.toolkit.readfile('pdb', protein_file))
        oddt_prot.protein = True
        scorer.set_protein(oddt_prot)
        re_scores = []
        df = PandasTools.LoadSDF(sdf, idName='Pose ID', molColName='Molecule', removeHs=False)
        molecules = df['Molecule']
        global score_mol
        def score_mol(mol):
            oddt_mol = oddt.toolkit.Molecule(mol)
            scored_mol = scorer.predict_ligand(oddt_mol)
            return float(scored_mol.data['rfscore_v1'])
        with multiprocessing.Pool() as p:
            # Score the molecules in parallel
            re_scores = p.map(score_mol, molecules)
        df['RFScoreV1']=re_scores
        df = df[['Pose ID', 'RFScoreV1']]
        toc = time.perf_counter()
        print(f'Rescoring with RFScoreV1 multiprocessing complete in {toc-tic:0.4f}!')
        return df
    def oddt_rfscoreV2_rescoring(sdf):
        tic = time.perf_counter()
        print('Rescoring with RFScoreV2')
        rescorers = {'rfscore':RFScore.rfscore()}
        create_temp_folder(rescoring_folder+'/rfscoreV2_rescoring/')
        scorer = rescorers['rfscore']
        pickle = software+'/RFScore_v2_pdbbind2016.pickle'
        scorer = scorer.load(pickle)
        oddt_prot=next(oddt.toolkit.readfile('pdb', protein_file))
        oddt_prot.protein = True
        scorer.set_protein(oddt_prot)
        re_scores = []
        df = PandasTools.LoadSDF(sdf, idName='Pose ID', molColName='Molecule', removeHs=False)
        for mol in tqdm(df['Molecule']):
            oddt_mol = oddt.toolkit.Molecule(mol)
            scored_mol = scorer.predict_ligand(oddt_mol)
            re_scores.append(float(scored_mol.data['rfscore_v2']))
        df['RFScoreV2']=re_scores
        df = df[['Pose ID', 'RFScoreV2']]
        toc = time.perf_counter()
        print(f'Rescoring with RFScoreV2 complete in {toc-tic:0.4f}!')
        return df
    def oddt_rfscoreV3_rescoring(sdf):
        tic = time.perf_counter()
        print('Rescoring with RFScoreV3')
        rescorers = {'rfscore':RFScore.rfscore()}
        create_temp_folder(rescoring_folder+'/rfscoreV3_rescoring/')
        scorer = rescorers['rfscore']
        pickle = software+'/RFScore_v3_pdbbind2016.pickle'
        scorer = scorer.load(pickle)
        oddt_prot=next(oddt.toolkit.readfile('pdb', protein_file))
        oddt_prot.protein = True
        scorer.set_protein(oddt_prot)
        re_scores = []
        df = PandasTools.LoadSDF(sdf, idName='Pose ID', molColName='Molecule', removeHs=False)
        for mol in tqdm(df['Molecule']):
            oddt_mol = oddt.toolkit.Molecule(mol)
            scored_mol = scorer.predict_ligand(oddt_mol)
            re_scores.append(float(scored_mol.data['rfscore_v3']))
        df['RFScoreV3']=re_scores
        df = df[['Pose ID', 'RFScoreV3']]
        toc = time.perf_counter()
        print(f'Rescoring with RFScoreV3 complete in {toc-tic:0.4f}!')
        return df
    def plp_rescoring(sdf):
        tic = time.perf_counter()
        print('Rescoring with PLP')
        plants_search_speed = 'speed1'
        ants = '20'
        plp_rescoring_folder = rescoring_folder+'/plp_rescoring/'
        create_temp_folder(plp_rescoring_folder)
        #Read protein and ref files generated during PLANTS docking
        plants_protein_mol2 = w_dir+'/temp/plants/protein.mol2'
        plants_ref_mol2 = w_dir+'/temp/plants/ref.mol2'
        #Convert clustered ligand file to .mol2 using open babel
        plants_ligands_mol2 = plp_rescoring_folder+'/ligands.mol2'
        try:
            obabel_command = 'obabel -isdf '+sdf+' -O '+plants_ligands_mol2
            os.system(obabel_command)
        except:
            print('ERROR: Failed to convert clustered library file to .mol2!')
        #Determine binding site coordinates
        plants_binding_site_command = 'cd '+software+' && ./PLANTS --mode bind '+plants_ref_mol2+' 6'
        run_plants_binding_site = subprocess.Popen(plants_binding_site_command, shell = True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        output, err = run_plants_binding_site.communicate()
        output_plants_binding_site = output.decode('utf-8').splitlines()
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
        binding_site_z = str(binding_site_center[3]).replace('+', '')
        results_csv_location = plp_rescoring_folder+'results/ranking.csv'
        #Generate plants config file
        plp_rescoring_config_path_txt = plp_rescoring_folder+'config.txt'
        plp_config = ['# search algorithm\n',
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
        'scoring_function plp\n',
        'outside_binding_site_penalty 50.0\n',
        'enable_sulphur_acceptors 1\n',
        '# Intramolecular ligand scoring\n',
        'ligand_intra_score clash2\n',
        'chemplp_clash_include_14 1\n',
        'chemplp_clash_include_HH 0\n',

        '# input\n',
        'protein_file '+plants_protein_mol2+'\n',
        'ligand_file '+plants_ligands_mol2+'\n',

        '# output\n',
        'output_dir '+plp_rescoring_folder+'results\n',

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
        'write_protein_bindingsite 1\n',
        'write_protein_conformations 1\n',
        'write_protein_splitted 1\n',
        'write_merged_protein 0\n',
        '####\n']
        #Write config file
        plp_rescoring_config_path_config = plp_rescoring_config_path_txt.replace('.txt', '.config')
        with open(plp_rescoring_config_path_config, 'w') as configwriter:
            configwriter.writelines(plp_config)
        configwriter.close()
        #Run PLANTS docking
        plp_rescoring_command = 'cd '+software+' && ./PLANTS --mode rescore '+plp_rescoring_config_path_config
        subprocess.call(plp_rescoring_command, shell=True, stdout=DEVNULL, stderr=STDOUT)
        #Fetch results
        results_csv_location = plp_rescoring_folder+'results/ranking.csv'
        plp_results = pd.read_csv(results_csv_location, sep=',', header=0)
        plp_results.rename(columns = {'TOTAL_SCORE':'PLP'}, inplace = True)
        for i, row in plp_results.iterrows():
            split = row['LIGAND_ENTRY'].split('_')
            plp_results.loc[i, ['Pose ID']] = split[0]+'_'+split[1]+'_'+split[2]
        plp_rescoring_output = plp_results[['Pose ID', 'PLP']]
        toc = time.perf_counter()
        print(f'Rescoring with PLP complete in {toc-tic:0.4f}!')
        return plp_rescoring_output
    def chemplp_rescoring(sdf):
        tic = time.perf_counter()
        print('Rescoring with CHEMPLP')
        plants_search_speed = 'speed1'
        ants = '20'
        chemplp_rescoring_folder = rescoring_folder+'/chemplp_rescoring/'
        create_temp_folder(chemplp_rescoring_folder)
        #Read protein and ref files generated during PLANTS docking
        plants_protein_mol2 = w_dir+'/temp/plants/protein.mol2'
        plants_ref_mol2 = w_dir+'/temp/plants/ref.mol2'
        #Convert clustered ligand file to .mol2 using open babel
        plants_ligands_mol2 = chemplp_rescoring_folder+'/ligands.mol2'
        try:
            obabel_command = 'obabel -isdf '+sdf+' -O '+plants_ligands_mol2
            os.system(obabel_command)
        except:
            print('ERROR: Failed to convert clustered library file to .mol2!')
        #Determine binding site coordinates
        plants_binding_site_command = 'cd '+software+' && ./PLANTS --mode bind '+plants_ref_mol2+' 6'
        run_plants_binding_site = subprocess.Popen(plants_binding_site_command, shell = True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        output, err = run_plants_binding_site.communicate()
        output_plants_binding_site = output.decode('utf-8').splitlines()
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
        #Generate plants config file
        chemplp_rescoring_config_path_txt = chemplp_rescoring_folder+'config.txt'
        chemplp_config = ['# search algorithm\n',
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
        'scoring_function chemplp\n',
        'outside_binding_site_penalty 50.0\n',
        'enable_sulphur_acceptors 1\n',
        '# Intramolecular ligand scoring\n',
        'ligand_intra_score clash2\n',
        'chemplp_clash_include_14 1\n',
        'chemplp_clash_include_HH 0\n',

        '# input\n',
        'protein_file '+plants_protein_mol2+'\n',
        'ligand_file '+plants_ligands_mol2+'\n',

        '# output\n',
        'output_dir '+chemplp_rescoring_folder+'results\n',

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
        'write_protein_bindingsite 1\n',
        'write_protein_conformations 1\n',
        'write_protein_splitted 1\n',
        'write_merged_protein 0\n',
        '####\n']
        #Write config file
        chemplp_rescoring_config_path_config = chemplp_rescoring_config_path_txt.replace('.txt', '.config')
        with open(chemplp_rescoring_config_path_config, 'w') as configwriter:
            configwriter.writelines(chemplp_config)
        configwriter.close()
        #Run PLANTS docking
        chemplp_rescoring_command = 'cd '+software+' && ./PLANTS --mode rescore '+chemplp_rescoring_config_path_config
        subprocess.call(chemplp_rescoring_command, shell=True, stdout=DEVNULL, stderr=STDOUT)
        #Fetch results
        results_csv_location = chemplp_rescoring_folder+'results/ranking.csv'
        chemplp_results = pd.read_csv(results_csv_location, sep=',', header=0)
        chemplp_results.rename(columns = {'TOTAL_SCORE':'CHEMPLP'}, inplace = True)
        for i, row in chemplp_results.iterrows():
            split = row['LIGAND_ENTRY'].split('_')
            chemplp_results.loc[i, ['Pose ID']] = split[0]+'_'+split[1]+'_'+split[2]
        chemplp_rescoring_output = chemplp_results[['Pose ID', 'CHEMPLP']]
        toc = time.perf_counter()
        print(f'Rescoring with CHEMPLP complete in {toc-tic:0.4f}!')
        return chemplp_rescoring_output
    def oddt_nnscore_rescoring(sdf):
        tic = time.perf_counter()
        print('Rescoring with NNScore')
        rescorers = {'nnscore':NNScore.nnscore()}
        nnscore_rescoring_folder = rescoring_folder+'/nnscore_rescoring/'
        create_temp_folder(nnscore_rescoring_folder)
        scorer = rescorers['nnscore']
        pickle = software+'/NNScore_pdbbind2016.pickle'
        scorer = scorer.load(pickle)
        oddt_prot=next(oddt.toolkit.readfile('pdb', protein_file))
        oddt_prot.protein = True
        scorer.set_protein(oddt_prot)
        re_scores = []
        df = PandasTools.LoadSDF(sdf, idName='Pose ID', molColName='Molecule', removeHs=False)
        for mol in tqdm(df['Molecule']):
            Chem.MolToMolFile(mol, nnscore_rescoring_folder+'/temp.sdf')
            oddt_lig = next(oddt.toolkit.readfile('sdf', nnscore_rescoring_folder+'/temp.sdf'))
            scored_mol = scorer.predict_ligand(oddt_lig)
            re_scores.append(float(scored_mol.data['nnscore']))
        df['NNScore']=re_scores
        df = df[['Pose ID', 'NNScore']]
        toc = time.perf_counter()
        print(f'Rescoring with NNScore complete in {toc-tic:0.4f}!')
        return df
    def oddt_nnscore_rescoring_multiprocessing(sdf):
        tic = time.perf_counter()
        print('Rescoring with NNScore')
        rescorers = {'nnscore':NNScore.nnscore()}
        create_temp_folder(rescoring_folder+'/nnscore_rescoring/')
        scorer = rescorers['nnscore']
        pickle = software+'/NNScore_pdbbind2016.pickle'
        scorer = scorer.load(pickle)
        oddt_prot=next(oddt.toolkit.readfile('pdb', protein_file))
        oddt_prot.protein = True
        scorer.set_protein(oddt_prot)
        re_scores = []
        df = PandasTools.LoadSDF(sdf, idName='Pose ID', molColName='Molecule', removeHs=False)
        global score_mol
        def score_mol(mol):
            oddt_mol = oddt.toolkit.Molecule(mol)
            scored_mol = scorer.predict_ligand(oddt_mol)
            return float(scored_mol.data['nnscore'])
        with multiprocessing.Pool() as p:
            re_scores = tqdm(p.imap(score_mol, df['Molecule']), total=len(df['Molecule']))
        df['NNScore']=re_scores
        df = df[['Pose ID', 'NNScore']]
        toc = time.perf_counter()
        print(f'Rescoring with NNScore multiprocessing complete in {toc-tic:0.4f}!')
        return df
    def oddt_plecscore_rescoring(sdf):
        tic = time.perf_counter()
        print('Rescoring with PLECscore')
        rescorers = {'PLECnn_p5_l1_s65536':PLECscore(version='nn')}
        plecscore_rescoring_folder = rescoring_folder+'/plecscore_rescoring/'
        create_temp_folder(plecscore_rescoring_folder)
        scorer = rescorers['PLECnn_p5_l1_s65536']
        pickle = software+'/PLECnn_p5_l1_2016.pickle'
        scorer = scorer.load(pickle)
        oddt_prot=next(oddt.toolkit.readfile('pdb', protein_file.replace('.pdb','_pocket.pdb')))
        oddt_prot.protein = True
        scorer.set_protein(oddt_prot)
        re_scores = []
        df = PandasTools.LoadSDF(sdf, idName='Pose ID', molColName='Molecule', removeHs=False)
        for mol in tqdm(df['Molecule']):
            Chem.MolToMolFile(mol, plecscore_rescoring_folder+'/temp.sdf')
            oddt_lig = next(oddt.toolkit.readfile('sdf', plecscore_rescoring_folder+'/temp.sdf'))
            scored_mol = scorer.predict_ligand(oddt_lig)
            re_scores.append(float(scored_mol.data['PLECnn_p5_l1_s65536']))
        df['PLECnn']=re_scores
        df = df[['Pose ID', 'PLECnn']]
        toc = time.perf_counter()
        print(f'Rescoring with PLECScore complete in {toc-tic:0.4f}!')
        return df
    def oddt_plecscore_rescoring_multiprocessing(sdf):
        tic = time.perf_counter()
        print('Rescoring with PLECscore')
        rescorers = {'PLECnn_p5_l1_s65536':PLECscore(version='nn')}
        create_temp_folder(rescoring_folder+'/plecscore_rescoring/')
        scorer = rescorers['PLECnn_p5_l1_s65536']
        pickle = software+'/PLECnn_p5_l1_2016.pickle'
        scorer = scorer.load(pickle)
        oddt_prot=next(oddt.toolkit.readfile('pdb', protein_file.replace('.pdb','_pocket.pdb')))
        oddt_prot.protein = True
        scorer.set_protein(oddt_prot)
        re_scores = []
        df = PandasTools.LoadSDF(sdf, idName='Pose ID', molColName='Molecule', removeHs=False)
        global score_mol
        def score_mol(mol):
            oddt_mol = oddt.toolkit.Molecule(mol)
            scored_mol = scorer.predict_ligand(oddt_mol)
            return float(scored_mol.data['PLECnn_p5_l1_s65536'])
        with multiprocessing.Pool() as p:
            re_scores = tqdm(p.imap(score_mol, df['Molecule']), total=len(df['Molecule']))
        df['PLECnn']=re_scores
        df = df[['Pose ID', 'PLECnn']]
        toc = time.perf_counter()
        print(f'Rescoring with PLECScore multiprocessing complete in {toc-tic:0.4f}!')
        return df
    def SCORCH_rescoring(clustered_sdf):
        tic = time.perf_counter()
        SCORCH_rescoring_folder = rescoring_folder+'/SCORCH_rescoring/'
        create_temp_folder(SCORCH_rescoring_folder)
        #Convert protein file to .mol2 using open babel
        SCORCH_protein = SCORCH_rescoring_folder+"protein.pdbqt"
        try:
            obabel_command = 'obabel -ipdb '+protein_file+' -O '+SCORCH_protein+' --partialcharges gasteiger'
            os.system(obabel_command)
        except:
            print('ERROR: Failed to convert protein file to .pdbqt!')
        SCORCH_ligands = SCORCH_rescoring_folder+"ligands.pdbqt"
        try:
            obabel_command = 'obabel -isdf '+clustered_sdf+' -O '+SCORCH_ligands+' --partialcharges gasteiger'
            os.system(obabel_command)
        except:
            print('ERROR: Failed to convert ligands to .pdbqt!')
        try:
            SCORCH_command = 'python '+software+'/SCORCH-main/scorch.py --receptor '+SCORCH_protein+' --ligand '+SCORCH_ligands+' --out '+SCORCH_rescoring_folder+'scoring_results.csv --threads 8 --verbose --return_pose_scores'
            print(SCORCH_command)
            os.system(SCORCH_command)
        except:
            print('ERROR: Failed to run SCORCH!')
        toc = time.perf_counter()
        print(f'Rescoring with SCORCH complete in {toc-tic:0.4f}!')
        return    
    #scorch_df = SCORCH_rescoring(clustered_sdf)
    rescoring_functions = {'GNINA': gnina_rescoring, 'VINARDO': vinardo_rescoring, 'AD4': AD4_rescoring, 
                        'rfscoreV1': oddt_rfscoreV1_rescoring, 'rfscoreV2': oddt_rfscoreV2_rescoring,
                        'rfscoreV3': oddt_rfscoreV3_rescoring, 'plp': plp_rescoring, 'chemplp': chemplp_rescoring,
                        'nn_score': oddt_nnscore_rescoring_multiprocessing, 'plec': oddt_plecscore_rescoring_multiprocessing}
    rescored_dfs = [rescoring_functions[key](clustered_sdf) for key in rescoring_functions.keys()]
    combined_dfs = functools.reduce(lambda  left,right: pd.merge(left,right,on=['Pose ID'],how='inner'), rescored_dfs)
    first_column = combined_dfs.pop('Pose ID')
    combined_dfs.insert(0, 'Pose ID', first_column)
    columns=combined_dfs.columns
    col=columns[1:]
    for c in col.tolist():
        if c == 'Pose ID':
            pass
        if combined_dfs[c].dtypes is not float:
            combined_dfs[c] = combined_dfs[c].apply(pd.to_numeric, errors='coerce')
        else:
            pass
    combined_dfs.to_csv(rescoring_folder+'/allposes_rescored.csv')
    toc = time.perf_counter()
    print(f'Rescoring complete in {toc-tic:0.4f}!')
    return combined_dfs
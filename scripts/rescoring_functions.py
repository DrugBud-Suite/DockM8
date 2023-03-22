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
import concurrent.futures
import time
from scripts.utilities import *
from software.ECIF.ecif import *
# from software.SCORCH.scorch import parse_module_args, scoring
from IPython.display import display
import pickle
from concurrent.futures import ThreadPoolExecutor
from functools import partial

#TODO: add new scoring functions:
# _ECIF
# _SIEVE_Score (no documentation)
# _AEScore
# _RTMScore

def delete_files(folder_path, save_file):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path) and item != save_file:
            os.remove(item_path)
        elif os.path.isdir(item_path):
            delete_files(item_path, save_file)
            if len(os.listdir(item_path)) == 0 and item != save_file:
                os.rmdir(item_path)

def rescore_all(w_dir, protein_file, ref_file, software, clustered_sdf, functions, mp, ncpus):
    tic = time.perf_counter()
    rescoring_folder_name = os.path.basename(clustered_sdf).split('/')[-1]
    rescoring_folder_name = rescoring_folder_name.replace('.sdf', '')
    rescoring_folder = w_dir+'/temp/rescoring_'+rescoring_folder_name
    create_temp_folder(rescoring_folder)
    def gnina_rescoring(sdf, mp):
        tic = time.perf_counter()
        create_temp_folder(rescoring_folder+'/gnina_rescoring/', silent=True)
        cnn = 'crossdock_default2018'
        if mp == 0:
            printlog('Rescoring with GNINA')
            results = rescoring_folder+'/gnina_rescoring/'+'rescored_'+cnn+'.sdf'
            gnina_cmd = 'cd '+software+' && ./gnina -r '+protein_file+' -l '+sdf+' --autobox_ligand '+ref_file+' -o '+results+' --cnn '+cnn+' --score_only --no_gpu'
            subprocess.call(gnina_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
            gnina_rescoring_results = PandasTools.LoadSDF(results, idName='Pose ID', molColName=None, includeFingerprints=False, removeHs=False)
        else:
            split_files_folder = split_sdf(rescoring_folder+'/gnina_rescoring', sdf, ncpus)
            split_files_sdfs = [os.path.join(split_files_folder, f) for f in os.listdir(split_files_folder) if f.endswith('.sdf')]
            printlog('Rescoring with GNINA')
            global gnina_rescoring_splitted
            def gnina_rescoring_splitted(split_file, protein_file, ref_file, software):
                gnina_folder = rescoring_folder+'/gnina_rescoring/'
                results = gnina_folder+os.path.basename(split_file).split('.')[0]+'_gnina.sdf'
                gnina_cmd = 'cd '+software+' && ./gnina -r '+protein_file+' -l '+split_file+' --autobox_ligand '+ref_file+' -o '+results+' --cnn '+cnn+' --score_only --no_gpu --cpu 1'
                try:
                    subprocess.call(gnina_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
                except Exception as e:
                    printlog('GNINA rescoring failed: '+e)
                return
            with concurrent.futures.ProcessPoolExecutor(max_workers=ncpus) as executor:
                jobs = []
                for split_file in tqdm(split_files_sdfs, desc='Submitting GNINA rescoring jobs', unit='file'):
                    try:
                        job = executor.submit(gnina_rescoring_splitted, split_file, protein_file, ref_file, software)
                        jobs.append(job)
                    except Exception as e:
                        printlog("Error in concurrent futures job creation: "+ str(e))
                for job in tqdm(concurrent.futures.as_completed(jobs), total=len(split_files_sdfs), desc='Rescoring with GNINA', unit='file'):
                    try:
                        res = job.result()
                    except Exception as e:
                        printlog("Error in concurrent futures job run: "+ str(e))
            try:
                gnina_dataframes = [PandasTools.LoadSDF(rescoring_folder+'/gnina_rescoring/'+file, idName='Pose ID', molColName=None,includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True) for file in os.listdir(rescoring_folder+'/gnina_rescoring/') if file.startswith('split') and file.endswith('.sdf')]
            except Exception as e:
                printlog('ERROR: Failed to Load GNINA rescoring SDF file!')
                printlog(e)
            try:
                gnina_rescoring_results = pd.concat(gnina_dataframes)
            except Exception as e:
                printlog('ERROR: Could not combine GNINA rescored poses')
                printlog(e)
        gnina_rescoring_results.rename(columns = {'minimizedAffinity':'GNINA_Affinity', 'CNNscore':'GNINA_CNN_Score', 'CNNaffinity':'GNINA_CNN_Affinity'}, inplace = True)
        gnina_rescoring_results = gnina_rescoring_results[['Pose ID', 'GNINA_Affinity', 'GNINA_CNN_Score', 'GNINA_CNN_Affinity']]
        gnina_rescoring_results.to_csv(rescoring_folder+'/gnina_rescoring/gnina_scores.csv')
        delete_files(rescoring_folder+'/gnina_rescoring/', 'gnina_scores.csv')
        toc = time.perf_counter()
        printlog(f'Rescoring with GNINA complete in {toc-tic:0.4f}!')
        return gnina_rescoring_results
    def vinardo_rescoring(sdf, mp):
        display(sdf)
        tic = time.perf_counter()
        printlog('Rescoring with Vinardo')
        create_temp_folder(rescoring_folder+'/vinardo_rescoring/', silent=True)
        results = rescoring_folder+'/vinardo_rescoring/'+'rescored_vinardo.sdf'
        vinardo_cmd = 'cd '+software+' && ./gnina -r '+protein_file+' -l '+sdf+' --autobox_ligand '+ref_file+' -o '+results+' --score_only --scoring vinardo --cnn_scoring none --no_gpu'
        subprocess.call(vinardo_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
        vinardo_rescoring_results = PandasTools.LoadSDF(results, idName='Pose ID', molColName=None, includeFingerprints=False, removeHs=False)
        vinardo_rescoring_results.rename(columns = {'minimizedAffinity':'Vinardo_Affinity'}, inplace = True)
        vinardo_rescoring_results = vinardo_rescoring_results[['Pose ID', 'Vinardo_Affinity']]
        vinardo_rescoring_results.to_csv(rescoring_folder+'/vinardo_rescoring/vinardo_scores.csv')
        delete_files(rescoring_folder+'/vinardo_rescoring/', 'vinardo_scores.csv')
        toc = time.perf_counter()
        printlog(f'Rescoring with Vinardo complete in {toc-tic:0.4f}!')
        return vinardo_rescoring_results
    def AD4_rescoring(sdf, mp):
        tic = time.perf_counter()
        printlog('Rescoring with AD4')
        create_temp_folder(rescoring_folder+'/AD4_rescoring/', silent=True)
        results = rescoring_folder+'/AD4_rescoring/'+'rescored_AD4.sdf'
        AD4_cmd = 'cd '+software+' && ./gnina -r '+protein_file+' -l '+sdf+' --autobox_ligand '+ref_file+' -o '+results+' --score_only --scoring ad4_scoring --cnn_scoring none --no_gpu'
        subprocess.call(AD4_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
        AD4_rescoring_results = PandasTools.LoadSDF(results, idName='Pose ID', molColName='None', includeFingerprints=False, removeHs=False)
        AD4_rescoring_results.rename(columns = {'minimizedAffinity':'AD4_Affinity'}, inplace = True)
        AD4_rescoring_results = AD4_rescoring_results[['Pose ID', 'AD4_Affinity']]
        AD4_rescoring_results.to_csv(rescoring_folder+f'/AD4_rescoring/AD4_scores.csv')
        delete_files(rescoring_folder+'/AD4_rescoring/', 'AD4_scores.csv')
        toc = time.perf_counter()
        printlog(f'Rescoring with AD4 complete in {toc-tic:0.4f}!')
        return AD4_rescoring_results
    def rfscore_rescoring(sdf, mp):
        tic = time.perf_counter()
        printlog('Rescoring with RFScoreVS')
        create_temp_folder(rescoring_folder+'/rfscorevs_rescoring', silent=True)
        results_path = rescoring_folder+'/rfscorevs_rescoring/rfscorevs_scores.csv'
        if mp == 1 :
            rfscore_cmd = 'cd '+software+' && ./rf-score-vs --receptor '+protein_file+' '+sdf+' -O '+results_path+' -n '+str(int(multiprocessing.cpu_count()-2))
        else:
            rfscore_cmd = 'cd '+software+' && ./rf-score-vs --receptor '+protein_file+' '+sdf+' -O '+results_path+' -n 1'
        subprocess.call(rfscore_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
        rfscore_results = pd.read_csv(results_path, delimiter=',', header=0)
        rfscore_results = rfscore_results.rename(columns={'name': 'Pose ID', 'RFScoreVS_v2':'RFScoreVS'})
        rfscore_results.to_csv(rescoring_folder+'/rfscorevs_rescoring/rfscorevs_scores.csv')
        delete_files(rescoring_folder+'/rfscorevs_rescoring/', 'rfscorevs_scores.csv')
        toc = time.perf_counter()
        printlog(f'Rescoring with RF-Score-VS complete in {toc-tic:0.4f}!')
        return rfscore_results
    def plp_rescoring(sdf, mp):
        tic = time.perf_counter()
        printlog('Rescoring with PLP')
        plants_search_speed = 'speed1'
        ants = '20'
        plp_rescoring_folder = rescoring_folder+'/plp_rescoring/'
        create_temp_folder(plp_rescoring_folder, silent=True)
        #Read protein and ref files generated during PLANTS docking
        plants_protein_mol2 = w_dir+'/temp/plants/protein.mol2'
        plants_ref_mol2 = w_dir+'/temp/plants/ref.mol2'
        #Convert clustered ligand file to .mol2 using open babel
        plants_ligands_mol2 = plp_rescoring_folder+'/ligands.mol2'
        try:
            obabel_command = 'obabel -isdf '+sdf+' -O '+plants_ligands_mol2
            os.system(obabel_command)
        except:
            printlog('ERROR: Failed to convert clustered library file to .mol2!')
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
        plp_rescoring_output.to_csv(rescoring_folder+'/plp_rescoring/plp_scores.csv')
        os.remove(plants_ligands_mol2)
        delete_files(rescoring_folder+'/plp_rescoring/', 'plp_scores.csv')
        toc = time.perf_counter()
        printlog(f'Rescoring with PLP complete in {toc-tic:0.4f}!')
        return plp_rescoring_output
    def chemplp_rescoring(sdf, mp):
        tic = time.perf_counter()
        printlog('Rescoring with CHEMPLP')
        plants_search_speed = 'speed1'
        ants = '20'
        chemplp_rescoring_folder = rescoring_folder+'/chemplp_rescoring/'
        create_temp_folder(chemplp_rescoring_folder, silent=True)
        #Read protein and ref files generated during PLANTS docking
        plants_protein_mol2 = w_dir+'/temp/plants/protein.mol2'
        plants_ref_mol2 = w_dir+'/temp/plants/ref.mol2'
        #Convert clustered ligand file to .mol2 using open babel
        plants_ligands_mol2 = chemplp_rescoring_folder+'/ligands.mol2'
        try:
            obabel_command = 'obabel -isdf '+sdf+' -O '+plants_ligands_mol2
            os.system(obabel_command)
        except:
            printlog('ERROR: Failed to convert clustered library file to .mol2!')
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
        chemplp_rescoring_output.to_csv(rescoring_folder+'/chemplp_rescoring/chemplp_scores.csv')
        os.remove(plants_ligands_mol2)
        delete_files(rescoring_folder+'/chemplp_rescoring/', 'chemplp_scores.csv')
        toc = time.perf_counter()
        printlog(f'Rescoring with CHEMPLP complete in {toc-tic:0.4f}!')
        return chemplp_rescoring_output
    def ECIF_rescoring (sdf, mp):
        printlog('Rescoring with ECIF')
        ECIF_rescoring_folder = rescoring_folder+'/ECIF_rescoring/'
        create_temp_folder(ECIF_rescoring_folder, silent=True)
        split_dir = split_sdf_single(rescoring_folder+'/ECIF_rescoring/', sdf)
        ligands = [split_dir+'/'+x for x in os.listdir(split_dir) if x[-3:] == "sdf"]
        if mp == 0:
            ECIF = [GetECIF(protein_file, ligand, distance_cutoff=6.0) for ligand in ligands]
            ligand_descriptors = [GetRDKitDescriptors(x) for x in ligands]
            all_descriptors = pd.DataFrame(ECIF, columns=PossibleECIF).join(pd.DataFrame(ligand_descriptors, columns=LigandDescriptors))
        if mp == 1:
            global ECIF_rescoring_single
            def ECIF_rescoring_single(ligand, protein_file):
                ECIF = GetECIF(protein_file, ligand, distance_cutoff=6.0)
                ligand_descriptors = GetRDKitDescriptors(ligand)
                all_descriptors_single = pd.DataFrame(ECIF, columns=PossibleECIF).join(pd.DataFrame(ligand_descriptors, columns=LigandDescriptors))
                return all_descriptors_single
            with concurrent.futures.ProcessPoolExecutor(max_workers=ncpus) as executor:
                jobs = []
                all_descriptors = pd.DataFrame()
                for ligand in tqdm(ligands, desc='Submitting ECIF rescoring jobs', unit='file'):
                    try:
                        job = executor.submit(ECIF_rescoring_single, ligand, protein_file)
                        jobs.append(job)
                    except Exception as e:
                        printlog("Error in concurrent futures job creation: "+ str(e))
                for job in tqdm(concurrent.futures.as_completed(jobs), total=len(ligands), desc='Rescoring with ECIF', unit='mol'):
                    try:
                        res = job.result()
                        all_descriptors = pd.concat(all_descriptors, res)
                    except Exception as e:
                        printlog("Error in concurrent futures job run: "+ str(e))
        model = pickle.load(open(software+'/ECIF6_LD_GBT.pkl', 'rb'))
        ids = PandasTools.LoadSDF(sdf, molColName=None, idName='Pose ID')
        ECIF_rescoring_results = pd.DataFrame(ids, columns=["Pose ID"]).join(pd.DataFrame(model.predict(all_descriptors), columns=["ECIF"]))
        ECIF_rescoring_results.to_csv(rescoring_folder+'/ECIF_rescoring/ECIF_scores.csv')
        delete_files(rescoring_folder+'/ECIF_rescoring/', 'ECIF_scores.csv')
        return ECIF_rescoring_results
    def oddt_nnscore_rescoring(sdf, mp):
        tic = time.perf_counter()
        printlog('Rescoring with NNScore')
        rescorers = {'nnscore':NNScore.nnscore()}
        nnscore_rescoring_folder = rescoring_folder+'/nnscore_rescoring/'
        create_temp_folder(nnscore_rescoring_folder, silent=True)
        scorer = rescorers['nnscore']
        pickle = software+'/NNScore_pdbbind2016.pickle'
        scorer = scorer.load(pickle)
        oddt_prot=next(oddt.toolkit.readfile('pdb', protein_file))
        oddt_prot.protein = True
        scorer.set_protein(oddt_prot)
        re_scores = []
        df = PandasTools.LoadSDF(sdf, idName='Pose ID', molColName='Molecule', removeHs=False)
        if mp == 0:
            for mol in tqdm(df['Molecule'], desc='Rescoring with NNScore', unit='mol'):
                Chem.MolToMolFile(mol, nnscore_rescoring_folder+'/temp.sdf')
                oddt_lig = next(oddt.toolkit.readfile('sdf', nnscore_rescoring_folder+'/temp.sdf'))
                scored_mol = scorer.predict_ligand(oddt_lig)
                re_scores.append(float(scored_mol.data['nnscore']))
        else:
            global score_mol
            def score_mol(mol):
                oddt_mol = oddt.toolkit.Molecule(mol)
                scored_mol = scorer.predict_ligand(oddt_mol)
                return float(scored_mol.data['nnscore'])
            with multiprocessing.Pool() as p:
                re_scores = p.map(score_mol, df['Molecule'])
        df['NNScore']=re_scores
        df = df[['Pose ID', 'NNScore']]
        df.to_csv(rescoring_folder+'/nnscore_rescoring/nnscore_scores.csv')
        delete_files(rescoring_folder+'/nnscore_rescoring/', 'nnscore_scores.csv')
        toc = time.perf_counter()
        printlog(f'Rescoring with NNScore complete in {toc-tic:0.4f}!')
        return df
    def oddt_plecscore_rescoring(sdf, mp):
        tic = time.perf_counter()
        printlog('Rescoring with PLECscore')
        rescorers = {'PLECnn_p5_l1_s65536':PLECscore(version='nn')}
        plecscore_rescoring_folder = rescoring_folder+'/plecscore_rescoring/'
        create_temp_folder(plecscore_rescoring_folder, silent=True)
        scorer = rescorers['PLECnn_p5_l1_s65536']
        pickle = software+'/PLECnn_p5_l1_2016.pickle'
        scorer = scorer.load(pickle)
        oddt_prot=next(oddt.toolkit.readfile('pdb', protein_file.replace('.pdb','_pocket.pdb')))
        oddt_prot.protein = True
        scorer.set_protein(oddt_prot)
        re_scores = []
        df = PandasTools.LoadSDF(sdf, idName='Pose ID', molColName='Molecule', removeHs=False)
        if mp == 0:
            for mol in tqdm(df['Molecule'], desc='Rescoring with PLECScore', unit='mol'):
                Chem.MolToMolFile(mol, plecscore_rescoring_folder+'/temp.sdf')
                oddt_lig = next(oddt.toolkit.readfile('sdf', plecscore_rescoring_folder+'/temp.sdf'))
                scored_mol = scorer.predict_ligand(oddt_lig)
                re_scores.append(float(scored_mol.data['PLECnn_p5_l1_s65536']))
        else:
            global score_mol
            def score_mol(mol):
                oddt_mol = oddt.toolkit.Molecule(mol)
                scored_mol = scorer.predict_ligand(oddt_mol)
                return float(scored_mol.data['PLECnn_p5_l1_s65536'])
            with multiprocessing.Pool() as p:
                re_scores = p.map(score_mol, df['Molecule'])
        df['PLECnn']=re_scores
        df = df[['Pose ID', 'PLECnn']]
        df.to_csv(rescoring_folder+'/plecscore_rescoring/plecscore_scores.csv')
        toc = time.perf_counter()
        printlog(f'Rescoring with PLECScore complete in {toc-tic:0.4f}!')
        delete_files(rescoring_folder+'/plecscore_rescoring/', 'plecscore_scores.csv')
        return df
    def SCORCH_rescoring(sdf, mp):
        tic = time.perf_counter()
        SCORCH_rescoring_folder = rescoring_folder + '/SCORCH_rescoring/'
        create_temp_folder(SCORCH_rescoring_folder)
        SCORCH_protein = SCORCH_rescoring_folder + "protein.pdbqt"
        printlog('Converting protein file to .pdbqt ...')
        obabel_command = f'obabel -ipdb {protein_file} -O {SCORCH_protein} --partialcharges gasteiger'
        subprocess.call(obabel_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        #Convert ligands to pdbqt
        sdf_file_name = os.path.basename(sdf).replace('.sdf', '')
        printlog(f'Converting SDF file {sdf_file_name}.sdf to .pdbqt files...')
        split_files_folder = SCORCH_rescoring_folder + f'/split_{sdf_file_name}'
        create_temp_folder(split_files_folder, silent=True)
        num_molecules = parallel_sdf_to_pdbqt(sdf, split_files_folder, ncpus)
        print(f"Converted {num_molecules} molecules.")
        # Run SCORCH
        printlog('Rescoring with SCORCH')
        SCORCH_command = f'python {software}/SCORCH/scorch.py --receptor {SCORCH_protein} --ligand {split_files_folder} --out {SCORCH_rescoring_folder}scoring_results.csv --threads {ncpus} --return_pose_scores'
        subprocess.call(SCORCH_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        #Clean data
        SCORCH_scores = pd.read_csv(SCORCH_rescoring_folder + 'scoring_results.csv')
        SCORCH_scores = SCORCH_scores.rename(columns={'Ligand_ID': 'Pose ID'})
        SCORCH_scores = SCORCH_scores[['SCORCH_pose_score', 'Pose ID']]
        SCORCH_scores.to_csv(SCORCH_rescoring_folder + 'SCORCH_scores.csv')
        delete_files(SCORCH_rescoring_folder, 'SCORCH_scores.csv')
        toc = time.perf_counter()
        printlog(f'Rescoring with SCORCH complete in {toc-tic:0.4f}!')
    def LinF9_rescoring(sdf, mp):
        tic = time.perf_counter()
        create_temp_folder(rescoring_folder+'/LinF9_rescoring/', silent=True)
        if mp == 0:
            printlog('Rescoring with LinF9')
            results = rescoring_folder+'/LinF9_rescoring/'+'rescored_LinF9.sdf'
            LinF9_cmd = 'cd '+software+' && ./smina.static -r '+protein_file+' -l '+sdf+' --autobox_ligand '+ref_file+' -o '+results+' --scoring Lin_F9 --score_only'
            subprocess.call(LinF9_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
            LinF9_rescoring_results = PandasTools.LoadSDF(results, idName='Pose ID', molColName=None, includeFingerprints=False, removeHs=False)
        else:
            split_files_folder = split_sdf(rescoring_folder+'/LinF9_rescoring', sdf, ncpus)
            split_files_sdfs = [os.path.join(split_files_folder, f) for f in os.listdir(split_files_folder) if f.endswith('.sdf')]
            global LinF9_rescoring_splitted
            def LinF9_rescoring_splitted(split_file, protein_file, ref_file, software):
                LinF9_folder = rescoring_folder+'/LinF9_rescoring/'
                results = LinF9_folder+os.path.basename(split_file).split('.')[0]+'_LinF9.sdf'
                LinF9_cmd = 'cd '+software+' && ./smina.static -r '+protein_file+' -l '+split_file+' --autobox_ligand '+ref_file+' -o '+results+' --scoring Lin_F9 --score_only'
                try:
                    subprocess.call(LinF9_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
                except Exception as e:
                    printlog('LinF9 rescoring failed: '+e)
                return
            with concurrent.futures.ProcessPoolExecutor(max_workers=ncpus) as executor:
                jobs = []
                for split_file in tqdm(split_files_sdfs, desc='Submitting LinF9 rescoring jobs', unit='file'):
                    try:
                        job = executor.submit(LinF9_rescoring_splitted, split_file, protein_file, ref_file, software)
                        jobs.append(job)
                    except Exception as e:
                        printlog("Error in concurrent futures job creation: ", str(e))
                for job in tqdm(concurrent.futures.as_completed(jobs), total=len(split_files_sdfs), desc='Rescoring with LinF9', unit='file'):
                    try:
                        res = job.result()
                    except Exception as e:
                        printlog("Error in concurrent futures job run: ", str(e))
            try:
                LinF9_dataframes = [PandasTools.LoadSDF(rescoring_folder+'/LinF9_rescoring/'+file, idName='Pose ID', molColName=None,includeFingerprints=False, embedProps=False, removeHs=False, strictParsing=True) for file in os.listdir(rescoring_folder+'/LinF9_rescoring/') if file.startswith('split') and file.endswith('.sdf')]
            except Exception as e:
                printlog('ERROR: Failed to Load LinF9 rescoring SDF file!')
                printlog(e)
            try:
                LinF9_rescoring_results = pd.concat(LinF9_dataframes)
            except Exception as e:
                printlog('ERROR: Could not combine LinF9 rescored poses')
                printlog(e)
        LinF9_rescoring_results.rename(columns = {'minimizedAffinity':'LinF9_Affinity'}, inplace = True)
        LinF9_rescoring_results = LinF9_rescoring_results[['Pose ID', 'LinF9_Affinity']]
        LinF9_rescoring_results.to_csv(rescoring_folder+'/LinF9_rescoring/LinF9_scores.csv')
        delete_files(rescoring_folder+'/LinF9_rescoring/', 'LinF9_scores.csv')
        toc = time.perf_counter()
        printlog(f'Rescoring with LinF9 complete in {toc-tic:0.4f}!')
        return LinF9_rescoring_results
    def delta_Lin_F9_XGB_rescoring(sdf, mp):
        return
    def AAScore_rescoring(sdf, mp):
        tic = time.perf_counter()
        create_temp_folder(rescoring_folder+'/AAScore_rescoring/', silent=True)
        pocket = protein_file.replace('.pdb', '_pocket.pdb')
        if mp == 0:
            printlog('Rescoring with AAScore')
            results = rescoring_folder+'/AAScore_rescoring/rescored_AAScore.csv'
            AAscore_cmd = 'python '+software+'/AA-Score-Tool-main/AA_Score.py --Rec '+pocket+' --Lig '+sdf+' --Out '+results
            subprocess.call(AAscore_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
            AAScore_rescoring_results=pd.read_csv(results, delimiter='\t', header=None, names=['Pose ID', 'AAScore'])
        else:
            split_files_folder = split_sdf(rescoring_folder+'/AAScore_rescoring', sdf, ncpus)
            split_files_sdfs = [os.path.join(split_files_folder, f) for f in os.listdir(split_files_folder) if f.endswith('.sdf')]
            global AAScore_rescoring_splitted
            def AAScore_rescoring_splitted(split_file, software):
                AAScore_folder = rescoring_folder+'/AAScore_rescoring/'
                results = AAScore_folder+os.path.basename(split_file).split('.')[0]+'_AAScore.csv'
                AAScore_cmd = 'python '+software+'/AA-Score-Tool-main/AA_Score.py --Rec '+pocket+' --Lig '+split_file+' --Out '+results
                try:
                    subprocess.call(AAScore_cmd, shell=True)
                except Exception as e:
                    printlog('AAScore rescoring failed: '+e)
                return
            with concurrent.futures.ProcessPoolExecutor(max_workers=ncpus) as executor:
                jobs = []
                for split_file in tqdm(split_files_sdfs, desc='Submitting AAScore rescoring jobs', unit='file'):
                    try:
                        job = executor.submit(AAScore_rescoring_splitted, split_file, software)
                        jobs.append(job)
                    except Exception as e:
                        printlog("Error in concurrent futures job creation: ", str(e))
                for job in tqdm(concurrent.futures.as_completed(jobs), total=len(split_files_sdfs), desc='Rescoring with AAScore', unit='file'):
                    try:
                        res = job.result()
                    except Exception as e:
                        printlog("Error in concurrent futures job run: ", str(e))
            try:
                AAScore_dataframes = [pd.read_csv(rescoring_folder+'/AAScore_rescoring/'+file, delimiter='\t', header=None, names=['Pose ID', 'AAScore']) for file in os.listdir(rescoring_folder+'/AAScore_rescoring/') if file.startswith('split') and file.endswith('.csv')]
            except Exception as e:
                printlog('ERROR: Failed to Load AAScore rescoring SDF file!')
                printlog(e)
            try:
                AAScore_rescoring_results = pd.concat(AAScore_dataframes)
            except Exception as e:
                printlog('ERROR: Could not combine AAScore rescored poses')
                printlog(e)
            else:
                delete_files(rescoring_folder+'/AAScore_rescoring/', 'AAScore_scores.csv')
        AAScore_rescoring_results.to_csv(rescoring_folder+f'/AAScore_rescoring/AAScore_scores.csv')
        toc = time.perf_counter()
        printlog(f'Rescoring with AAScore complete in {toc-tic:0.4f}!')
        return
    rescoring_functions = {'gnina': gnina_rescoring, 'vinardo': vinardo_rescoring, 'AD4': AD4_rescoring, 
                        'rfscorevs': rfscore_rescoring, 'plp': plp_rescoring, 'chemplp': chemplp_rescoring,
                        'nnscore': oddt_nnscore_rescoring, 'plecscore': oddt_plecscore_rescoring, 'LinF9':LinF9_rescoring, 
                        'AAScore':AAScore_rescoring, 'ECIF':ECIF_rescoring, 'SCORCH':SCORCH_rescoring}
    for function in functions:
        if os.path.isdir(rescoring_folder+f'/{function}_rescoring') == False:
            rescoring_functions[function](clustered_sdf, mp)
        else:
            printlog(f'/{function}_rescoring folder already exists, skipping {function} rescoring')
    if os.path.isfile(rescoring_folder+'/allposes_rescored.csv') == False:
        score_files = [f'{function}_scores.csv' for function in functions]
        printlog(f'Combining all score for {rescoring_folder}')
        csv_files = [os.path.join(subdir, file) for subdir, dirs, files in os.walk(rescoring_folder) for file in files if file in score_files]
        csv_dfs = [pd.read_csv(f, index_col=0) for f in csv_files]
        combined_dfs = csv_dfs[0]
        for df in tqdm(csv_dfs[1:], desc='Combining scores', unit='files'):
            combined_dfs = pd.merge(combined_dfs, df, left_on='Pose ID', right_on='Pose ID', how='inner')
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
        combined_dfs.to_csv(rescoring_folder+'/allposes_rescored.csv', index=False)
    toc = time.perf_counter()
    printlog(f'Rescoring complete in {toc-tic:0.4f}!')
    return
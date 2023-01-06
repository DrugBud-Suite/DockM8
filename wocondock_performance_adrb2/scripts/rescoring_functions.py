import os
import shutil
import subprocess
import pandas as pd
import functools
from rdkit import Chem
from rdkit.Chem import PandasTools
import oddt
from oddt.scoring.functions import NNScore
from oddt.scoring.functions import RFScore
from oddt.scoring.functions.PLECscore import PLECscore
import tqdm

def rescore_all(w_dir, protein_file, ref_file, software, clustered_sdf):
    rescoring_folder_name = os.path.basename(clustered_sdf).split('/')[-1]
    rescoring_folder_name = rescoring_folder_name.replace('.sdf', '')
    rescoring_folder = w_dir+'/temp/rescoring_'+rescoring_folder_name
    if os.path.isdir(rescoring_folder) == True:
        print('Deleting existing rescoring folder')
        shutil.rmtree(rescoring_folder)
    else:
        os.mkdir(rescoring_folder)
    def gnina_rescoring(sdf):
        print('Rescoring with GNINA')
        gnina_rescoring_folder = rescoring_folder+'/gnina_rescoring/'
        try:
            os.makedirs(gnina_rescoring_folder)
        except:
            pass
        cnn = 'crossdock_default2018'
        results = gnina_rescoring_folder+'/rescored_'+cnn+'.sdf'
        log_file = gnina_rescoring_folder+'log_'+cnn+'.txt'
        gnina_cmd = 'cd '+software+' && ./gnina -r '+protein_file+' -l '+sdf+' --autobox_ligand '+ref_file+' -o '+results+' --cnn '+cnn+' --log '+log_file+' --score_only'
        subprocess.call(gnina_cmd, shell=True)

        gnina_rescoring_results = PandasTools.LoadSDF(results, idName='Pose ID', molColName='None', includeFingerprints=False, removeHs=False)
        gnina_rescoring_results.drop(['None', 'CNN_VS'], axis=1, inplace=True)
        gnina_rescoring_results.rename(columns = {'minimizedAffinity':'GNINA_Affinity', 'CNNscore':'GNINA_CNN_Score', 'CNNaffinity':'GNINA_CNN_Affinity'}, inplace = True)
        return gnina_rescoring_results
    def vinardo_rescoring(sdf):
        print('Rescoring with Vinardo')
        vinardo_rescoring_folder = rescoring_folder+'/vinardo_rescoring/'
        try:
            os.makedirs(vinardo_rescoring_folder)
        except:
            pass
        results = vinardo_rescoring_folder+'/rescored_vinardo.sdf'
        log_file = vinardo_rescoring_folder+'log.txt'

        vinardo_cmd = 'cd '+software+' && ./gnina -r '+protein_file+' -l '+sdf+' --autobox_ligand '+ref_file+' -o '+results+' --log '+log_file+' --score_only --scoring vinardo --cnn_scoring none'
        subprocess.call(vinardo_cmd, shell=True)

        vinardo_rescoring_results = PandasTools.LoadSDF(results, idName='Pose ID', molColName='None', includeFingerprints=False, removeHs=False)
        vinardo_rescoring_results.drop('None', axis=1, inplace=True)
        vinardo_rescoring_results.rename(columns = {'minimizedAffinity':'Vinardo_Affinity'}, inplace = True)
        return vinardo_rescoring_results
    def AD4_rescoring(sdf):
        print('Rescoring with AD4')
        AD4_rescoring_folder = rescoring_folder+'/AD4_rescoring/'
        try:
            os.makedirs(AD4_rescoring_folder)
        except:
            pass
        results = AD4_rescoring_folder+'/rescored_AD4.sdf'
        log_file = AD4_rescoring_folder+'log.txt'

        AD4_cmd = 'cd '+software+' && ./gnina -r '+protein_file+' -l '+sdf+' --autobox_ligand '+ref_file+' -o '+results+' --log '+log_file+' --score_only --scoring ad4_scoring --cnn_scoring none'

        subprocess.call(AD4_cmd, shell=True)

        AD4_rescoring_results = PandasTools.LoadSDF(results, idName='Pose ID', molColName='None', includeFingerprints=False, removeHs=False)
        AD4_rescoring_results.drop('None', axis=1, inplace=True)
        AD4_rescoring_results.rename(columns = {'minimizedAffinity':'AD4_Affinity'}, inplace = True)
        return AD4_rescoring_results
    def oddt_rfscoreV1_rescoring(method, sdf):
        print('Rescoring with RFScoreV1')
        rescorers = {'rfscore':RFScore.rfscore()}
        rfscore_rescoring_folder = rescoring_folder+'/rfscoreV1_rescoring/'
        try:
            os.makedirs(rfscore_rescoring_folder)
        except:
            print('Could not create RFscoreV1 rescoring folder')
        scorer = rescorers[method]
        pickle = software+'/RFScore_v1_pdbbind2016.pickle'
        scorer = scorer.load(pickle)
        oddt_prot=next(oddt.toolkit.readfile('pdb', protein_file))
        oddt_prot.protein = True
        scorer.set_protein(oddt_prot)
        re_scores = []
        df = PandasTools.LoadSDF(sdf, idName='Pose ID', molColName='Molecule', removeHs=False)
        for mol in df['Molecule']:
            Chem.MolToMolFile(mol, rfscore_rescoring_folder+'/temp.sdf')
            oddt_lig = next(oddt.toolkit.readfile('sdf', rfscore_rescoring_folder+'/temp.sdf'))
            scored_mol = scorer.predict_ligand(oddt_lig)
            re_scores.append(float(scored_mol.data[method+'_v1']))
        df['RFScoreV1']=re_scores
        df.drop(['Molecule'], axis=1, inplace=True)
        return df
    def oddt_rfscoreV2_rescoring(method, sdf):
        print('Rescoring with RFScoreV2')
        rescorers = {'rfscore':RFScore.rfscore()}
        rfscore_rescoring_folder = rescoring_folder+'/rfscoreV2_rescoring/'
        try:
            os.makedirs(rfscore_rescoring_folder)
        except:
            print('Could not create RFscoreV2 rescoring folder')
        scorer = rescorers[method]
        pickle = software+'/RFScore_v2_pdbbind2016.pickle'
        scorer = scorer.load(pickle)
        oddt_prot=next(oddt.toolkit.readfile('pdb', protein_file))
        oddt_prot.protein = True
        scorer.set_protein(oddt_prot)
        re_scores = []
        df = PandasTools.LoadSDF(sdf, idName='Pose ID', molColName='Molecule', removeHs=False)
        for mol in df['Molecule']:
            Chem.MolToMolFile(mol, rfscore_rescoring_folder+'/temp.sdf')
            oddt_lig = next(oddt.toolkit.readfile('sdf', rfscore_rescoring_folder+'/temp.sdf'))
            scored_mol = scorer.predict_ligand(oddt_lig)
            re_scores.append(float(scored_mol.data[method+'_v2']))
        df['RFScoreV2']=re_scores
        df.drop(['Molecule'], axis=1, inplace=True)
        return df
    def oddt_rfscoreV3_rescoring(method, sdf):
        print('Rescoring with RFScoreV3')
        rescorers = {'rfscore':RFScore.rfscore()}
        rfscore_rescoring_folder = rescoring_folder+'/rfscoreV3_rescoring/'
        try:
            os.makedirs(rfscore_rescoring_folder)
        except:
            print('Could not create RFscoreV3 rescoring folder')
        scorer = rescorers[method]
        pickle = software+'/RFScore_v3_pdbbind2016.pickle'
        scorer = scorer.load(pickle)
        oddt_prot=next(oddt.toolkit.readfile('pdb', protein_file))
        oddt_prot.protein = True
        scorer.set_protein(oddt_prot)
        re_scores = []
        df = PandasTools.LoadSDF(sdf, idName='Pose ID', molColName='Molecule', removeHs=False)
        for mol in df['Molecule']:
            Chem.MolToMolFile(mol, rfscore_rescoring_folder+'/temp.sdf')
            oddt_lig = next(oddt.toolkit.readfile('sdf', rfscore_rescoring_folder+'/temp.sdf'))
            scored_mol = scorer.predict_ligand(oddt_lig)
            re_scores.append(float(scored_mol.data[method+'_v3']))
        df['RFScoreV3']=re_scores
        df.drop(['Molecule'], axis=1, inplace=True)
        return df
    def plp_rescoring(sdf):
        print('Rescoring with PLP')
        plants_search_speed = 'speed1'
        ants = '20'
        plp_rescoring_folder = rescoring_folder+'/plp_rescoring/'
        try:
            os.makedirs(plp_rescoring_folder)
        except:
            pass
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
        os.system(plp_rescoring_command)
        #Fetch results
        results_csv_location = plp_rescoring_folder+'results/ranking.csv'
        plp_results = pd.read_csv(results_csv_location, sep=',', header=0)
        plp_results.rename(columns = {'TOTAL_SCORE':'PLP'}, inplace = True)
        for i, row in plp_results.iterrows():
            split = row['LIGAND_ENTRY'].split('_')
            plp_results.loc[i, ['Pose ID']] = split[0]+'_'+split[1]+'_'+split[2]
        plp_rescoring_output = plp_results[['Pose ID', 'PLP']]
        return plp_rescoring_output
    def chemplp_rescoring(sdf):
        print('Rescoring with CHEMPLP')
        plants_search_speed = 'speed1'
        ants = '20'
        chemplp_rescoring_folder = rescoring_folder+'/chemplp_rescoring/'
        try:
            os.makedirs(chemplp_rescoring_folder)
        except:
            pass
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
        os.system(chemplp_rescoring_command)
        #Fetch results
        results_csv_location = chemplp_rescoring_folder+'results/ranking.csv'
        chemplp_results = pd.read_csv(results_csv_location, sep=',', header=0)
        chemplp_results.rename(columns = {'TOTAL_SCORE':'CHEMPLP'}, inplace = True)
        for i, row in chemplp_results.iterrows():
            split = row['LIGAND_ENTRY'].split('_')
            chemplp_results.loc[i, ['Pose ID']] = split[0]+'_'+split[1]+'_'+split[2]
        chemplp_rescoring_output = chemplp_results[['Pose ID', 'CHEMPLP']]
        return chemplp_rescoring_output
    def oddt_nnscore_rescoring(method, sdf):
        print('Rescoring with NNScore')
        rescorers = {'nnscore':NNScore.nnscore()}
        nnscore_rescoring_folder = rescoring_folder+'/nnscore_rescoring/'
        try:
            os.makedirs(nnscore_rescoring_folder)
        except:
            print('Could not create nnscore rescoring folder')
        scorer = rescorers[method]
        pickle = software+'/NNScore_pdbbind2016.pickle'
        scorer = scorer.load(pickle)
        oddt_prot=next(oddt.toolkit.readfile('pdb', protein_file))
        oddt_prot.protein = True
        scorer.set_protein(oddt_prot)
        re_scores = []
        df = PandasTools.LoadSDF(sdf, idName='Pose ID', molColName='Molecule', removeHs=False)
        for mol in df['Molecule']:
            Chem.MolToMolFile(mol, nnscore_rescoring_folder+'/temp.sdf')
            oddt_lig = next(oddt.toolkit.readfile('sdf', nnscore_rescoring_folder+'/temp.sdf'))
            scored_mol = scorer.predict_ligand(oddt_lig)
            re_scores.append(float(scored_mol.data[method]))
        df['NNScore']=re_scores
        df.drop(['Molecule'], axis=1, inplace=True)
        return df
    def oddt_plecscore_rescoring(method, sdf):
        print('Rescoring with PLECscore')
        rescorers = {'PLECnn_p5_l1_s65536':PLECscore(version='nn')}
        plecscore_rescoring_folder = rescoring_folder+'/plecscore_rescoring/'
        try:
            os.makedirs(plecscore_rescoring_folder)
        except:
            print('Could not create plecscore rescoring folder')
        scorer = rescorers[method]
        pickle = software+'/PLECnn_p5_l1_2016.pickle'
        scorer = scorer.load(pickle)
        oddt_prot=next(oddt.toolkit.readfile('pdb', protein_file))
        oddt_prot.protein = True
        scorer.set_protein(oddt_prot)
        re_scores = []
        df = PandasTools.LoadSDF(sdf, idName='Pose ID', molColName='Molecule', removeHs=False)
        for mol in df['Molecule']:
            Chem.MolToMolFile(mol, plecscore_rescoring_folder+'/temp.sdf')
            oddt_lig = next(oddt.toolkit.readfile('sdf', plecscore_rescoring_folder+'/temp.sdf'))
            scored_mol = scorer.predict_ligand(oddt_lig)
            re_scores.append(float(scored_mol.data[method]))
        df['PLECnn']=re_scores
        df.drop(['Molecule'], axis=1, inplace=True)
        return df
    GNINA_df = gnina_rescoring(clustered_sdf)
    VINARDO_df = vinardo_rescoring(clustered_sdf)
    AD4_df = AD4_rescoring(clustered_sdf)
    rfscoreV1_df = oddt_rfscoreV1_rescoring('rfscore', clustered_sdf)
    rfscoreV2_df = oddt_rfscoreV2_rescoring('rfscore', clustered_sdf)
    rfscoreV3_df = oddt_rfscoreV3_rescoring('rfscore', clustered_sdf)
    plp_df = plp_rescoring(clustered_sdf)
    chemplp_df = chemplp_rescoring(clustered_sdf)
    nn_score_df = oddt_nnscore_rescoring('nnscore', clustered_sdf)
    #plec_df = oddt_plecscore_rescoring('PLECnn_p5_l1_s65536', clustered_sdf)
    rescored_dfs = [GNINA_df, VINARDO_df, AD4_df, rfscoreV1_df, rfscoreV2_df, rfscoreV3_df, plp_df, chemplp_df, nn_score_df]#, plec_df]
    combined_dfs = functools.reduce(lambda  left,right: pd.merge(left,right,on=['Pose ID'],how='outer'), rescored_dfs)
    first_column = combined_dfs.pop('Pose ID')
    combined_dfs.insert(0, 'Pose ID', first_column)
    columns=combined_dfs.columns
    col=columns[1:]
    for c in col.tolist():
        if c == 'Pose ID':
            pass
        if combined_dfs[c].dtypes is not float:
            combined_dfs[c] = pd.to_numeric(combined_dfs[c])
        else:
            pass
    combined_dfs.to_csv(rescoring_folder+'/allposes_rescored.csv')
    print('Rescoring complete!')
    return combined_dfs
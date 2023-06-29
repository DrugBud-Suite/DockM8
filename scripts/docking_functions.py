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
from pathlib import Path


def qvinaw_docking(
        protein_file,
        pocket_definition,
        exhaustiveness,
        n_poses):
    printlog('Docking library using QVINAW...')
    tic = time.perf_counter()
    # Make required folders
    protein_file_path = Path(protein_file)
    w_dir = protein_file_path.parent
    library = w_dir / 'temp' / 'final_library.sdf'
    qvinaw_folder = w_dir / 'temp' / 'qvinaw'
    pdbqt_files_folder = qvinaw_folder / 'pdbqt_files'
    pdbqt_files_folder.mkdir(parents=True, exist_ok=True)
    results_path = qvinaw_folder / 'docked'
    results_path.mkdir(parents=True, exist_ok=True)
    # Make .pdbqt files
    protein_file_pdbqt = convert_pdb_to_pdbqt(protein_file)
    try:
        print('Converting molecules to .pdbqt using Meeko')
        meeko_to_pdbqt(str(library), str(pdbqt_files_folder))
    except Exception as e:
        print('Failed to convert sdf file to .pdbqt')
        print(e)
    pdbqt_files = list(pdbqt_files_folder.glob('*.pdbqt'))
    # Dock
    for pdbqt_file in tqdm(
            pdbqt_files,
            desc='Docking with QVINAW',
            total=len(pdbqt_files)):
        qvinaw_cmd = (
            f"{software / 'qvina-w'}" +
            f" --receptor {protein_file_pdbqt}" +
            f" --ligand {pdbqt_file}" +
            f" --out {str(pdbqt_file).replace('pdbqt_files', 'docked')}" +
            f" --center_x {pocket_definition['center'][0]}" +
            f" --center_y {pocket_definition['center'][1]}" +
            f" --center_z {pocket_definition['center'][2]}" +
            f" --size_x {pocket_definition['size'][0]}" +
            f" --size_y {pocket_definition['size'][1]}" +
            f" --size_z {pocket_definition['size'][2]}" +
            f" --exhaustiveness {exhaustiveness}" +
            " --cpu 1" +
            f" --num_modes {n_poses}"
        )
        try:
            subprocess.call(
                qvinaw_cmd,
                shell=True,
                stdout=DEVNULL,
                stderr=STDOUT)
        except Exception as e:
            printlog('QVINAW docking failed: ' + e)
    toc = time.perf_counter()
    printlog(f'Docking with QVINAW complete in {toc-tic:0.4f}!')
    tic = time.perf_counter()
    qvinaw_docking_results = qvinaw_folder / 'qvinaw_poses.sdf'
    printlog('Fetching QVINAW poses...')

    results_pdbqt_files = results_path.glob('*.pdbqt')

    for pdbqt_file in results_pdbqt_files:
        # Convert to sdf
        sdf_file = pdbqt_file.with_suffix('.sdf')
        obabel_command = f'obabel {pdbqt_file} -O {sdf_file}'
        try:
            subprocess.call(
                obabel_command,
                shell=True,
                stdout=DEVNULL,
                stderr=STDOUT)
        except Exception as e:
            print(f'Conversion from PDBQT to SDF failed: {e}')
    try:
        sdf_files = results_path.glob('*.sdf')
        qvinaw_poses = []
        for sdf in sdf_files:
            df = PandasTools.LoadSDF(
                str(sdf),
                idName='ID',
                molColName='Molecule',
                includeFingerprints=False,
                embedProps=False,
                removeHs=False,
                strictParsing=False)
            list_ = [*range(1, int(n_poses) + 1, 1)]
            ser = list_ * (len(df) // len(list_))
            df['Pose ID'] = [
                f'{sdf.name.replace(".sdf", "")}_QVINAW_{num}' for num,
                (_,
                 row) in zip(
                    ser +
                    list_[
                        :len(df) -
                        len(ser)],
                    df.iterrows())]
            df = df.rename(columns={'REMARK': 'QVINAW_Affinity'})[
                ['Molecule', 'QVINAW_Affinity', 'Pose ID']]
            df['QVINAW_Affinity'] = df['QVINAW_Affinity'].str.split().str[2]
            qvinaw_poses.append(df)
        qvinaw_poses = pd.concat(qvinaw_poses)
        PandasTools.WriteSDF(
            qvinaw_poses,
            str(qvinaw_docking_results),
            molColName='Molecule',
            idName='Pose ID',
            properties=list(
                qvinaw_poses.columns))
    except Exception as e:
        printlog('ERROR: Failed to combine QVINAW SDF file!')
        printlog(e)
    else:
        shutil.rmtree(pdbqt_files_folder, ignore_errors=True)
        shutil.rmtree(results_path, ignore_errors=True)
    return str(qvinaw_docking_results)

def qvina2_docking(
        protein_file,
        pocket_definition,
        exhaustiveness,
        n_poses):
    printlog('Docking library using QVINA2...')
    tic = time.perf_counter()
    # Make required folders
    protein_file_path = Path(protein_file)
    w_dir = protein_file_path.parent
    library = w_dir / 'temp' / 'final_library.sdf'
    qvina2_folder = w_dir / 'temp' / 'qvina2'
    pdbqt_files_folder = qvina2_folder / 'pdbqt_files'
    pdbqt_files_folder.mkdir(parents=True, exist_ok=True)
    results_path = qvina2_folder / 'docked'
    results_path.mkdir(parents=True, exist_ok=True)
    # Make .pdbqt files
    protein_file_pdbqt = convert_pdb_to_pdbqt(protein_file)
    try:
        print('Converting molecules to .pdbqt using Meeko')
        meeko_to_pdbqt(str(library), str(pdbqt_files_folder))
    except Exception as e:
        print('Failed to convert sdf file to .pdbqt')
        print(e)
    pdbqt_files = list(pdbqt_files_folder.glob('*.pdbqt'))
    # Dock
    for pdbqt_file in tqdm(
            pdbqt_files,
            desc='Docking with QVINA2',
            total=len(pdbqt_files)):
        qvina2_cmd = (
            f"{software / 'qvina2.1'}" +
            f" --receptor {protein_file_pdbqt}" +
            f" --ligand {pdbqt_file}" +
            f" --out {str(pdbqt_file).replace('pdbqt_files', 'docked')}" +
            f" --center_x {pocket_definition['center'][0]}" +
            f" --center_y {pocket_definition['center'][1]}" +
            f" --center_z {pocket_definition['center'][2]}" +
            f" --size_x {pocket_definition['size'][0]}" +
            f" --size_y {pocket_definition['size'][1]}" +
            f" --size_z {pocket_definition['size'][2]}" +
            f" --exhaustiveness {exhaustiveness}" +
            " --cpu 1" +
            f" --num_modes {n_poses}"
        )
        try:
            subprocess.call(
                qvina2_cmd,
                shell=True,
                stdout=DEVNULL,
                stderr=STDOUT)
        except Exception as e:
            printlog('QVINA2 docking failed: ' + e)
    toc = time.perf_counter()
    printlog(f'Docking with QVINA2 complete in {toc-tic:0.4f}!')
    tic = time.perf_counter()
    printlog('Fetching QVINA2 poses...')
    results_pdbqt_files = results_path.glob('*.pdbqt')
    qvina2_docking_results = qvina2_folder / 'qvina2_poses.sdf'
    for pdbqt_file in results_pdbqt_files:
        # Convert to sdf
        sdf_file = pdbqt_file.with_suffix('.sdf')
        obabel_command = f'obabel {pdbqt_file} -O {sdf_file}'
        try:
            subprocess.call(
                obabel_command,
                shell=True,
                stdout=DEVNULL,
                stderr=STDOUT)
        except Exception as e:
            print(f'Conversion from PDBQT to SDF failed: {e}')
    try:
        sdf_files = results_path.glob('*.sdf')
        qvina2_poses = []
        for sdf in sdf_files:
            df = PandasTools.LoadSDF(
                str(sdf),
                idName='ID',
                molColName='Molecule',
                includeFingerprints=False,
                embedProps=False,
                removeHs=False,
                strictParsing=False)
            list_ = [*range(1, int(n_poses) + 1, 1)]
            ser = list_ * (len(df) // len(list_))
            df['Pose ID'] = [
                f'{sdf.name.replace(".sdf", "")}_QVINA2_{num}' for num,
                (_,
                 row) in zip(
                    ser +
                    list_[
                        :len(df) -
                        len(ser)],
                    df.iterrows())]
            df = df.rename(columns={'REMARK': 'QVINA2_Affinity'})[
                ['Molecule', 'QVINA2_Affinity', 'Pose ID']]
            df['QVINA2_Affinity'] = df['QVINA2_Affinity'].str.split().str[2]
            qvina2_poses.append(df)
        qvina2_poses = pd.concat(qvina2_poses)
        PandasTools.WriteSDF(
            qvina2_poses,
            str(qvina2_docking_results),
            molColName='Molecule',
            idName='Pose ID',
            properties=list(
                qvina2_poses.columns))
    except Exception as e:
        printlog('ERROR: Failed to combine QVINA2 SDF file!')
        printlog(e)
    else:
        shutil.rmtree(pdbqt_files_folder, ignore_errors=True)
        shutil.rmtree(results_path, ignore_errors=True)
    return str(qvina2_docking_results)

def smina_docking(
        protein_file,
        pocket_definition,
        exhaustiveness,
        n_poses):
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
    w_dir = Path(protein_file).parent
    library = w_dir / 'temp' / 'final_library.sdf'
    smina_folder = w_dir / 'temp' / 'smina'
    smina_folder.mkdir(parents=True, exist_ok=True)
    results_path = smina_folder / 'docked.sdf'
    log = smina_folder / 'log.txt'
    smina_cmd = (
        f"{software / 'gnina'}" +
        f" --receptor {protein_file}" +
        f" --ligand {library}" +
        f" --out {results_path}" +
        f" --center_x {pocket_definition['center'][0]}" +
        f" --center_y {pocket_definition['center'][1]}" +
        f" --center_z {pocket_definition['center'][2]}" +
        f" --size_x {pocket_definition['size'][0]}" +
        f" --size_y {pocket_definition['size'][1]}" +
        f" --size_z {pocket_definition['size'][2]}" +
        f" --exhaustiveness {exhaustiveness}" +
        " --cpu 1" +
        f" --num_modes {n_poses}" +
        f" --log {log}" +
        " --cnn_scoring none --no_gpu"
    )
    try:
        subprocess.call(smina_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
    except Exception as e:
        printlog('SMINA docking failed: ' + e)
    toc = time.perf_counter()
    printlog(f'Docking with SMINA complete in {toc-tic:0.4f}!')

    tic = time.perf_counter()
    printlog('Fetching SMINA poses...')
    try:
        smina_df = PandasTools.LoadSDF(
            str(results_path),
            idName='ID',
            molColName='Molecule',
            includeFingerprints=False,
            embedProps=False,
            removeHs=False,
            strictParsing=True)
        list_ = [*range(1, int(n_poses) + 1, 1)]
        ser = list_ * (len(smina_df) // len(list_))
        smina_df['Pose ID'] = [
            f'{row["ID"]}_SMINA_{num}' for num,
            (_,
             row) in zip(
                ser +
                list_[
                    :len(smina_df) -
                    len(ser)],
                smina_df.iterrows())]
        smina_df.rename(
            columns={
                'minimizedAffinity': 'SMINA_Affinity'},
            inplace=True)
    except Exception as e:
        printlog('ERROR: Failed to Load SMINA poses SDF file!')
        printlog(e)
    try:
        PandasTools.WriteSDF(smina_df,
                             str(smina_folder / 'smina_poses.sdf'),
                             molColName='Molecule',
                             idName='Pose ID',
                             properties=list(smina_df.columns))
        toc = time.perf_counter()
        printlog(f'Cleaned up SMINA poses in {toc-tic:0.4f}!')
    except Exception as e:
        printlog('ERROR: Failed to Write SMINA poses SDF file!')
        printlog(e)
    return str(smina_folder / 'smina_poses.sdf')

def gnina_docking(
        protein_file,
        pocket_definition,
        exhaustiveness,
        n_poses):
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
    w_dir = Path(protein_file).parent
    library = w_dir / 'temp' / 'final_library.sdf'
    gnina_folder = w_dir / 'temp' / 'gnina'
    gnina_folder.mkdir(parents=True, exist_ok=True)
    results_path = gnina_folder / 'docked.sdf'
    log = gnina_folder / 'log.txt'
    gnina_cmd = (
        f"{software / 'gnina'}" +
        f" --receptor {protein_file}" +
        f" --ligand {library}" +
        f" --out {results_path}" +
        f" --center_x {pocket_definition['center'][0]}" +
        f" --center_y {pocket_definition['center'][1]}" +
        f" --center_z {pocket_definition['center'][2]}" +
        f" --size_x {pocket_definition['size'][0]}" +
        f" --size_y {pocket_definition['size'][1]}" +
        f" --size_z {pocket_definition['size'][2]}" +
        f" --exhaustiveness {exhaustiveness}" +
        " --cpu 1" +
        f" --num_modes {n_poses}" +
        f" --log {log}" +
        " --cnn_scoring rescore --cnn crossdock_default2018 --no_gpu"
    )
    try:
        subprocess.call(gnina_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
    except Exception as e:
        printlog('GNINA docking failed: ' + e)
    toc = time.perf_counter()
    printlog(f'Docking with GNINA complete in {toc-tic:0.4f}!')

    tic = time.perf_counter()
    printlog('Fetching GNINA poses...')
    try:
        gnina_df = PandasTools.LoadSDF(
            str(results_path),
            idName='ID',
            molColName='Molecule',
            includeFingerprints=False,
            embedProps=False,
            removeHs=False,
            strictParsing=True)
        list_ = [*range(1, int(n_poses) + 1, 1)]
        ser = list_ * (len(gnina_df) // len(list_))
        gnina_df['Pose ID'] = [
            f'{row["ID"]}_GNINA_{num}' for num,
            (_,
             row) in zip(
                ser +
                list_[
                    :len(gnina_df) -
                    len(ser)],
                gnina_df.iterrows())]
        gnina_df.rename(
            columns={
                'minimizedAffinity': 'GNINA_Affinity'},
            inplace=True)
    except Exception as e:
        printlog('ERROR: Failed to Load GNINA poses SDF file!')
        printlog(e)
    try:
        PandasTools.WriteSDF(gnina_df,
                             str(gnina_folder / 'gnina_poses.sdf'),
                             molColName='Molecule',
                             idName='Pose ID',
                             properties=list(gnina_df.columns))
        toc = time.perf_counter()
        printlog(f'Cleaned up GNINA poses in {toc-tic:0.4f}!')
    except Exception as e:
        printlog('ERROR: Failed to Write GNINA poses SDF file!')
        printlog(e)
    return str(gnina_folder / 'gnina_poses.sdf')

def plants_docking(
        protein_file, 
        pocket_definition, 
        n_poses):
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
    w_dir = Path(protein_file).parent
    # Define initial variables
    plants_folder = w_dir / 'temp' / 'plants'
    plants_folder.mkdir(parents=True, exist_ok=True)
    # Convert protein file to .mol2 using open babel
    plants_protein_mol2 = w_dir / 'temp' / 'plants' / 'protein.mol2'
    try:
        printlog('Converting protein file to .mol2 format for PLANTS docking...')
        obabel_command = 'obabel -ipdb ' + \
            str(protein_file) + ' -O ' + str(plants_protein_mol2)
        subprocess.call(
            obabel_command,
            shell=True,
            stdout=DEVNULL,
            stderr=STDOUT)
    except Exception as e:
        printlog('ERROR: Failed to convert protein file to .mol2!')
        printlog(e)
    # Convert prepared ligand file to .mol2 using open babel
    library = w_dir / 'temp' / 'final_library.sdf'
    plants_library_mol2 = plants_folder / 'ligands.mol2'
    try:
        obabel_command = 'obabel -isdf ' + \
            str(library) + ' -O ' + str(plants_library_mol2)
        os.system(obabel_command)
    except Exception as e:
        printlog('ERROR: Failed to convert docking library file to .mol2!')
        printlog(e)
    # Generate plants config file
    plants_docking_config_path = plants_folder / 'config.config'
    plants_config = ['# search algorithm\n',
                     'search_speed speed1\n',
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
                     'protein_file ' + str(plants_protein_mol2) + '\n',
                     'ligand_file ' + str(plants_library_mol2) + '\n',

                     '# output\n',
                     'output_dir ' + str(plants_folder / 'results') + '\n',

                     '# write single mol2 files (e.g. for RMSD calculation)\n',
                     'write_multi_mol2 1\n',

                     '# binding site definition\n',
                     'bindingsite_center ' + str(pocket_definition['center'][0]) + ' ' + str(pocket_definition['center'][1]) + ' ' + str(pocket_definition['center'][2]) + '\n',
                     'bindingsite_radius ' + str(pocket_definition['size'][0] / 2) + '\n',

                     '# cluster algorithm\n',
                     'cluster_structures ' + str(n_poses) + '\n',
                     'cluster_rmsd 2.0\n',

                     '# write\n',
                     'write_ranking_links 0\n',
                     'write_protein_bindingsite 0\n',
                     'write_protein_conformations 0\n',
                     'write_protein_splitted 0\n',
                     'write_merged_protein 0\n',
                     '####\n']
    # Write config file
    printlog('Writing PLANTS config file...')
    with plants_docking_config_path.open('w') as configwriter:
        configwriter.writelines(plants_config)
    # Run PLANTS docking
    try:
        printlog('Starting PLANTS docking...')
        plants_docking_command = f'{software / "PLANTS"} --mode screen ' + str(plants_docking_config_path)
        subprocess.call(
            plants_docking_command,
            shell=True,
            stdout=DEVNULL,
            stderr=STDOUT)
    except Exception as e:
        printlog('ERROR: PLANTS docking command failed...')
        printlog(e)
    plants_docking_results_mol2 = plants_folder / 'results' / 'docked_ligands.mol2'
    plants_docking_results_sdf = plants_docking_results_mol2.with_suffix(
        '.sdf')
    # Convert PLANTS poses to sdf
    try:
        printlog('Converting PLANTS poses to .sdf format...')
        obabel_command = 'obabel -imol2 ' + \
            str(plants_docking_results_mol2) + ' -O ' + str(plants_docking_results_sdf)
        subprocess.call(
            obabel_command,
            shell=True,
            stdout=DEVNULL,
            stderr=STDOUT)
    except Exception as e:
        printlog('ERROR: Failed to convert PLANTS poses file to .sdf!')
        printlog(e)
    toc = time.perf_counter()
    printlog(f'Docking with PLANTS complete in {toc-tic:0.4f}!')
    plants_scoring_results = plants_folder / 'results' / 'ranking.csv'
    # Fetch PLANTS poses
    printlog('Fetching PLANTS poses...')
    try:
        plants_poses = PandasTools.LoadSDF(
            str(plants_docking_results_sdf),
            idName='ID',
            molColName='Molecule',
            includeFingerprints=False,
            embedProps=False,
            removeHs=False,
            strictParsing=True)
        plants_scores = pd.read_csv(
            str(plants_scoring_results), usecols=[
                'LIGAND_ENTRY', 'TOTAL_SCORE'])
        plants_scores = plants_scores.rename(
            columns={'LIGAND_ENTRY': 'ID', 'TOTAL_SCORE': 'CHEMPLP'})
        plants_scores = plants_scores[['ID', 'CHEMPLP']]
        plants_df = pd.merge(plants_scores, plants_poses, on='ID')
        plants_df['Pose ID'] = plants_df['ID'].str.split(
            '_').str[0] + '_PLANTS_' + plants_df['ID'].str.split('_').str[4]
        plants_df['ID'] = plants_df['ID'].str.split('_').str[0]
    except Exception as e:
        printlog('ERROR: Failed to Load PLANTS poses SDF file!')
        printlog(e)
    try:
        PandasTools.WriteSDF(plants_df,
                             str(plants_folder / 'plants_poses.sdf'),
                             molColName='Molecule',
                             idName='Pose ID',
                             properties=list(plants_df.columns))
        shutil.rmtree(plants_folder / 'results', ignore_errors=True)
        files = 'software/'.glob('*.pid')
        for file in files:
            file.unlink()
        toc = time.perf_counter()
        printlog(f'Cleaned up PLANTS poses in {toc-tic:0.4f}!')
    except Exception as e:
        printlog('ERROR: Failed to Write PLANTS poses SDF file!')
        printlog(e)
    return str(plants_folder / 'plants_poses.sdf')

def smina_docking_splitted(
        split_file,
        protein_file,
        pocket_definition,
        exhaustiveness,
        n_poses):
    w_dir = Path(protein_file).parent
    smina_folder = w_dir / 'temp' / 'smina'
    smina_folder.mkdir(parents=True, exist_ok=True)
    results_path = smina_folder / \
        f"{os.path.basename(split_file).split('.')[0]}_smina.sdf"
    smina_cmd = (
        f'{software / "gnina"}' +
        f' --receptor {protein_file}' +
        f' --ligand {split_file}' +
        f' --out {results_path}' +
        f' --center_x {pocket_definition["center"][0]}' +
        f' --center_y {pocket_definition["center"][1]}' +
        f' --center_z {pocket_definition["center"][2]}' +
        f' --size_x {pocket_definition["size"][0]}' +
        f' --size_y {pocket_definition["size"][1]}' +
        f' --size_z {pocket_definition["size"][2]}' +
        f' --exhaustiveness {exhaustiveness}' +
        ' --cpu 1' +
        f' --num_modes {n_poses}' +
        ' --cnn_scoring none --no_gpu'
    )
    try:
        subprocess.call(smina_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
    except Exception as e:
        printlog(f'SMINA docking failed: {e}')
    return

def gnina_docking_splitted(
        split_file,
        protein_file,
        pocket_definition,
        exhaustiveness,
        n_poses):
    w_dir = Path(protein_file).parent
    gnina_folder = w_dir / 'temp' / 'gnina'
    gnina_folder.mkdir(parents=True, exist_ok=True)
    results_path = gnina_folder / \
        f"{os.path.basename(split_file).split('.')[0]}_gnina.sdf"
    gnina_cmd = (
        f"{software / 'gnina'}" +
        f" --receptor {protein_file}" +
        f" --ligand {split_file}" +
        f" --out {results_path}" +
        f" --center_x {pocket_definition['center'][0]}" +
        f" --center_y {pocket_definition['center'][1]}" +
        f" --center_z {pocket_definition['center'][2]}" +
        f" --size_x {pocket_definition['size'][0]}" +
        f" --size_y {pocket_definition['size'][1]}" +
        f" --size_z {pocket_definition['size'][2]}" +
        f" --exhaustiveness {exhaustiveness}" +
        " --cpu 1" +
        f" --num_modes {n_poses}" +
        " --cnn_scoring rescore --cnn crossdock_default2018 --no_gpu"
    )

    try:
        subprocess.call(gnina_cmd, 
            shell=True,
            stdout=DEVNULL,
            stderr=STDOUT)
    except Exception as e:
        printlog(f"GNINA docking failed: {e}")
    return

def plants_docking_splitted(
        split_file,
        w_dir,
        n_poses,
        pocket_definition):
    plants_docking_results_dir = w_dir / 'temp' / \
        'plants' / ('results_' + split_file.stem)
    # Generate plants config file
    plants_docking_config_path = w_dir / 'temp' / \
        'plants' / ('config_' + split_file.stem + '.config')
    plants_config = ['# search algorithm\n',
                     'search_speed speed1\n',
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
                     'protein_file ' + str(w_dir / 'temp' / 'plants' / 'protein.mol2') + '\n',
                     'ligand_file ' + str(w_dir / 'temp' / 'plants' / os.path.basename(split_file).replace('.sdf', '.mol2')) + '\n',

                     '# output\n',
                     'output_dir ' + str(plants_docking_results_dir) + '\n',

                     '# write single mol2 files (e.g. for RMSD calculation)\n',
                     'write_multi_mol2 1\n',

                     '# binding site definition\n',
                     'bindingsite_center ' + str(pocket_definition['center'][0]) + ' ' + str(pocket_definition['center'][1]) + ' ' + str(pocket_definition['center'][2]) + '+\n',
                     'bindingsite_radius ' + str(pocket_definition['size'][0] / 2) + '\n',

                     '# cluster algorithm\n',
                     'cluster_structures ' + str(n_poses) + '\n',
                     'cluster_rmsd 2.0\n',

                     '# write\n',
                     'write_ranking_links 0\n',
                     'write_protein_bindingsite 0\n',
                     'write_protein_conformations 0\n',
                     'write_protein_splitted 0\n',
                     'write_merged_protein 0\n',
                     '####\n']
    # Write config file
    with open(plants_docking_config_path, 'w') as configwriter:
        configwriter.writelines(plants_config)
    # Run PLANTS docking
    try:
        plants_docking_command = f'{software / "PLANTS"} --mode screen ' + str(plants_docking_config_path)
        subprocess.call(
            plants_docking_command,
            stdout=DEVNULL,
            stderr=STDOUT)
    except Exception as e:
        printlog('ERROR: PLANTS docking command failed...')
        printlog(e)
    return

def qvinaw_docking_splitted(
        split_file,
        protein_file_pdbqt,
        pocket_definition,
        exhaustiveness,
        n_poses):
    w_dir = Path(protein_file_pdbqt).parent
    qvinaw_folder = w_dir / 'temp' / 'qvinaw'
    pdbqt_files_folder = qvinaw_folder / Path(split_file).stem / 'pdbqt_files'
    pdbqt_files_folder.mkdir(parents=True, exist_ok=True)
    results_path = qvinaw_folder / Path(split_file).stem / 'docked'
    results_path.mkdir(parents=True, exist_ok=True)

    try:
        meeko_to_pdbqt(str(split_file), str(pdbqt_files_folder))
    except Exception as e:
        print('Failed to convert sdf file to .pdbqt')
        print(e)

    pdbqt_files = list(pdbqt_files_folder.glob('*.pdbqt'))

    for pdbqt_file in pdbqt_files:
        qvina_cmd = (
            f"{software / 'qvina-w'}" +
            f" --receptor {protein_file_pdbqt}" +
            f" --ligand {pdbqt_file}" +
            f" --out {str(pdbqt_file).replace('pdbqt_files', 'docked')}" +
            f" --center_x {pocket_definition['center'][0]}" +
            f" --center_y {pocket_definition['center'][1]}" +
            f" --center_z {pocket_definition['center'][2]}" +
            f" --size_x {pocket_definition['size'][0]}" +
            f" --size_y {pocket_definition['size'][1]*2}" +
            f" --size_z {pocket_definition['size'][2]*2}" +
            f" --exhaustiveness {exhaustiveness}" +
            " --cpu 1" +
            f" --num_modes {n_poses}"
        )
        try:
            subprocess.call(
                qvina_cmd,
                shell=True,
                stdout=DEVNULL,
                stderr=STDOUT)
        except Exception as e:
            printlog('QVINAW docking failed: ' + e)

    qvinaw_docking_results = qvinaw_folder / \
        (Path(split_file).stem + '_qvinaw.sdf')

    results_pdbqt_files = list(results_path.glob('*.pdbqt'))

    for pdbqt_file in results_pdbqt_files:
        # Convert to sdf
        sdf_file = pdbqt_file.with_suffix('.sdf')
        obabel_command = f'obabel {pdbqt_file} -O {sdf_file}'
        try:
            subprocess.call(
                obabel_command,
                shell=True,
                stdout=DEVNULL,
                stderr=STDOUT)
        except Exception as e:
            print(f'Conversion from PDBQT to SDF failed: {e}')

    try:
        sdf_files = list(results_path.glob('*.sdf'))
        qvinaw_poses = []
        for sdf in sdf_files:
            df = PandasTools.LoadSDF(
                str(sdf),
                idName='ID',
                molColName='Molecule',
                includeFingerprints=False,
                embedProps=False,
                removeHs=False,
                strictParsing=False)
            list_ = [*range(1, int(n_poses) + 1, 1)]
            ser = list_ * (len(df) // len(list_))
            df['Pose ID'] = [
                f'{sdf.name.replace(".sdf", "")}_QVINAW_{num}' for num,
                (_,
                 row) in zip(
                    ser +
                    list_[
                        :len(df) -
                        len(ser)],
                    df.iterrows())]
            df = df.rename(columns={'REMARK': 'QVINAW_Affinity'})[
                ['Molecule', 'QVINAW_Affinity', 'Pose ID']]
            df['QVINAW_Affinity'] = df['QVINAW_Affinity'].str.split().str[2]
            qvinaw_poses.append(df)
        qvinaw_poses = pd.concat(qvinaw_poses)
        PandasTools.WriteSDF(
            qvinaw_poses,
            str(qvinaw_docking_results),
            molColName='Molecule',
            idName='Pose ID',
            properties=list(
                qvinaw_poses.columns))
    except Exception as e:
        printlog('ERROR: Failed to combine QVINAW SDF file!')
        printlog(e)
    else:
        shutil.rmtree(
            qvinaw_folder /
            Path(split_file).stem,
            ignore_errors=True)
    return qvinaw_docking_results

def qvina2_docking_splitted(
        split_file,
        protein_file_pdbqt,
        pocket_definition,
        exhaustiveness,
        n_poses):
    w_dir = Path(protein_file_pdbqt).parent
    qvina2_folder = w_dir / 'temp' / 'qvina2'
    pdbqt_files_folder = qvina2_folder / Path(split_file).stem / 'pdbqt_files'
    pdbqt_files_folder.mkdir(parents=True, exist_ok=True)
    results_path = qvina2_folder / Path(split_file).stem / 'docked'
    results_path.mkdir(parents=True, exist_ok=True)

    try:
        meeko_to_pdbqt(str(split_file), str(pdbqt_files_folder))
    except Exception as e:
        print('Failed to convert sdf file to .pdbqt')
        print(e)

    pdbqt_files = list(pdbqt_files_folder.glob('*.pdbqt'))

    for pdbqt_file in pdbqt_files:
        qvina_cmd = (
            f"{software / 'qvina2.1'}" +
            f" --receptor {protein_file_pdbqt}" +
            f" --ligand {pdbqt_file}" +
            f" --out {str(pdbqt_file).replace('pdbqt_files', 'docked')}" +
            f" --center_x {pocket_definition['center'][0]}" +
            f" --center_y {pocket_definition['center'][1]}" +
            f" --center_z {pocket_definition['center'][2]}" +
            f" --size_x {pocket_definition['size'][0]}" +
            f" --size_y {pocket_definition['size'][1]*2}" +
            f" --size_z {pocket_definition['size'][2]*2}" +
            f" --exhaustiveness {exhaustiveness}" +
            " --cpu 1" +
            f" --num_modes {n_poses}"
        )

        try:
            subprocess.call(
                qvina_cmd,
                shell=True,
                stdout=DEVNULL,
                stderr=STDOUT)
        except Exception as e:
            printlog('QVINA2 docking failed: ' + e)

    qvina2_docking_results = qvina2_folder / \
        (Path(split_file).stem + '_qvina2.sdf')

    results_pdbqt_files = list(results_path.glob('*.pdbqt'))

    for pdbqt_file in results_pdbqt_files:
        # Convert to sdf
        sdf_file = pdbqt_file.with_suffix('.sdf')
        obabel_command = f'obabel {pdbqt_file} -O {sdf_file}'
        try:
            subprocess.call(
                obabel_command,
                shell=True,
                stdout=DEVNULL,
                stderr=STDOUT)
        except Exception as e:
            print(f'Conversion from PDBQT to SDF failed: {e}')

    try:
        sdf_files = list(results_path.glob('*.sdf'))
        qvina2_poses = []
        for sdf in sdf_files:
            df = PandasTools.LoadSDF(
                str(sdf),
                idName='ID',
                molColName='Molecule',
                includeFingerprints=False,
                embedProps=False,
                removeHs=False,
                strictParsing=False)
            list_ = [*range(1, int(n_poses) + 1, 1)]
            ser = list_ * (len(df) // len(list_))
            df['Pose ID'] = [
                f'{sdf.name.replace(".sdf", "")}_QVINA2_{num}' for num,
                (_,
                 row) in zip(
                    ser +
                    list_[
                        :len(df) -
                        len(ser)],
                    df.iterrows())]
            df = df.rename(columns={'REMARK': 'QVINA2_Affinity'})[
                ['Molecule', 'QVINA2_Affinity', 'Pose ID']]
            df['QVINA2_Affinity'] = df['QVINA2_Affinity'].str.split().str[2]
            qvina2_poses.append(df)
        qvina2_poses = pd.concat(qvina2_poses)
        PandasTools.WriteSDF(
            qvina2_poses,
            str(qvina2_docking_results),
            molColName='Molecule',
            idName='Pose ID',
            properties=list(
                qvina2_poses.columns))
    except Exception as e:
        printlog('ERROR: Failed to combine QVINA2 SDF file!')
        printlog(e)
    else:
        shutil.rmtree(
            qvina2_folder /
            Path(split_file).stem,
            ignore_errors=True)
    return qvina2_docking_results

def docking(
        w_dir,
        protein_file,
        pocket_definition,

        docking_programs,
        exhaustiveness,
        n_poses,
        ncpus):
    if ncpus == 1:
        tic = time.perf_counter()
        if 'SMINA' in docking_programs and (
                w_dir / 'temp' / 'smina').is_dir() == False:
            smina_docking(
                protein_file,
                pocket_definition,

                exhaustiveness,
                n_poses)
        if 'GNINA' in docking_programs and (
                w_dir / 'temp' / 'gnina').is_dir() == False:
            gnina_docking(
                protein_file,
                pocket_definition,

                exhaustiveness,
                n_poses)
        if 'PLANTS' in docking_programs and (
                w_dir / 'temp' / 'plants').is_dir() == False:
            plants_docking(protein_file, pocket_definition, n_poses)
        if 'QVINAW' in docking_programs and (
                w_dir / 'temp' / 'qvinaw').is_dir() == False:
            qvinaw_docking(
                protein_file,
                pocket_definition,

                exhaustiveness,
                n_poses)
        if 'QVINA2' in docking_programs and (
                w_dir / 'temp' / 'qvina2').is_dir() == False:
            qvina2_docking(
                protein_file,
                pocket_definition,

                exhaustiveness,
                n_poses)
        toc = time.perf_counter()
        printlog(f'Finished docking in {toc-tic:0.4f}!')
        
    else:

        split_final_library_path = w_dir / 'temp' / 'split_final_library'
        if not split_final_library_path.is_dir():
            split_files_folder = split_sdf(
                str(w_dir / 'temp'), str(w_dir / 'temp' / 'final_library.sdf'), ncpus)
        else:
            printlog('Split final library folder already exists...')
            split_files_folder = split_final_library_path

        split_files_sdfs = [(split_files_folder / f)
                            for f in os.listdir(split_files_folder) if f.endswith('.sdf')]
        if 'PLANTS' in docking_programs and not (
                w_dir / 'temp' / 'plants').is_dir():
            tic = time.perf_counter()
            plants_folder = w_dir / 'temp' / 'plants'
            plants_folder.mkdir(parents=True, exist_ok=True)
            # Convert protein file to .mol2 using open babel
            plants_protein_mol2 = plants_folder / 'protein.mol2'
            try:
                printlog(
                    'Converting protein file to .mol2 format for PLANTS docking...')
                obabel_command = f'obabel -ipdb {protein_file} -O {plants_protein_mol2}'
                subprocess.call(
                    obabel_command,
                    shell=True,
                    stdout=DEVNULL,
                    stderr=STDOUT)
            except Exception as e:
                printlog('ERROR: Failed to convert protein file to .mol2!')
                printlog(e)
            # Convert prepared ligand file to .mol2 using open babel
            for file in os.listdir(split_files_folder):
                if file.endswith('.sdf'):
                    try:
                        obabel_command = f'obabel -isdf {split_files_folder}/{file} -O {w_dir / "temp" / "plants"}/{Path(file).stem}.mol2'
                        subprocess.call(
                            obabel_command,
                            shell=True)
                    except Exception as e:
                        printlog(f'ERROR: Failed to convert {file} to .mol2!')
                        printlog(e)
            printlog('Docking split files using PLANTS...')
            with concurrent.futures.ProcessPoolExecutor(max_workers=ncpus) as executor:
                jobs = []
                for split_file in tqdm(
                        split_files_sdfs,
                        desc='Submitting PLANTS jobs',
                        unit='Jobs'):
                    try:
                        job = executor.submit(
                            plants_docking_splitted,
                            split_file,
                            w_dir,

                            n_poses,
                            pocket_definition)
                        jobs.append(job)
                    except Exception as e:
                        printlog(
                            'Error in concurrent futures job creation: ' + str(e))
                for job in tqdm(
                        concurrent.futures.as_completed(jobs),
                        desc='Docking with PLANTS',
                        total=len(jobs)):
                    try:
                        _ = job.result()
                    except Exception as e:
                        printlog(
                            'Error in concurrent futures job execution: ' + str(e))
            toc = time.perf_counter()
            printlog(f'Docking with PLANTS complete in {toc - tic:0.4f}!')
        # Fetch PLANTS poses
        if 'PLANTS' in docking_programs and (
                w_dir /
                'temp' /
                'plants').is_dir() and not (
                w_dir /
                'temp' /
                'plants' /
                'plants_poses.sdf').is_file():
            plants_dataframes = []
            results_folders = [
                item for item in os.listdir(
                    w_dir / 'temp' / 'plants')]
            for item in tqdm(
                    results_folders,
                    desc='Fetching PLANTS docking poses'):
                if item.startswith('results'):
                    file_path = w_dir / 'temp' / 'plants' / item / 'docked_ligands.mol2'
                    if file_path.is_file():
                        try:
                            obabel_command = f'obabel -imol2 {file_path} -O {file_path.with_suffix(".sdf")}'
                            subprocess.call(
                                obabel_command, shell=True, stdout=DEVNULL, stderr=STDOUT)
                            plants_poses = PandasTools.LoadSDF(
                                str(
                                    file_path.with_suffix('.sdf')),
                                idName='ID',
                                molColName='Molecule',
                                includeFingerprints=False,
                                embedProps=False,
                                removeHs=False,
                                strictParsing=True)
                            plants_scores = pd.read_csv(str(file_path).replace('docked_ligands.mol2', 'ranking.csv')).rename(
                                columns={'LIGAND_ENTRY': 'ID', 'TOTAL_SCORE': 'CHEMPLP'})[['ID', 'CHEMPLP']]
                            plants_df = pd.merge(
                                plants_scores, plants_poses, on='ID')
                            plants_df['ID'] = plants_df['ID'].str.split(
                                '_').str[0]
                            list_ = [*range(1, int(n_poses) + 1, 1)]
                            ser = list_ * (len(plants_df) // len(list_))
                            plants_df['Pose ID'] = [f'{row["ID"]}_PLANTS_{num}' for num, (_, row) in zip(
                                ser + list_[:len(plants_df) - len(ser)], plants_df.iterrows())]
                            plants_dataframes.append(plants_df)
                        except Exception as e:
                            printlog(
                                'ERROR: Failed to convert PLANTS docking results file to .sdf!')
                            printlog(e)
                elif item in ['protein.mol2', 'ref.mol2']:
                    pass
                else:
                    Path(
                        w_dir /
                        'temp' /
                        'plants',
                        item).unlink(
                        missing_ok=True)
            try:
                plants_df = pd.concat(plants_dataframes)
                PandasTools.WriteSDF(plants_df,
                                     str(w_dir / 'temp' / 'plants' / 'plants_poses.sdf'),
                                     molColName='Molecule',
                                     idName='Pose ID',
                                     properties=list(plants_df.columns))
                files = 'software/'.glob('*.pid')
                for file in files:
                    file.unlink()
            except Exception as e:
                printlog('ERROR: Failed to write combined PLANTS docking poses')
                printlog(e)
            else:
                for file in os.listdir(w_dir / 'temp' / 'plants'):
                    if not file.startswith('plants_poses'):
                        shutil.rmtree(w_dir / 'temp' / 'plants' / file)
        # Docking split files using SMINA
        if 'SMINA' in docking_programs and not (
                w_dir / 'temp' / 'smina').is_dir():
            printlog('Docking split files using SMINA...')
            tic = time.perf_counter()
            with concurrent.futures.ProcessPoolExecutor(max_workers=ncpus) as executor:
                jobs = []
                for split_file in tqdm(
                        split_files_sdfs,
                        desc='Submitting SMINA jobs',
                        unit='Jobs'):
                    try:
                        job = executor.submit(
                            smina_docking_splitted,
                            split_file,
                            protein_file,
                            pocket_definition,
                            exhaustiveness,
                            n_poses)
                        jobs.append(job)
                    except Exception as e:
                        printlog(
                            'Error in concurrent futures job creation: ' + str(e))
                for job in tqdm(
                        concurrent.futures.as_completed(jobs),
                        desc='Docking with SMINA',
                        total=len(jobs)):
                    try:
                        _ = job.result()
                    except Exception as e:
                        printlog(
                            'Error in concurrent futures job execution: ' + str(e))
            toc = time.perf_counter()
            printlog(f'Docking with SMINA complete in {toc - tic:0.4f}!')
        # Fetch SMINA poses
        if 'SMINA' in docking_programs and (
                w_dir /
                'temp' /
                'smina').is_dir() and not (
                w_dir /
                'temp' /
                'smina' /
                'smina_poses.sdf').is_file():
            try:
                smina_dataframes = []
                for file in tqdm(
                        os.listdir(
                            w_dir / 'temp' / 'smina'),
                        desc='Loading SMINA poses'):
                    if file.startswith('split'):
                        df = PandasTools.LoadSDF(
                            str(
                                w_dir / 'temp' / 'smina' / file),
                            idName='ID',
                            molColName='Molecule',
                            includeFingerprints=False,
                            embedProps=False,
                            removeHs=False,
                            strictParsing=True)
                        smina_dataframes.append(df)
                smina_df = pd.concat(smina_dataframes)
                list_ = [*range(1, int(n_poses) + 1, 1)]
                ser = list_ * (len(smina_df) // len(list_))
                smina_df['Pose ID'] = [
                    f'{row["ID"]}_SMINA_{num}' for num,
                    (_,
                     row) in zip(
                        ser +
                        list_[
                            :len(smina_df) -
                            len(ser)],
                        smina_df.iterrows())]
                smina_df.rename(
                    columns={
                        'minimizedAffinity': 'SMINA_Affinity'},
                    inplace=True)
            except Exception as e:
                printlog('ERROR: Failed to Load SMINA poses SDF file!')
                printlog(e)
            try:
                PandasTools.WriteSDF(smina_df,
                                     str(w_dir / 'temp' / 'smina' / 'smina_poses.sdf'),
                                     molColName='Molecule',
                                     idName='Pose ID',
                                     properties=list(smina_df.columns))
            except Exception as e:
                printlog('ERROR: Failed to write combined SMINA poses SDF file!')
                printlog(e)
            else:
                for file in os.listdir(w_dir / 'temp' / 'smina'):
                    if file.startswith('split'):
                        os.remove(w_dir / 'temp' / 'smina' / file)
        # Docking split files using GNINA
        if 'GNINA' in docking_programs and not (
                w_dir / 'temp' / 'gnina').is_dir():
            printlog('Docking split files using GNINA...')
            tic = time.perf_counter()
            with concurrent.futures.ProcessPoolExecutor(max_workers=ncpus) as executor:
                jobs = []
                for split_file in tqdm(
                        split_files_sdfs,
                        desc='Submitting GNINA jobs',
                        unit='Jobs'):
                    try:
                        job = executor.submit(
                            gnina_docking_splitted,
                            split_file,
                            protein_file,
                            pocket_definition,
                            exhaustiveness,
                            n_poses)
                        jobs.append(job)
                    except Exception as e:
                        printlog(
                            'Error in concurrent futures job creation: ' + str(e))
                for job in tqdm(
                        concurrent.futures.as_completed(jobs),
                        desc='Docking with GNINA',
                        total=len(jobs)):
                    try:
                        _ = job.result()
                    except Exception as e:
                        printlog(
                            'Error in concurrent futures job execution: ' + str(e))
            toc = time.perf_counter()
            printlog(f'Docking with GNINA complete in {toc - tic:0.4f}!')
        # Fetch GNINA poses
        if 'GNINA' in docking_programs and (
                w_dir /
                'temp' /
                'gnina').is_dir() and not (
                w_dir /
                'temp' /
                'gnina' /
                'gnina_poses.sdf').is_file():
            try:
                gnina_dataframes = []
                for file in tqdm(
                        os.listdir(
                            w_dir / 'temp' / 'gnina'),
                        desc='Loading GNINA poses'):
                    if file.startswith('split'):
                        df = PandasTools.LoadSDF(
                            str(
                                w_dir / 'temp' / 'gnina' / file),
                            idName='ID',
                            molColName='Molecule',
                            includeFingerprints=False,
                            embedProps=False,
                            removeHs=False,
                            strictParsing=True)
                        gnina_dataframes.append(df)
                gnina_df = pd.concat(gnina_dataframes)
                list_ = [*range(1, int(n_poses) + 1, 1)]
                ser = list_ * (len(gnina_df) // len(list_))
                gnina_df['Pose ID'] = [
                    f'{row["ID"]}_GNINA_{num}' for num,
                    (_,
                     row) in zip(
                        ser +
                        list_[
                            :len(gnina_df) -
                            len(ser)],
                        gnina_df.iterrows())]
                gnina_df.rename(
                    columns={
                        'minimizedAffinity': 'GNINA_Affinity'},
                    inplace=True)
            except Exception as e:
                printlog('ERROR: Failed to Load GNINA poses SDF file!')
                printlog(e)
            try:
                PandasTools.WriteSDF(gnina_df,
                                     str(w_dir / 'temp' / 'gnina' / 'gnina_poses.sdf'),
                                     molColName='Molecule',
                                     idName='Pose ID',
                                     properties=list(gnina_df.columns))
            except Exception as e:
                printlog('ERROR: Failed to write combined GNINA docking poses')
                printlog(e)
            else:
                for file in os.listdir(w_dir / 'temp' / 'gnina'):
                    if file.startswith('split'):
                        os.remove(w_dir / 'temp' / 'gnina' / file)
        # Docking split files using QVINAW
        if 'QVINAW' in docking_programs and not (
                w_dir / 'temp' / 'qvinaw').is_dir():
            printlog('Docking split files using QVINAW...')
            tic = time.perf_counter()
            protein_file_pdbqt = convert_pdb_to_pdbqt(protein_file)
            with concurrent.futures.ProcessPoolExecutor(max_workers=ncpus) as executor:
                jobs = []
                for split_file in tqdm(
                        split_files_sdfs,
                        desc='Submitting QVINAW jobs',
                        unit='Jobs'):
                    try:
                        job = executor.submit(
                            qvinaw_docking_splitted,
                            split_file,
                            protein_file_pdbqt,
                            pocket_definition,
                            exhaustiveness,
                            n_poses)
                        jobs.append(job)
                    except Exception as e:
                        printlog(
                            'Error in concurrent futures job creation: ' + str(e))
                for job in tqdm(
                        concurrent.futures.as_completed(jobs),
                        desc='Docking with QVINAW',
                        total=len(jobs)):
                    try:
                        _ = job.result()
                    except Exception as e:
                        printlog(
                            'Error in concurrent futures job execution: ' + str(e))
            toc = time.perf_counter()
            printlog(f'Docking with QVINAW complete in {toc - tic:0.4f}!')
        # Fetch QVINAW poses
        if 'QVINAW' in docking_programs and (
                w_dir /
                'temp' /
                'qvinaw').is_dir() and not (
                w_dir /
                'temp' /
                'qvinaw' /
                'qvinaw_poses.sdf').is_file():
            try:
                qvinaw_dataframes = []
                for file in tqdm(
                        os.listdir(
                            w_dir / 'temp' / 'qvinaw'),
                        desc='Loading QVINAW poses'):
                    if file.startswith('split'):
                        df = PandasTools.LoadSDF(
                            str(
                                w_dir / 'temp' / 'qvinaw' / file),
                            idName='Pose ID',
                            molColName='Molecule',
                            includeFingerprints=False,
                            embedProps=False,
                            removeHs=False,
                            strictParsing=True)
                        qvinaw_dataframes.append(df)
                qvinaw_df = pd.concat(qvinaw_dataframes)
            except Exception as e:
                printlog('ERROR: Failed to Load QVINAW poses SDF file!')
                printlog(e)
            try:
                PandasTools.WriteSDF(qvinaw_df,
                                     str(w_dir / 'temp' / 'qvinaw' / 'qvinaw_poses.sdf'),
                                     molColName='Molecule',
                                     idName='Pose ID',
                                     properties=list(qvinaw_df.columns))
            except Exception as e:
                printlog('ERROR: Failed to write combined QVINAW poses SDF file!')
                printlog(e)
            else:
                for file in os.listdir(w_dir / 'temp' / 'qvinaw'):
                    if file.startswith('split'):
                        os.remove(w_dir / 'temp' / 'qvinaw' / file)
        # Docking split files using QVINA2
        if 'QVINA2' in docking_programs and not (
                w_dir / 'temp' / 'qvina2').is_dir():
            printlog('Docking split files using QVINA2...')
            tic = time.perf_counter()
            protein_file_pdbqt = convert_pdb_to_pdbqt(protein_file)
            with concurrent.futures.ProcessPoolExecutor(max_workers=ncpus) as executor:
                jobs = []
                for split_file in tqdm(
                        split_files_sdfs,
                        desc='Submitting QVINA2 jobs',
                        unit='Jobs'):
                    try:
                        job = executor.submit(
                            qvina2_docking_splitted,
                            split_file,
                            protein_file_pdbqt,
                            pocket_definition,
                            exhaustiveness,
                            n_poses)
                        jobs.append(job)
                    except Exception as e:
                        printlog(
                            'Error in concurrent futures job creation: ' + str(e))
                for job in tqdm(
                        concurrent.futures.as_completed(jobs),
                        desc='Docking with QVINA2',
                        total=len(jobs)):
                    try:
                        _ = job.result()
                    except Exception as e:
                        printlog(
                            'Error in concurrent futures job execution: ' + str(e))
            toc = time.perf_counter()
            printlog(f'Docking with QVINA2 complete in {toc - tic:0.4f}!')
        # Fetch QVINA2 poses
        if 'QVINA2' in docking_programs and (
                w_dir /
                'temp' /
                'qvina2').is_dir() and not (
                w_dir /
                'temp' /
                'qvina2' /
                'qvina2_poses.sdf').is_file():
            try:
                qvina2_dataframes = []
                for file in tqdm(
                        os.listdir(
                            w_dir / 'temp' / 'qvina2'),
                        desc='Loading QVINA2 poses'):
                    if file.startswith('split'):
                        df = PandasTools.LoadSDF(
                            str(
                                w_dir / 'temp' / 'qvina2' / file),
                            idName='Pose ID',
                            molColName='Molecule',
                            includeFingerprints=False,
                            embedProps=False,
                            removeHs=False,
                            strictParsing=True)
                        qvina2_dataframes.append(df)
                qvina2_df = pd.concat(qvina2_dataframes)
            except Exception as e:
                printlog('ERROR: Failed to Load QVINA2 poses SDF file!')
                printlog(e)
            try:
                PandasTools.WriteSDF(qvina2_df,
                                     str(w_dir / 'temp' / 'qvina2' / 'qvina2_poses.sdf'),
                                     molColName='Molecule',
                                     idName='Pose ID',
                                     properties=list(qvina2_df.columns))
            except Exception as e:
                printlog('ERROR: Failed to write combined QVINA2 poses SDF file!')
                printlog(e)
            else:
                for file in os.listdir(w_dir / 'temp' / 'qvina2'):
                    if file.startswith('split'):
                        os.remove(w_dir / 'temp' / 'qvina2' / file)
    return

def concat_all_poses(w_dir, docking_programs):
        all_poses = pd.DataFrame()
        for program in docking_programs:
            try:
                df = PandasTools.LoadSDF(
                    f"{w_dir}/temp/{program.lower()}/{program.lower()}_poses.sdf",
                    idName='Pose ID',
                    molColName='Molecule',
                    includeFingerprints=False,
                    embedProps=False,
                    removeHs=False,
                    strictParsing=True)
                all_poses = pd.concat([all_poses, df])
            except Exception as e:
                printlog(f'ERROR: Failed to write {program} SDF file!')
                printlog(e)
        try:
            PandasTools.WriteSDF(all_poses,
                                    f"{w_dir}/temp/allposes.sdf",
                                    molColName='Molecule',
                                    idName='Pose ID',
                                    properties=list(all_poses.columns))
            printlog('All poses succesfully combined!')
        except Exception as e:
            printlog('ERROR: Failed to write all_poses SDF file!')
            printlog(e)
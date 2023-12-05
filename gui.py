# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import os
import subprocess
from subprocess import DEVNULL, STDOUT, PIPE
import sys
import shlex
import pathlib
from contextlib import contextmanager, redirect_stdout
from io import StringIO

from scripts.clustering_metrics import CLUSTERING_METRICS
from scripts.docking_functions import DOCKING_PROGRAMS
from scripts.rescoring_functions import RESCORING_FUNCTIONS

st.set_page_config(page_title="DockM8", page_icon=":8ball:", layout="wide")
# Sidebar
st.sidebar.image(image='./media/Logo_white-cropped.svg', width=50)
st.sidebar.title("DockM8")
st.sidebar.subheader("Open-source consensus docking for everyone")
st.sidebar.link_button("Gitlab", 
                       url='https://gitlab.com/Tonylac77/DockM8.git')
st.sidebar.link_button("Visit Website", 
                       url='https://tonylac77.gitlab.io/dockm8-web/')
st.sidebar.link_button("Publication", 
                       url='https://doi.org/your-doi')

# Logo
st.columns(3)[1].image(image='./media/DockM8_white_vertical-cropped.svg', 
                       width=400)

# Setup
CWD = os.getcwd()
st.header("Setup", divider='orange')
mode = st.selectbox(label = 'Which mode do you want to run DockM8 in?',
                    key = 0,
                    options = ('Single', 'Ensemble'),
                    help='Single Docking: DockM8 will dock each ligand in the library to a single receptor. '+
                    'Ensemble Docking: DockM8 will dock each ligand in the library to all specified receptors and then combine the results to create an ensemble consensus.'
)
if mode != 'Single':
    threshold = st.slider(label='Threshold for ensemble consensus (in %)', 
                          min_value=0.0, 
                          max_value=10.0, 
                          step=0.1, 
                          help = 'Threshold for ensemble consensus (in %). DockM8 will only consider a ligand as a consensus hit if it is a top scorer for all the receptors.')
num_cpus = st.slider('Number of CPUs', 
                     min_value=1, 
                     step=1, 
                     help='Number of CPUs to use for calculations')
software = st.text_input("Choose a software directory", 
                         value=os.getcwd() + '/software', 
                         help='Type the directory containing the software folder: For example: /home/user/Dockm8/software')

col1, col2 = st.columns(2)

# Receptor(s)
col1.header("Receptor(s)", divider='orange')
receptor_file = col1.text_input(label="File path(s) of one or more multiple receptor files (.pdb format), separated by commas", 
                                help = 'Choose one or multiple receptor files (.pdb format)',
                                value = CWD+'/testing_single_docking/protein.pdb', 
                                placeholder= 'Enter path(s) here')

# Prepare receptor
prepare_receptor = col1.toggle(label="Prepare receptor using Protoss", 
                               value=True, 
                               help='Choose whether or not to use Protoss Web service to protonate the protein structure')

# Pocket finding
col1.subheader("Pocket finding", divider='orange')
pocket_mode = col1.selectbox(label='How should the pocket be defined?', 
                             options=('Reference', 'RoG', 'Dogsitescorer'), 
                             help='Reference Ligand: DockM8 will use the reference ligand to define the pocket. '+
                             'Reference Ligand RoG: DockM8 will use the reference ligand radius of gyration. '+
                             'DogSiteScorer: DockM8 will use the DogSiteScorer pocket finding algorithm to define the pocket.')

# Reference ligand
if pocket_mode != 'Dogsitescorer':
    reference_file = col1.text_input(label="File path(s) of one or more multiple reference ligand files (.sdf format), separated by commas", 
                                     help = 'Choose one or multiple reference ligand files (.pdb format)', 
                                     value = CWD+'/testing_single_docking/ref.sdf', 
                                     placeholder= 'Enter path(s) here')

# Ligand library
col2.header("Ligands", divider='orange')
ligand_file = col2.text_input(label="Entre the path to the ligand library file (.sdf format)", 
                                value = CWD+'/testing_single_docking/library.sdf', 
                                help='Choose a ligand library file (.sdf format)')

# ID column
id_column = col2.text_input(label="Choose the column name that contains the ID of the ligand", 
                            value='ID', 
                            help='Choose the column name that contains the ID of the ligand')

# Ligand conformers
col2.subheader("Ligand conformers", divider='orange')
ligand_conformers = col2.selectbox(label='How should the ligands be prepared?',
                                   options=['MMFF', 'GypsumDL'],
                                   help='MMFF: DockM8 will use MMFF to prepare the ligand 3D conformers. '+
                                   'GypsumDL: DockM8 will use Gypsum-DL to prepare the ligand 3D conformers.')

# Ligand protonation
col2.subheader("Ligand protonation", divider='orange')
ligand_protonation = col2.selectbox(label='How should the ligands be protonated?',
                                    options=('None', 'GypsumDL'), 
                                    help='None: No protonation '+
                                    'Gypsum-DL: DockM8 will use  Gypsum-DL to protonate the ligands')

# Docking programs
st.header("Docking programs", divider='orange')
docking_programs = st.multiselect(label = "Choose the docking programs you want to use", 
                                  options=DOCKING_PROGRAMS, 
                                  help='Choose the docking programs you want to use, mutliple selection is allowed')

# Number of poses
nposes = st.slider(label='Number of poses', 
                   min_value=1, 
                   max_value=100, 
                   step=5, 
                   value=10, 
                   help='Number of poses to generate for each ligand')

# Exhaustiveness
exhaustiveness = st.select_slider(label='Exhaustiveness', 
                                  options=[1, 2, 4, 8, 16, 32], 
                                  value=8, 
                                  help='Exhaustiveness of the docking, only applies to GNINA, SMINA, QVINA2 and QVINAW')

# Pose selection
st.header("Pose Selection", divider='orange')
pose_selection = st.multiselect(label="Choose the pose selection method you want to use",
                                options = list(CLUSTERING_METRICS.keys()) + 
                                ['bestpose', 'bestpose_GNINA', 'bestpose_SMINA', 'bestpose_PLANTS', 'bestpose_QVINA2', 'bestpose_QVINAW'] +
                                list(RESCORING_FUNCTIONS.keys()), 
                                help='The method(s) to use for pose clustering. Must be one or more of:\n'+
                                        '- RMSD : Cluster compounds on RMSD matrix of poses \n'+
                                        '- spyRMSD : Cluster compounds on symmetry-corrected RMSD matrix of poses\n'+
                                        '- espsim : Cluster compounds on electrostatic shape similarity matrix of poses\n'+
                                        '- USRCAT : Cluster compounds on shape similarity matrix of poses\n'+
                                        '- 3DScore : Selects pose with the lowest average RMSD to all other poses\n'+
                                        '- bestpose : Takes the best pose from each docking program\n'+
                                        '- bestpose_GNINA : Takes the best pose from GNINA docking program\n'+
                                        '- bestpose_SMINA : Takes the best pose from SMINA docking program\n'+
                                        '- bestpose_QVINAW : Takes the best pose from QVINAW docking program\n'+
                                        '- bestpose_QVINA2 : Takes the best pose from QVINA2 docking program\n'+
                                        '- bestpose_PLANTS : Takes the best pose from PLANTS docking program  \n'+
                                        '- You can also use any of the scoring functions and DockM8 will select the best pose for each compound according to the specified scoring function.')
# Clustering algorithm
if any(x in CLUSTERING_METRICS.keys() for x in pose_selection):
    clustering_algorithm = st.selectbox(label = 'Which clustering algorithm do you want to use?',
                                        options = ('KMedoids', 'Aff_Prop'), 
                                        index=0, 
                                        help='Which algorithm to use for clustering. Must be one of "KMedoids", "Aff_prop". Must be set when using "RMSD", "spyRMSD", "espsim", "USRCAT" clustering metrics.')
else:
    clustering_algorithm = None

# Rescoring
st.header("Scoring functions", divider='orange')
rescoring = st.multiselect(label = "Choose the scoring functions you want to use",
                                options=list(RESCORING_FUNCTIONS.keys()), 
                                help = 'The method(s) to use for scoring. Multiple selection allowed')

# Consensus
st.header("Consensus", divider='orange')
consensus_method = st.selectbox(label = 'Choose which consensus algorithm to use: ',
                                options = ('ECR_best', 'ECR_avg', 'avg_ECR', 'RbR', 'RbV', 'Zscore_best', 'Zscore_avg'), 
                                help = 'The method to use for consensus.')

# Define the common arguments
command = (f'{sys.executable} {CWD}/dockm8.py 'f'--software {software} '
                                                    f'--receptor {receptor_file} '
                                                    f'--reffile {reference_file} '
                                                    f'--pocket {pocket_mode} '
                                                    f'--docking_library {ligand_file} '
                                                    f'--idcolumn {id_column} '
                                                    f'--prepare_proteins {prepare_receptor} '
                                                    f'--protonation {ligand_protonation} '
                                                    f'--docking_programs {" ".join(docking_programs)} '
                                                    f'--clustering_metric {" ".join(pose_selection)} '
                                                    f'--nposes {nposes} '
                                                    f'--exhaustiveness {exhaustiveness} '
                                                    f'--ncpus {num_cpus} '
                                                    f'--clustering_method {clustering_algorithm} '
                                                    f'--rescoring {" ".join(rescoring)} '
                                                    f'--consensus {consensus_method}'
                                                    )
# Add mode-specific arguments
if mode == 'ensemble' or mode == 'active_learning':
    command += f' --mode {mode} --threshold {threshold}'
else:
    command += f' --mode {mode}'
    
command_list = shlex.split(command)

command

@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret
        
        stdout.write = new_write
        yield

output = st.empty()
if st.button('Run DockM8'):
    with st_capture(output.code):
        print("Hello")
        process = subprocess.Popen(command_list, text=True, bufsize=1, universal_newlines=True)
    
# def run_and_display_stdout(cmd_with_args):
#     result = subprocess.Popen(cmd_with_args, stdout=PIPE)
#     output_lines = result.stdout.readlines()
#     print(output_lines)
#     for line in output_lines:
#         st.text(line.strip())
        
# if st.button('Run DockM8'):
#     run_and_display_stdout(command_list)

# import select
    
# def run_and_display_stdout(cmd_with_args):
#     process = subprocess.Popen(cmd_with_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)

#     while process.poll() is None:
#         # Check for new output on stdout and stderr
#         readable, _, _ = select.select([process.stdout, process.stderr], [], [], 0.1)

#         for stream in readable:
#             line = stream.readline()
#             if line:
#                 st.text(line.strip())

#     # Read any remaining output after the process has completed
#     for line in process.stdout:
#         st.text(line.strip())

# if st.button('Run DockM8'):
#     run_and_display_stdout(command_list)

# import io
    
# def run_and_display_stdout(cmd_with_args):
#     process = subprocess.Popen(cmd_with_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)

#     # Create an iterator to read lines from the subprocess stdout
#     stdout_iterator = iter(io.TextIOWrapper(process.stdout, line_buffering=True))

#     while process.poll() is None or stdout_iterator:
#         try:
#             line = next(stdout_iterator)
#             st.text(line.strip())
#         except StopIteration:
#             # The iterator is exhausted
#             stdout_iterator = None

#     # Read any remaining output after the process has completed
#     for line in process.stdout:
#         st.text(line.strip())

# if st.button('Run DockM8'):
#     run_and_display_stdout(command_list)
    
# output = st.empty()
# with st_capture(output.code):
#     if st.button('Run DockM8'):
#         print('Running DockM8...')
#         # Run the command
#         command_list = shlex.split(command)
#         subprocess.run(command_list, )



# with st_capture(st.write):
#     st.write('test')
#     st.button("Run DockM8", on_click = run_command, kwargs= {'software' : software,
#                                                             'mode' : mode,
#                                                             'receptor' : receptor_file,
#                                                             'pocket' : pocket_mode,
#                                                             'ref' : reference_file,
#                                                             'docking_library' : ligand_file,
#                                                             'idcolumn' : id_column,
#                                                             'prepare_protein' : prepare_receptor,
#                                                             'protonation' : ligand_protonation,
#                                                             'docking_programs' : docking_programs,
#                                                             'clustering_metrics' : pose_selection,
#                                                             'nposes' : nposes,
#                                                             'exhaustiveness' : exhaustiveness,
#                                                             'ncpus' : num_cpus,
#                                                             'clustering' : clustering_algorithm,
#                                                             'rescoring' : rescoring,
#                                                             'consensus' : consensus_method,
#                                                             'threshold' : threshold})

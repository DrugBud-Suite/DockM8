import os
import subprocess
import requests
import tarfile
import zipfile
import shutil
import stat
from pathlib import Path
import sys

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.logging import printlog


def download_file(url, filename):
	"""
	Downloads a file from the given URL and saves it with the specified filename.

	Args:
		url (str): The URL of the file to download.
		filename (str): The name of the file to save.

	Returns:
		None
	"""
	response = requests.get(url, stream=True)
	with open(filename, 'wb') as file:
		for chunk in response.iter_content(chunk_size=8192):
			file.write(chunk)


def remove_git_files(path):
	"""
	Removes Git-related files and directories from the specified path.

	Args:
		path (str): The path to the directory from which Git files and directories should be removed.

	Returns:
		None
	"""
	git_files = ['.git', '.gitignore', '.gitmodules']
	for item in git_files:
		item_path = os.path.join(path, item)
		if os.path.isdir(item_path):
			shutil.rmtree(item_path)
		elif os.path.isfile(item_path):
			os.remove(item_path)

	# Recursively remove .git directories from submodules
	for root, dirs, files in os.walk(path):
		if '.git' in dirs:
			shutil.rmtree(os.path.join(root, '.git'))
			dirs.remove('.git')


def install_gnina(software_path):
	gnina_path = os.path.join(software_path, 'gnina')
	url = "https://github.com/gnina/gnina/releases/latest/download/gnina"
	download_file(url, gnina_path)
	os.chmod(gnina_path, os.stat(gnina_path).st_mode | stat.S_IEXEC)


def install_qvina_w(software_path):
	qvina_w_path = os.path.join(software_path, 'qvina-w')
	url = "https://github.com/QVina/qvina/raw/master/bin/qvina-w"
	download_file(url, qvina_w_path)
	os.chmod(qvina_w_path, os.stat(qvina_w_path).st_mode | stat.S_IEXEC)


def install_qvina2(software_path):
	qvina2_path = os.path.join(software_path, 'qvina2.1')
	url = "https://github.com/QVina/qvina/raw/master/bin/qvina2.1"
	download_file(url, qvina2_path)
	os.chmod(qvina2_path, os.stat(qvina2_path).st_mode | stat.S_IEXEC)


def install_psovina(software_path):
	psovina_path = os.path.join(software_path, 'psovina')
	url = "https://github.com/li-jin-xing/RDPSOVina/raw/master/binary/Ubuntu-15.04/psovina"
	download_file(url, psovina_path)
	os.chmod(psovina_path, os.stat(psovina_path).st_mode | stat.S_IEXEC)


def install_plants(software_path):
	printlog(
		"PLANTS automatic installation is not supported. Please install PLANTS manually from : http://www.tcd.uni-konstanz.de/research/plants.php."
	)


def install_panther(software_path):
	subprocess.call('conda create -n panther python=2.7 -y',
					shell=True,
					stderr=subprocess.DEVNULL,
					stdout=subprocess.DEVNULL)
	printlog(
		"PANTHER automatic installation is not supported. \n" +
		"Please install PANTHER manually from : https://www.medchem.fi/panther/. \n" +
		"Ensure the panther directory is in the software folder. \n" +
		"Please also install ShaEP from : https://users.abo.fi/mivainio/shaep/download.php. \n" +
		"Ensure its executable is in the software folder. \n" +
		"Please note that a conda environment has already been created so the steps above are the only manual steps required."
	)


def install_plantain(software_path):
	plantain_path = Path(software_path) / "plantain"
	subprocess.call(f"git clone https://github.com/molecularmodelinglab/plantain.git {str(plantain_path)} --depth 1",
					shell=True,
					stdout=subprocess.DEVNULL,
					stderr=subprocess.DEVNULL)


def install_korp_pl(software_path):
	korp_pl_path = os.path.join(software_path, 'KORP-PL')
	url = "https://files.inria.fr/NanoDFiles/Website/Software/KORP-PL/0.1.2/Linux/KORP-PL-LINUX-v0.1.2.2.tar.gz"
	tar_file = os.path.join(software_path, 'KORP-PL-LINUX-v0.1.2.2.tar.gz')
	download_file(url, tar_file)
	with tarfile.open(tar_file, 'r:gz') as tar:
		tar.extractall(path=software_path)
	os.remove(tar_file)
	os.chmod(korp_pl_path, os.stat(korp_pl_path).st_mode | stat.S_IEXEC)


def install_convex_pl(software_path):
	convex_pl_path = os.path.join(software_path, 'Convex-PL')
	url = "https://files.inria.fr/NanoDFiles/Website/Software/Convex-PL/Files/Convex-PL-Linux-v0.5.tar.zip"
	zip_file = os.path.join(software_path, 'Convex-PL-Linux-v0.5.tar.zip')
	download_file(url, zip_file)
	with zipfile.ZipFile(zip_file, 'r') as zip_ref:
		zip_ref.extractall(software_path)
	tar_file = os.path.join(software_path, 'Convex-PL-Linux-v0.5.tar')
	with tarfile.open(tar_file, 'r') as tar:
		tar.extractall(path=software_path)
	os.remove(zip_file)
	os.remove(tar_file)
	shutil.rmtree(os.path.join(software_path, '__MACOSX'), ignore_errors=True)
	os.chmod(convex_pl_path, os.stat(convex_pl_path).st_mode | stat.S_IEXEC)


def install_lin_f9(software_path):
	lin_f9_path = os.path.join(software_path, 'LinF9')
	url = "https://github.com/cyangNYU/Lin_F9_test/raw/master/smina.static"
	download_file(url, lin_f9_path)
	os.chmod(lin_f9_path, os.stat(lin_f9_path).st_mode | stat.S_IEXEC)


def install_aa_score(software_path):
	aa_score_path = os.path.join(software_path, 'AA-Score-Tool-main')
	url = "https://github.com/Xundrug/AA-Score-Tool/archive/refs/heads/main.zip"
	zip_file = os.path.join(software_path, 'AA-Score-Tool-main.zip')
	download_file(url, zip_file)
	with zipfile.ZipFile(zip_file, 'r') as zip_ref:
		zip_ref.extractall(software_path)
	os.remove(zip_file)
	remove_git_files(aa_score_path)
	subprocess.call("conda create -n AAScore python=3.6 openbabel rdkit numpy scipy pandas py3dmol biopandas -y",
					shell=True,
					stderr=subprocess.DEVNULL,
					stdout=subprocess.DEVNULL)


def install_gypsum_dl(software_path):
	gypsum_dl_path = os.path.join(software_path, 'gypsum_dl-1.2.1')
	url = "https://github.com/durrantlab/gypsum_dl/archive/refs/tags/v1.2.1.tar.gz"
	tar_file = os.path.join(software_path, 'gypsum_dl-1.2.1.tar.gz')
	download_file(url, tar_file)
	with tarfile.open(tar_file, 'r:gz') as tar:
		tar.extractall(path=software_path)
	os.remove(tar_file)
	remove_git_files(gypsum_dl_path)


def install_scorch(software_path):
	scorch_path = os.path.join(software_path, 'SCORCH-1.0.0')
	url = "https://github.com/SMVDGroup/SCORCH/archive/refs/tags/v1.0.0.tar.gz"
	tar_file = os.path.join(software_path, 'SCORCH-1.0.0.tar.gz')
	download_file(url, tar_file)
	with tarfile.open(tar_file, 'r:gz') as tar:
		tar.extractall(path=software_path)
	os.remove(tar_file)
	remove_git_files(scorch_path)
	# Modify files as per the original script
	with open(os.path.join(scorch_path, 'utils', 'dock_functions.py'), 'r') as file:
		content = file.read()
	content = content.replace('import pybel', 'from openbabel import pybel')
	with open(os.path.join(scorch_path, 'utils', 'dock_functions.py'), 'w') as file:
		file.write(content)
	with open(os.path.join(scorch_path, 'scorch.py'), 'r') as file:
		content = file.read()
	content = content.replace("dtest = xgb.DMatrix(df, feature_names=df.columns)",
								"dtest = xgb.DMatrix(df, feature_names=list(df.columns))")
	with open(os.path.join(scorch_path, 'scorch.py'), 'w') as file:
		file.write(content)


def install_rf_score_vs(software_path):
	rf_score_vs_path = os.path.join(software_path, 'rf-score-vs')
	url = "https://github.com/oddt/rfscorevs_binary/releases/download/1.0/rf-score-vs_v1.0_linux_2.7.zip"
	zip_file = os.path.join(software_path, 'rf-score-vs_v1.0_linux_2.7.zip')
	download_file(url, zip_file)
	with zipfile.ZipFile(zip_file, 'r') as zip_ref:
		zip_ref.extractall(software_path)
	os.remove(zip_file)
	shutil.rmtree(os.path.join(software_path, 'test'), ignore_errors=True)
	os.remove(os.path.join(software_path, 'README.md'))
	os.chmod(rf_score_vs_path, os.stat(rf_score_vs_path).st_mode | stat.S_IEXEC)
	remove_git_files(rf_score_vs_path)


def install_rtmscore(software_path):
	rtmscore_path = os.path.join(software_path, 'RTMScore-main')
	url = "https://github.com/sc8668/RTMScore/archive/refs/heads/main.zip"
	zip_file = os.path.join(software_path, 'RTMScore-main.zip')
	download_file(url, zip_file)
	with zipfile.ZipFile(zip_file, 'r') as zip_ref:
		zip_ref.extractall(software_path)
	os.remove(zip_file)
	shutil.rmtree(os.path.join(rtmscore_path, 'scripts'), ignore_errors=True)
	os.remove(os.path.join(rtmscore_path, '121.jpg'))
	remove_git_files(rtmscore_path)


def install_posecheck(software_path):
	posecheck_path = os.path.join(software_path, 'posecheck-main')
	url = "https://github.com/cch1999/posecheck/archive/refs/heads/main.zip"
	zip_file = os.path.join(software_path, 'posecheck-main.zip')
	download_file(url, zip_file)
	with zipfile.ZipFile(zip_file, 'r') as zip_ref:
		zip_ref.extractall(software_path)
	os.remove(zip_file)
	remove_git_files(posecheck_path)
	subprocess.run(['pip', 'install', '-e', './posecheck-main'], cwd=software_path)


def install_deepcoy_models(software_path):
	deepcoy_path = os.path.join(software_path, 'models', 'DeepCoy')
	url = "https://opig.stats.ox.ac.uk/data/downloads/DeepCoy_pretrained_models.tar.gz"
	tar_file = os.path.join(software_path, 'DeepCoy_pretrained_models.tar.gz')
	download_file(url, tar_file)
	with tarfile.open(tar_file, 'r:gz') as tar:
		tar.extractall(path=software_path)
	os.remove(tar_file)


def install_fabind(software_path):
	fabind_path = Path(software_path) / 'FABind'

	# Clone FABind repository
	subprocess.run(['git', 'clone', 'https://github.com/QizhiPei/FABind.git', '--recursive', '--depth', '1'],
					cwd=software_path,
					stderr=subprocess.DEVNULL,
					stdout=subprocess.DEVNULL,
					check=True)
	remove_git_files(fabind_path)

	# Create and set up conda environment
	env_name = "fabind"
	subprocess.run(f"conda create -n {env_name} python=3.8 -y", shell=True, check=True)

	# Install conda packages
	conda_install_cmd = (
		f"conda run -n {env_name} conda install -c conda-forge graph-tool -y && "
		f"conda run -n {env_name} conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cpuonly -c pytorch -y && "
		f"conda run -n {env_name} conda install -c conda-forge openbabel -y")
	subprocess.run(conda_install_cmd, shell=True, check=True)

	# Install pip packages
	pip_install_cmd = (
		f"conda run -n {env_name} pip install https://data.pyg.org/whl/torch-1.12.0%2Bcpu/torch_cluster-1.6.0%2Bpt112cpu-cp38-cp38-linux_x86_64.whl &&"
		f"conda run -n {env_name} pip install https://data.pyg.org/whl/torch-1.12.0%2Bcpu/torch_scatter-2.1.0%2Bpt112cpu-cp38-cp38-linux_x86_64.whl &&"
		f"conda run -n {env_name} pip install https://data.pyg.org/whl/torch-1.12.0%2Bcpu/torch_sparse-0.6.15%2Bpt112cpu-cp38-cp38-linux_x86_64.whl &&"
		f"conda run -n {env_name} pip install https://data.pyg.org/whl/torch-1.12.0%2Bcpu/torch_spline_conv-1.2.1%2Bpt112cpu-cp38-cp38-linux_x86_64.whl &&"
		f"conda run -n {env_name} pip install https://data.pyg.org/whl/torch-1.12.0%2Bcpu/pyg_lib-0.2.0%2Bpt112cpu-cp38-cp38-linux_x86_64.whl &&"
		f"conda run -n {env_name} pip install torch-geometric==2.4.0 torchdrug==0.1.2 torchmetrics==0.10.2 tqdm mlcrate pyarrow accelerate Bio lmdb fair-esm tensorboard &&"
		f"conda run -n {env_name} pip install wandb spyrmsd rdkit-pypi==2021.03.4 setuptools==69.5.1")
	subprocess.run(pip_install_cmd, shell=True, check=True)


def install_censible(software_path):
	censible_folder = os.path.join(software_path, 'censible')
	url = "https://github.com/durrantlab/censible/archive/refs/heads/main.zip"
	zip_file = os.path.join(software_path, 'censible.zip')
	download_file(url, zip_file)
	with zipfile.ZipFile(zip_file, 'r') as zip_ref:
		zip_ref.extractall(software_path)
	os.rename(os.path.join(software_path, 'censible-main'), censible_folder)
	os.remove(zip_file)
	remove_git_files(censible_folder)


def install_dligand2(software_path):
	dligand2_folder = os.path.join(software_path, 'DLIGAND2')
	url = "https://github.com/yuedongyang/DLIGAND2/archive/refs/heads/master.zip"
	zip_file = os.path.join(software_path, 'DLIGAND2.zip')
	download_file(url, zip_file)
	with zipfile.ZipFile(zip_file, 'r') as zip_ref:
		zip_ref.extractall(software_path)
	os.rename(os.path.join(software_path, 'DLIGAND2-master'), dligand2_folder)
	os.remove(zip_file)
	remove_git_files(dligand2_folder)
	executable_path = os.path.join(dligand2_folder, 'bin', 'dligand2.gnu')
	os.chmod(executable_path, os.stat(executable_path).st_mode | stat.S_IEXEC)


def install_itscoreAff(software_path):
	itscore_folder = os.path.join(software_path, 'ITScoreAff_v1.0')
	url = "http://huanglab.phys.hust.edu.cn/ITScoreAff/ITScoreAff_v1.0.tar.gz"
	tar_file = os.path.join(software_path, 'ITScoreAff_v1.0.tar.gz')
	download_file(url, tar_file)
	with tarfile.open(tar_file, 'r:gz') as tar:
		tar.extractall(path=software_path)
	os.remove(tar_file)
	executable_path = os.path.join(itscore_folder, 'ITScoreAff')
	os.chmod(executable_path, os.stat(executable_path).st_mode | stat.S_IEXEC)


def install_genscore(software_path):
	genscore_path = Path(software_path) / 'GenScore'
	url = "https://github.com/sc8668/GenScore/archive/refs/heads/main.zip"
	zip_file = genscore_path.parent / 'GenScore-main.zip'

	# Download and extract GenScore
	download_file(url, str(zip_file))
	with zipfile.ZipFile(zip_file, 'r') as zip_ref:
		zip_ref.extractall(software_path)
	os.rename(genscore_path.parent / 'GenScore-main', genscore_path)
	os.remove(zip_file)
	remove_git_files(genscore_path)

	# Create and activate conda environment
	env_name = "genscore"
	subprocess.run(f"conda create -n {env_name} python=3.8 -y", shell=True, check=True)

	# Install dependencies
	conda_install_cmd = (
		f"conda run -n {env_name} conda install "
		"pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cpuonly -c pytorch -y && "
		f"conda run -n {env_name} conda install MDAnalysis==2.0.0 prody==2.1.0 pandas rdkit==2021.03.5 openbabel "
		"scikit-learn scipy seaborn numpy joblib matplotlib -y")
	subprocess.run(conda_install_cmd, shell=True, check=True)

	# Install PyTorch Geometric and related packages
	pip_install_cmd = (f"conda run -n {env_name} pip install torch-geometric==2.0.3 "
						"https://data.pyg.org/whl/torch-1.11.0%2Bcpu/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl "
						"https://data.pyg.org/whl/torch-1.11.0%2Bcpu/torch_sparse-0.6.13-cp38-cp38-linux_x86_64.whl")
	subprocess.run(pip_install_cmd, shell=True, check=True)

def install_mgltools():
	env_name = 'mgltools'
	subprocess.run(f"conda create -n {env_name} python=2.7 -y", shell=True, check=True)
	subprocess.run(f"conda run -n {env_name} conda install -c bioconda mgltools -y", shell=True, check=True)

def install_all_software(software_path):
	os.makedirs(software_path, exist_ok=True)
	install_gnina(software_path)
	install_qvina_w(software_path)
	install_qvina2(software_path)
	install_psovina(software_path)
	install_korp_pl(software_path)
	install_convex_pl(software_path)
	install_lin_f9(software_path)
	install_aa_score(software_path)
	install_gypsum_dl(software_path)
	install_scorch(software_path)
	install_rf_score_vs(software_path)
	install_rtmscore(software_path)
	install_posecheck(software_path)
	install_deepcoy_models(software_path)
	install_fabind(software_path)
	install_censible(software_path)
	install_dligand2(software_path)
	install_itscoreAff(software_path)
	install_genscore(software_path)

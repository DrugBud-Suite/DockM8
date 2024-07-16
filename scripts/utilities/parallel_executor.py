import concurrent.futures
from pathlib import Path
import sys
import warnings
from joblib import delayed
from joblib import Parallel
import pebble
from tqdm import tqdm

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def is_running_in_streamlit():
	try:
		from streamlit.runtime.scriptrunner import get_script_run_ctx
		return get_script_run_ctx() is not None
	except ImportError:
		return False


if is_running_in_streamlit():
	from stqdm import stqdm as tqdm
else:
	from tqdm import tqdm


def parallel_executor(function,
						list_of_objects: list,
						n_cpus: int,
						job_manager="concurrent_process",
						display_name: str = None,
						**kwargs):
	"""
    Executes a function in parallel using multiple processes.

    Args:
        function (function): The function to execute in parallel.
        list_of_objects (list): A list of input arguments to pass to the function.
        n_cpus (int): The number of CPUs to use for parallel execution.
        job_manager (str, optional): The job manager to use for parallel execution. 
            Options are "concurrent_process", "concurrent_process_silent", "concurrent_thread", 
            "joblib", "pebble_process", and "pebble_thread". Default is "concurrent_process".
        **kwargs: Additional keyword arguments to pass to the function.

    Returns:
        list: The result of the function execution.
    """
	function_name = function.__name__

	if job_manager == "concurrent_process":
		with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpus) as executor:
			jobs = [executor.submit(function, obj, **kwargs) for obj in list_of_objects]
			results = [
				job.result() for job in tqdm(concurrent.futures.as_completed(jobs),
												total=len(list_of_objects),
												desc=f"Running {display_name if display_name else function_name}")]

	elif job_manager == "concurrent_process_silent":
		with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpus) as executor:
			jobs = [executor.submit(function, obj, **kwargs) for obj in list_of_objects]
			results = [job.result() for job in concurrent.futures.as_completed(jobs)]

	elif job_manager == "concurrent_thread":
		with concurrent.futures.ThreadPoolExecutor(max_workers=n_cpus) as executor:
			jobs = [executor.submit(function, obj, **kwargs) for obj in list_of_objects]
			results = [
				job.result() for job in tqdm(concurrent.futures.as_completed(jobs),
												total=len(list_of_objects),
												desc=f"Running {display_name if display_name else function_name}")]

	elif job_manager == "joblib":
		jobs = [delayed(function)(obj, **kwargs) for obj in list_of_objects]
		results = Parallel(n_jobs=n_cpus)(tqdm(jobs,
												total=len(list_of_objects),
												desc=f"Running {display_name if display_name else function_name}"))

	elif job_manager == "pebble_process":
		with pebble.ProcessPool(max_workers=n_cpus) as executor:
			jobs = [executor.schedule(function, args=(obj, ), kwargs=kwargs) for obj in list_of_objects]
			results = [job.result() for job in jobs]

	elif job_manager == "pebble_thread":
		with pebble.ThreadPool(max_workers=n_cpus) as executor:
			jobs = [executor.schedule(function, args=(obj, ), kwargs=kwargs) for obj in list_of_objects]
			results = [job.result() for job in jobs]
	else:
		raise ValueError(f"Invalid job_manager: {job_manager}")

	return results

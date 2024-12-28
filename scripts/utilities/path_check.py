import shutil
from pathlib import Path

def get_executable_path(software_path: Path, executable_name: str) -> str:
    """
    Check if executable exists in system path, otherwise use provided software path.
    
    Args:
        software_path: Path to software directory containing executable
        executable_name: Name of the executable to find
        
    Returns:
        Full path to executable as string
    """
    # Check if executable is in system PATH
    system_exe = shutil.which(executable_name)
    if system_exe:
        return executable_name
        
    # Fall back to software path
    return f"{software_path}/{executable_name}"

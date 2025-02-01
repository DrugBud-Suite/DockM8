# scripts/utilities/subprocess_handler.py
import subprocess

def run_subprocess_command(
    command: str,
    shell: bool = True
) -> tuple[str | None, str | None]:
    """
    Execute a subprocess command and return its outputs.
    
    Args:
        command: The command to execute
        shell: Whether to run the command in a shell
        
    Returns:
        Tuple containing:
        - Standard output (str or None)
        - Error output (str or None)
    """
    try:
        result = subprocess.run(
            command,
            shell=shell,
            capture_output=True,
            text=True
        )
        return result.stdout, result.stderr

    except Exception as e:
        return None, str(e)

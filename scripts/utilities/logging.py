from pathlib import Path
import sys
import warnings
import datetime

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


def printlog(message):
    """
    Prints the given message along with a timestamp to the console or Streamlit app,
    and appends it to a log file.

    Args:
        message (str): The message to be logged.

    Returns:
        None
    """

    def timestamp_generator():
        dateTimeObj = datetime.datetime.now()
        return "[" + dateTimeObj.strftime("%Y-%b-%d %H:%M:%S") + "]"

    timestamp = timestamp_generator()
    msg = str(timestamp) + ": " + str(message) + "\n"

    if is_running_in_streamlit():
        import streamlit as st

        st.markdown(f"*{message}*")
    else:
        print(msg)

    log_file_path = dockm8_path / "log.txt"
    with open(log_file_path, "a") as f_out:
        f_out.write(msg)

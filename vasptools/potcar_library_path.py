import os
import sys

def get_potcar_library_path():
    """
    Retrieve the POTCAR library path from the environment variable VASPPOTPATH.
    Raise an error if it's not set, and explain how to set it on macOS/Linux or Windows.
    """
    if "VASPPOTPATH" in os.environ:
        return os.environ["VASPPOTPATH"]
    else:
        sys.exit(
            "Environment variable VASPPOTPATH is not set.\n"
            "Please set it to the path of your POTCAR library.\n\n"
            "For macOS/Linux (bash or zsh), in your ~/.bashrc or ~/.zshrc:\n"
            "  export VASPPOTPATH=\"/absolute/path/to/potcar_library\"\n"
            "Then run: source ~/.bashrc  (or source ~/.zshrc)\n\n"
            "For Windows (PowerShell):\n"
            "  $env:VASPPOTPATH = \"C:\\path\\to\\potcar_library\"\n"
            "For Windows (Command Prompt):\n"
            "  set VASPPOTPATH=C:\\path\\to\\potcar_library\n"
        )

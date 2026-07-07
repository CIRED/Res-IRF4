"""Entry point for PyInstaller-bundled executable.

Ensures the working directory is set correctly so that relative paths
(e.g. 'project/config/...', 'project/output/...') resolve properly
both when running from source and from a frozen PyInstaller bundle.
"""

import sys
import os
import logging


def _setup_frozen_env():
    """When running as a PyInstaller bundle, change the working directory
    to the directory containing the executable so that 'project/' paths
    resolve relative to the bundle location."""
    if getattr(sys, 'frozen', False):
        # PyInstaller sets sys._MEIPASS to the temp extraction folder (onefile)
        # or the bundle directory (onedir).  For --onedir the executable sits
        # next to the 'project/' folder inside the dist directory.
        bundle_dir = os.path.dirname(sys.executable)
        os.chdir(bundle_dir)


if __name__ == '__main__':
    _setup_frozen_env()

    logging.basicConfig()
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib.axes').disabled = True

    from project.main import run
    run()

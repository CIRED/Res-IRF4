# -*- mode: python ; coding: utf-8 -*-

"""PyInstaller spec file for Res-IRF4.

Build with:
    pyinstaller res_irf.spec

This produces a one-folder bundle in dist/ResIRF/ containing the
executable and all required data files.
"""

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all submodules so importlib.resources can find package data
hidden_imports = collect_submodules('project')

# Collect all non-Python data files (CSV, JSON, etc.) from the project package
# This ensures importlib.resources can locate them at runtime
datas = collect_data_files('project', include_py_files=False)

# Also include config files alongside the executable (accessed via open())
# These are read with plain file I/O, not importlib.resources
datas += [
    ('project/config', 'project/config'),
]

a = Analysis(
    ['project/cli_entry.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary packages to reduce bundle size
        'tkinter',
        'ipykernel',
        'ipython',
        'jupyter',
        'notebook',
        'tornado',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ResIRF',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ResIRF',
)

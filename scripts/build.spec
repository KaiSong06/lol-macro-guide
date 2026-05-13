# PyInstaller spec for lolcoach.
#
# Run from the repo root with:
#   python scripts/build-exe.py

from pathlib import Path

ROOT = Path.cwd()

block_cipher = None

datas = [
    (str(ROOT / "config.yaml.example"), "."),
    (str(ROOT / "scripts" / "binaries" / "piper"), "piper"),
    (str(ROOT / "scripts" / "binaries" / "voices"), "voices"),
]

a = Analysis(
    [str(ROOT / "src" / "lolcoach" / "__main__.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "cv2",
        "numpy",
        "requests",
        "yaml",
        "sounddevice",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "lolcoach.autolaunch",
        "lolcoach.ui.app",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="lol-macro-guide",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

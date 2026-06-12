# -*- mode: python ; coding: utf-8 -*-

import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules


def _app_version() -> str:
    env_version = os.environ.get("APP_VERSION")
    if env_version:
        return env_version
    try:
        return _dist_version("ptarmigan-flow")
    except PackageNotFoundError:
        return "0.0.0"


block_cipher = None
ROOT = Path(SPECPATH).parents[1]
APP_VERSION = _app_version()
hiddenimports = collect_submodules("ptarmigan_flow")
datas = [(str(ROOT / "config.example.toml"), ".")]
binaries = []

# These packages ship non-Python assets (Metal shaders, dylibs, tokenizer data)
# that pyinstaller-hooks-contrib has no hooks for; without collect_all the
# bundled app fails at runtime (e.g. mlx cannot load mlx.metallib).
for package in (
    "mlx",
    "mlx_whisper",
    "mlx_audio",
    "voxmlx",
    "moonshine_voice",
    "mistral_common",
):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package)
    if package == "moonshine_voice":
        # moonshine-voice macOS wheels (≤0.0.59) ship a Linux ELF named
        # libmoonshine.so due to a packaging bug. Exclude it from binaries
        # so PyInstaller does not attempt Mach-O analysis on an ELF file.
        pkg_binaries = [
            (src, dst, typ)
            for src, dst, typ in pkg_binaries
            if not os.path.basename(src).startswith("libmoonshine")
        ]
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

a = Analysis(
    [str(ROOT / "src/ptarmigan_flow/macos_app.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PtarmiganFlow',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    target_arch='arm64',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='PtarmiganFlow',
)
app = BUNDLE(
    coll,
    name='PtarmiganFlow.app',
    icon=None,
    bundle_identifier='com.ptarmiganflow.app',
    info_plist={
        "CFBundleDisplayName": "PtarmiganFlow",
        "CFBundleName": "PtarmiganFlow",
        "CFBundleShortVersionString": APP_VERSION,
        "CFBundleVersion": APP_VERSION,
        "LSMinimumSystemVersion": "14.0",
        "NSMicrophoneUsageDescription": "PtarmiganFlow records audio only while the hotkey is held.",
        "NSAppleEventsUsageDescription": "PtarmiganFlow sends text to the active app when you release the hotkey.",
    },
)

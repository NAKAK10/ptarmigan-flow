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
datas = [(str(ROOT / "config.example.toml"), ".")]
binaries = []

OPTIONAL_BACKEND_MODULE_PREFIXES = (
    "ptarmigan_flow.stt.granite_mlx",
    "ptarmigan_flow.stt.granite_transformers",
    "ptarmigan_flow.stt.mlx_whisper",
    "ptarmigan_flow.stt.voxtral_mlx",
    "ptarmigan_flow.stt.voxtral_transformers",
    "ptarmigan_flow.stt._test_support",
)

OPTIONAL_BACKEND_PACKAGE_EXCLUDES = (
    "mlx",
    "mlx_audio",
    "mlx_whisper",
    "voxmlx",
    "mistral_common",
    "torch",
    "transformers",
    "scipy",
    "sklearn",
    "numba",
    "llvmlite",
    "pytest",
)

RELEASE_BACKEND_PACKAGES = (
    "moonshine_voice",
)

MOONSHINE_EXCLUDED_BASENAMES = {
    "libmoonshine.so",
    "beckett.wav",
    "two_cities.wav",
}

MOONSHINE_EXCLUDED_HIDDENIMPORTS = {
    "moonshine_voice.libmoonshine",
}


def _keep_release_module(name: str) -> bool:
    return not name.startswith(OPTIONAL_BACKEND_MODULE_PREFIXES)


def _drop_unneeded_moonshine_entries(entries):
    return [
        entry
        for entry in entries
        if Path(entry[0]).name not in MOONSHINE_EXCLUDED_BASENAMES
    ]


hiddenimports = [
    name
    for name in collect_submodules("ptarmigan_flow")
    if _keep_release_module(name)
]

# The downloadable app is intentionally a compact Moonshine build. Larger
# MLX/Granite/Voxtral/Torch backends remain available through the CLI/Homebrew
# environment, but are not shipped in this small app artifact.
for package in RELEASE_BACKEND_PACKAGES:
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package)
    if package == "moonshine_voice":
        # moonshine-voice macOS wheels (≤0.0.59) ship a Linux ELF named
        # libmoonshine.so due to a packaging bug. Exclude it from datas,
        # binaries, and hidden imports so PyInstaller never attempts Mach-O
        # analysis on it.
        pkg_datas = _drop_unneeded_moonshine_entries(pkg_datas)
        pkg_binaries = _drop_unneeded_moonshine_entries(pkg_binaries)
        pkg_hiddenimports = [
            name for name in pkg_hiddenimports if name not in MOONSHINE_EXCLUDED_HIDDENIMPORTS
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
    excludes=[
        *OPTIONAL_BACKEND_MODULE_PREFIXES,
        *OPTIONAL_BACKEND_PACKAGE_EXCLUDES,
    ],
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

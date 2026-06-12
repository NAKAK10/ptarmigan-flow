# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


block_cipher = None
ROOT = Path(SPECPATH).parents[1]
hiddenimports = collect_submodules("ptarmigan_flow")

a = Analysis(
    [str(ROOT / "src/ptarmigan_flow/macos_app.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[(str(ROOT / "config.example.toml"), ".")],
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
        "CFBundleShortVersionString": "0.0.0",
        "CFBundleVersion": "0.0.0",
        "LSMinimumSystemVersion": "14.0",
        "NSMicrophoneUsageDescription": "PtarmiganFlow records audio only while the hotkey is held.",
        "NSAppleEventsUsageDescription": "PtarmiganFlow sends text to the active app when you release the hotkey.",
    },
)

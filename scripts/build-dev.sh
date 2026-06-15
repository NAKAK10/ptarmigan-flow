#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Stopping PtarmiganFlow if running..."
pkill -x "PtarmiganFlow" 2>/dev/null && sleep 1 || true

echo "==> Resetting TCC permissions..."
tccutil reset Microphone com.ptarmiganflow.app
tccutil reset Accessibility com.ptarmiganflow.app
tccutil reset ListenEvent com.ptarmiganflow.app

echo "==> Building macOS app..."
cd "$REPO_ROOT"
export APP_VERSION="${APP_VERSION:-0.0.0-dev}"
uv run pyinstaller --clean --noconfirm packaging/macos/PtarmiganFlow.spec

echo "==> Opening app..."
open dist/PtarmiganFlow.app

echo ""
echo "Done. Re-grant permissions in System Settings when prompted."

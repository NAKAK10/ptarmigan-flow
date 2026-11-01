"""Check GitHub releases for a newer app and prepare an in-place update.

This module is PyObjC-free: network, subprocess and filesystem access go
through injectable callables so the macOS app only wires the UI around it.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import plistlib
import re
import shlex
import shutil
import ssl
import subprocess
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GITHUB_REPOSITORY = "NAKAK10/ptarmigan-flow"
LATEST_RELEASE_API_URL = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/releases/latest"
RELEASES_PAGE_URL = f"https://github.com/{GITHUB_REPOSITORY}/releases/latest"
ASSET_NAME_TEMPLATE = "PtarmiganFlow-macos-{arch}.zip"
APP_BUNDLE_NAME = "PtarmiganFlow.app"
USER_AGENT = "PtarmiganFlow-Updater"
REQUEST_TIMEOUT_SECONDS = 15.0
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
SUBPROCESS_TIMEOUT_SECONDS = 120.0

# Reasons a newer release cannot be installed in place. The app maps them to
# localized text; they are stable identifiers, not user-facing strings.
INSTALL_BLOCK_NOT_FROZEN = "not_frozen"
INSTALL_BLOCK_DEV_BUILD = "dev_build"
INSTALL_BLOCK_NO_BUNDLE = "no_bundle"
INSTALL_BLOCK_TRANSLOCATED = "translocated"
INSTALL_BLOCK_NOT_WRITABLE = "not_writable"
INSTALL_BLOCK_NO_ASSET = "no_asset"
# Blockers that clear up on their own (the release assets are still uploading).
TRANSIENT_INSTALL_BLOCKERS = frozenset({INSTALL_BLOCK_NO_ASSET})

# What the app should do after a check; see ``classify_check_result``.
CHECK_OUTCOME_UP_TO_DATE = "up_to_date"
CHECK_OUTCOME_PREPARE = "prepare"
CHECK_OUTCOME_RETRY = "retry"
CHECK_OUTCOME_BLOCKED = "blocked"

# One GitHub API call after a transient failure; never a repeating poll (#29).
UPDATE_RECHECK_DELAY_SECONDS = 300.0
# Each attempt downloads the whole app, so stop after this many bad downloads.
MAX_VERIFICATION_FAILURES = 3

_RELEASE_VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")

LOGGER = logging.getLogger(__name__)

Runner = Callable[[list[str]], subprocess.CompletedProcess]
UrlOpen = Callable[..., Any]
ProgressCallback = Callable[[int, int | None], None]


class UpdateError(Exception):
    """Raised when checking, downloading or verifying an update fails."""


class VerificationError(UpdateError):
    """Raised when a downloaded app fails signature or identity verification."""


@dataclass(frozen=True)
class ReleaseAsset:
    name: str
    url: str
    size: int | None = None


@dataclass(frozen=True)
class ReleaseInfo:
    tag: str
    version: str
    html_url: str
    assets: tuple[ReleaseAsset, ...] = ()
    body: str = ""


@dataclass(frozen=True)
class UpdateCheckResult:
    """Outcome of comparing the running app with the latest release."""

    current_version: str
    release: ReleaseInfo | None = None
    asset: ReleaseAsset | None = None
    update_available: bool = False
    install_blocker: str | None = None
    error: str | None = None

    @property
    def can_install(self) -> bool:
        return self.update_available and self.asset is not None and self.install_blocker is None

    @property
    def is_dev_build(self) -> bool:
        return not is_release_version(self.current_version)

    @property
    def release_page_url(self) -> str:
        if self.release is not None and self.release.html_url:
            return self.release.html_url
        return RELEASES_PAGE_URL


@dataclass(frozen=True)
class VerificationFacts:
    codesign_ok: bool
    spctl_ok: bool
    new_team_id: str | None
    current_team_id: str | None
    new_bundle_id: str | None
    current_bundle_id: str | None
    new_version: str | None
    expected_version: str


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    failures: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PreparedUpdate:
    new_app: Path
    work_dir: Path
    version: str


@dataclass
class PrepareAttempts:
    """Count verification failures per release so a bad asset is not re-downloaded forever."""

    max_verification_failures: int = MAX_VERIFICATION_FAILURES
    version: str | None = None
    verification_failures: int = 0

    def record_verification_failure(self, version: str) -> int:
        if version != self.version:
            self.version = version
            self.verification_failures = 0
        self.verification_failures += 1
        return self.verification_failures

    def exhausted(self, version: str) -> bool:
        return (
            version == self.version
            and self.verification_failures >= self.max_verification_failures
        )

    def reset(self) -> None:
        self.version = None
        self.verification_failures = 0


# --- versions -----------------------------------------------------------------


def parse_release_version(text: str | None) -> tuple[int, int, int] | None:
    """Parse ``X.Y.Z`` (optionally ``vX.Y.Z``); anything else is not a release."""
    if not text:
        return None
    match = _RELEASE_VERSION_RE.match(text.strip())
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def is_release_version(text: str | None) -> bool:
    """Return whether ``text`` is a plain release version (not ``-dev``/``0.0.0``)."""
    parsed = parse_release_version(text)
    return parsed is not None and parsed != (0, 0, 0)


def normalize_version(text: str) -> str:
    text = text.strip()
    return text[1:] if text[:1] in {"v", "V"} else text


def compare_versions(left: str, right: str) -> int:
    """Compare two release versions; raise ``ValueError`` for non-release input."""
    left_parsed = parse_release_version(left)
    right_parsed = parse_release_version(right)
    if left_parsed is None or right_parsed is None:
        raise ValueError(f"not a release version: {left!r} / {right!r}")
    return (left_parsed > right_parsed) - (left_parsed < right_parsed)


def is_newer_version(latest: str, current: str) -> bool:
    """Return whether ``latest`` is newer than ``current``.

    A development build (``0.0.0-dev`` etc.) is never considered current, so any
    release counts as newer for it.
    """
    latest_parsed = parse_release_version(latest)
    if latest_parsed is None:
        return False
    current_parsed = parse_release_version(current)
    if current_parsed is None:
        return True
    return latest_parsed > current_parsed


# --- releases -----------------------------------------------------------------


def normalize_arch(machine: str | None = None) -> str | None:
    value = (machine if machine is not None else platform.machine()).strip().lower()
    if value in {"arm64", "aarch64"}:
        return "arm64"
    if value in {"x86_64", "amd64"}:
        return "x86_64"
    return None


def asset_name_for_arch(arch: str) -> str:
    return ASSET_NAME_TEMPLATE.format(arch=arch)


def select_asset(release: ReleaseInfo, arch: str | None) -> ReleaseAsset | None:
    if arch is None:
        return None
    wanted = asset_name_for_arch(arch)
    for asset in release.assets:
        if asset.name == wanted:
            return asset
    return None


def parse_release(payload: Mapping[str, Any]) -> ReleaseInfo:
    """Build ``ReleaseInfo`` from a GitHub release JSON object."""
    if not isinstance(payload, Mapping):
        raise UpdateError("unexpected release payload")
    if payload.get("draft") or payload.get("prerelease"):
        raise UpdateError("latest release is a draft or prerelease")
    tag = payload.get("tag_name")
    if not isinstance(tag, str) or not is_release_version(tag):
        raise UpdateError(f"unexpected release tag: {tag!r}")
    assets: list[ReleaseAsset] = []
    for raw in payload.get("assets") or []:
        if not isinstance(raw, Mapping):
            continue
        name = raw.get("name")
        url = raw.get("browser_download_url")
        if not isinstance(name, str) or not isinstance(url, str):
            continue
        if not url.startswith("https://"):
            continue
        size = raw.get("size")
        assets.append(
            ReleaseAsset(name=name, url=url, size=size if isinstance(size, int) else None)
        )
    html_url = payload.get("html_url")
    body = payload.get("body")
    return ReleaseInfo(
        tag=tag,
        version=normalize_version(tag),
        html_url=html_url if isinstance(html_url, str) else RELEASES_PAGE_URL,
        assets=tuple(assets),
        body=body if isinstance(body, str) else "",
    )


def _request(url: str, *, accept: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})


def _ssl_context() -> ssl.SSLContext:
    # The frozen app's OpenSSL has no usable system CA path; certifi is bundled.
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _default_urlopen(request: urllib.request.Request, timeout: float) -> Any:
    return urllib.request.urlopen(request, timeout=timeout, context=_ssl_context())


def fetch_latest_release(
    *,
    urlopen: UrlOpen | None = None,
    url: str = LATEST_RELEASE_API_URL,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> ReleaseInfo:
    opener = urlopen or _default_urlopen
    try:
        with opener(_request(url, accept="application/vnd.github+json"), timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise UpdateError(f"could not fetch the latest release: {exc}") from exc
    return parse_release(payload)


def install_blocker(
    *,
    current_version: str,
    frozen: bool,
    bundle_path: Path | None,
    is_writable: Callable[[Path], bool] | None = None,
) -> str | None:
    """Return why the running app cannot replace itself, or ``None`` if it can."""
    if not frozen:
        return INSTALL_BLOCK_NOT_FROZEN
    if not is_release_version(current_version):
        return INSTALL_BLOCK_DEV_BUILD
    if bundle_path is None:
        return INSTALL_BLOCK_NO_BUNDLE
    if "/AppTranslocation/" in str(bundle_path):
        return INSTALL_BLOCK_TRANSLOCATED
    writable = is_writable or bundle_location_is_writable
    if not writable(bundle_path):
        return INSTALL_BLOCK_NOT_WRITABLE
    return None


def check_for_update(
    current_version: str,
    *,
    frozen: bool,
    bundle_path: Path | None,
    arch: str | None = None,
    fetch: Callable[[], ReleaseInfo] = fetch_latest_release,
    is_writable: Callable[[Path], bool] | None = None,
) -> UpdateCheckResult:
    """Fetch the latest release and decide whether it can be installed."""
    try:
        release = fetch()
    except UpdateError as exc:
        LOGGER.warning("Update check failed (current %s): %s", current_version, exc)
        return UpdateCheckResult(current_version=current_version, error=str(exc))
    if not is_newer_version(release.version, current_version):
        LOGGER.info(
            "Update check: current %s, latest %s, update available: no",
            current_version,
            release.version,
        )
        return UpdateCheckResult(current_version=current_version, release=release)
    asset = select_asset(release, arch if arch is not None else normalize_arch())
    blocker = install_blocker(
        current_version=current_version,
        frozen=frozen,
        bundle_path=bundle_path,
        is_writable=is_writable,
    )
    if blocker is None and asset is None:
        blocker = INSTALL_BLOCK_NO_ASSET
    LOGGER.info(
        "Update check: current %s, latest %s, update available: yes, asset %s, install blocker %s",
        current_version,
        release.version,
        asset.name if asset is not None else None,
        blocker,
    )
    return UpdateCheckResult(
        current_version=current_version,
        release=release,
        asset=asset,
        update_available=True,
        install_blocker=blocker,
    )


def classify_check_result(result: UpdateCheckResult) -> str:
    """Map a check result to the next step: prepare, retry later, blocked or nothing."""
    if result.error is not None:
        return CHECK_OUTCOME_RETRY
    if not result.update_available or result.release is None:
        return CHECK_OUTCOME_UP_TO_DATE
    if result.can_install:
        return CHECK_OUTCOME_PREPARE
    if result.install_blocker in TRANSIENT_INSTALL_BLOCKERS:
        return CHECK_OUTCOME_RETRY
    return CHECK_OUTCOME_BLOCKED


# --- running app --------------------------------------------------------------


def read_bundle_info(app_path: Path) -> dict[str, Any]:
    try:
        with (app_path / "Contents" / "Info.plist").open("rb") as handle:
            info = plistlib.load(handle)
    except (OSError, plistlib.InvalidFileException, ValueError):
        return {}
    return info if isinstance(info, dict) else {}


def bundle_location_is_writable(
    bundle_path: Path,
    *,
    access: Callable[[str, int], bool] = os.access,
) -> bool:
    """The helper renames the bundle and deletes the old copy, so both need write access."""
    writable = access(str(bundle_path.parent), os.W_OK) and access(str(bundle_path), os.W_OK)
    LOGGER.info("Update bundle location %s writable: %s", bundle_path, writable)
    return writable


# --- download / extract -------------------------------------------------------


def download_asset(
    asset: ReleaseAsset,
    destination: Path,
    *,
    urlopen: UrlOpen | None = None,
    progress: ProgressCallback | None = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
    chunk_size: int = DOWNLOAD_CHUNK_SIZE,
) -> Path:
    opener = urlopen or _default_urlopen
    LOGGER.info("Update download started: %s -> %s", asset.url, destination)
    try:
        with opener(
            _request(asset.url, accept="application/octet-stream"), timeout=timeout
        ) as resp:
            total = asset.size
            headers = getattr(resp, "headers", None)
            length = headers.get("Content-Length") if headers is not None else None
            if length and str(length).isdigit():
                total = int(length)
            downloaded = 0
            with destination.open("wb") as out:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if progress is not None:
                        progress(downloaded, total)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        LOGGER.warning("Update download failed: %s", exc, exc_info=True)
        raise UpdateError(f"download failed: {exc}") from exc
    if total is not None and downloaded != total:
        LOGGER.warning("Update download incomplete: %d of %d bytes", downloaded, total)
        raise UpdateError(f"download incomplete: {downloaded} of {total} bytes")
    LOGGER.info("Update download finished: %d bytes (expected %s)", downloaded, total)
    return destination


def run_command(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
        check=False,
    )


def _run(runner: Runner, argv: list[str]) -> subprocess.CompletedProcess | None:
    try:
        return runner(argv)
    except (OSError, subprocess.SubprocessError):
        return None


def extract_app(zip_path: Path, destination: Path, *, runner: Runner = run_command) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    result = _run(runner, ["/usr/bin/ditto", "-x", "-k", str(zip_path), str(destination)])
    if result is None or result.returncode != 0:
        detail = (result.stderr or "").strip() if result is not None else "ditto did not run"
        LOGGER.warning("Update extraction failed: %s", detail)
        raise UpdateError(f"could not extract the update: {detail}")
    app_path = destination / APP_BUNDLE_NAME
    if not app_path.is_dir():
        raise UpdateError(f"{APP_BUNDLE_NAME} not found in the downloaded archive")
    return app_path


# --- verification -------------------------------------------------------------


def parse_team_identifier(codesign_output: str) -> str | None:
    for line in codesign_output.splitlines():
        if line.startswith("TeamIdentifier="):
            value = line.split("=", 1)[1].strip()
            if value and value.lower() != "not set":
                return value
    return None


def team_identifier(app_path: Path, *, runner: Runner = run_command) -> str | None:
    result = _run(runner, ["/usr/bin/codesign", "-dv", str(app_path)])
    if result is None or result.returncode != 0:
        return None
    # codesign -d prints its details on stderr.
    return parse_team_identifier(f"{result.stderr or ''}\n{result.stdout or ''}")


def evaluate_verification(facts: VerificationFacts) -> VerificationResult:
    """Decide whether a downloaded app may replace the running one."""
    failures: list[str] = []
    if not facts.codesign_ok:
        failures.append("codesign verification failed")
    if not facts.spctl_ok:
        failures.append("Gatekeeper assessment failed")
    if facts.current_team_id is None:
        failures.append("running app has no Team ID (ad-hoc signed)")
    elif facts.new_team_id != facts.current_team_id:
        failures.append(
            f"Team ID mismatch ({facts.new_team_id or 'none'} != {facts.current_team_id})"
        )
    if not facts.current_bundle_id or facts.new_bundle_id != facts.current_bundle_id:
        failures.append(
            f"bundle identifier mismatch ({facts.new_bundle_id} != {facts.current_bundle_id})"
        )
    if facts.new_version != facts.expected_version:
        failures.append(f"version mismatch ({facts.new_version} != {facts.expected_version})")
    return VerificationResult(ok=not failures, failures=tuple(failures))


def collect_verification_facts(
    new_app: Path,
    current_app: Path,
    expected_version: str,
    *,
    runner: Runner = run_command,
) -> VerificationFacts:
    codesign = _run(runner, ["/usr/bin/codesign", "--verify", "--deep", "--strict", str(new_app)])
    spctl = _run(runner, ["/usr/sbin/spctl", "-a", "-t", "exec", str(new_app)])
    new_info = read_bundle_info(new_app)
    current_info = read_bundle_info(current_app)
    return VerificationFacts(
        codesign_ok=codesign is not None and codesign.returncode == 0,
        spctl_ok=spctl is not None and spctl.returncode == 0,
        new_team_id=team_identifier(new_app, runner=runner),
        current_team_id=team_identifier(current_app, runner=runner),
        new_bundle_id=new_info.get("CFBundleIdentifier"),
        current_bundle_id=current_info.get("CFBundleIdentifier"),
        new_version=new_info.get("CFBundleShortVersionString"),
        expected_version=normalize_version(expected_version),
    )


def verify_update(
    new_app: Path,
    current_app: Path,
    expected_version: str,
    *,
    runner: Runner = run_command,
) -> VerificationResult:
    facts = collect_verification_facts(new_app, current_app, expected_version, runner=runner)
    LOGGER.info(
        "Update verification facts: codesign ok %s, spctl ok %s, team id new %s / current %s, "
        "bundle id new %s / current %s, version new %s / expected %s",
        facts.codesign_ok,
        facts.spctl_ok,
        facts.new_team_id,
        facts.current_team_id,
        facts.new_bundle_id,
        facts.current_bundle_id,
        facts.new_version,
        facts.expected_version,
    )
    result = evaluate_verification(facts)
    if result.ok:
        LOGGER.info("Update verification passed for %s", new_app)
    else:
        LOGGER.warning("Update verification failed: %s", "; ".join(result.failures))
    return result


def prepare_update(
    result: UpdateCheckResult,
    current_app: Path,
    *,
    urlopen: UrlOpen | None = None,
    runner: Runner = run_command,
    progress: ProgressCallback | None = None,
    on_verify: Callable[[], None] | None = None,
    make_work_dir: Callable[[], Path] | None = None,
) -> PreparedUpdate:
    """Download, extract and verify the update. The work dir is removed on failure."""
    if not result.can_install or result.release is None or result.asset is None:
        raise UpdateError("no installable update")
    work_dir = (
        make_work_dir()
        if make_work_dir is not None
        else Path(tempfile.mkdtemp(prefix="ptarmigan-flow-update-"))
    )
    try:
        zip_path = download_asset(
            result.asset,
            work_dir / result.asset.name,
            urlopen=urlopen,
            progress=progress,
        )
        if on_verify is not None:
            on_verify()
        new_app = extract_app(zip_path, work_dir / "extracted", runner=runner)
        zip_path.unlink(missing_ok=True)
        verification = verify_update(new_app, current_app, result.release.version, runner=runner)
        if not verification.ok:
            raise VerificationError(
                "verification failed: " + "; ".join(verification.failures)
            )
    except BaseException:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    return PreparedUpdate(new_app=new_app, work_dir=work_dir, version=result.release.version)


def discard_prepared_update(prepared: PreparedUpdate) -> None:
    shutil.rmtree(prepared.work_dir, ignore_errors=True)


# --- install helper -----------------------------------------------------------


def backup_path_for(bundle_path: Path) -> Path:
    return bundle_path.parent / f".{bundle_path.name}.update-backup"


def build_install_script(
    *,
    pid: int,
    current_app: Path,
    new_app: Path,
    work_dir: Path,
    log_path: Path,
    wait_seconds: int = 60,
    relaunch: bool = True,
) -> str:
    """Return a POSIX shell script that swaps the bundle once ``pid`` exits.

    With ``relaunch=False`` (the app is quitting) the helper never opens the app,
    not even the restored old copy after a failure.
    """
    target = shlex.quote(str(current_app))
    backup = shlex.quote(str(backup_path_for(current_app)))
    q_new = shlex.quote(str(new_app))
    q_work = shlex.quote(str(work_dir))
    q_log = shlex.quote(str(log_path))
    polls = max(1, int(wait_seconds) * 10)
    return f"""#!/bin/sh
# PtarmiganFlow update helper (generated).
set -u
PID={int(pid)}
TARGET={target}
BACKUP={backup}
NEW_APP={q_new}
WORK_DIR={q_work}
RELAUNCH={1 if relaunch else 0}
relaunch_app() {{
  if [ "$RELAUNCH" = 1 ]; then
    open "$TARGET"
  fi
}}
mkdir -p "$(dirname {q_log})" 2>/dev/null
exec >>{q_log} 2>&1
echo "$(date) update helper: waiting for pid $PID (relaunch $RELAUNCH)"
i=0
while kill -0 "$PID" 2>/dev/null; do
  i=$((i + 1))
  if [ "$i" -gt {polls} ]; then
    echo "$(date) app did not exit; update aborted"
    rm -rf "$WORK_DIR"
    exit 1
  fi
  sleep 0.1
done
rm -rf "$BACKUP"
if ! mv "$TARGET" "$BACKUP"; then
  echo "$(date) could not move the old app aside; update aborted"
  rm -rf "$WORK_DIR"
  relaunch_app
  exit 1
fi
if /usr/bin/ditto "$NEW_APP" "$TARGET"; then
  echo "$(date) update installed"
  rm -rf "$BACKUP"
  rm -rf "$WORK_DIR"
  relaunch_app
  exit 0
fi
echo "$(date) copying the new app failed; restoring the old app"
rm -rf "$TARGET"
mv "$BACKUP" "$TARGET"
rm -rf "$WORK_DIR"
relaunch_app
exit 1
"""


def launch_install_helper(
    prepared: PreparedUpdate,
    current_app: Path,
    *,
    pid: int | None = None,
    log_path: Path,
    popen: Callable[..., Any] = subprocess.Popen,
    relaunch: bool = True,
) -> Path:
    """Write the helper script into the work dir and start it detached."""
    script = build_install_script(
        pid=os.getpid() if pid is None else pid,
        current_app=current_app,
        new_app=prepared.new_app,
        work_dir=prepared.work_dir,
        log_path=log_path,
        relaunch=relaunch,
    )
    script_path = prepared.work_dir / "install-update.sh"
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o700)
    try:
        popen(
            ["/bin/sh", str(script_path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
    except OSError as exc:
        LOGGER.warning("Could not start the update helper %s", script_path, exc_info=True)
        raise UpdateError(f"could not start the update helper: {exc}") from exc
    LOGGER.info(
        "Update helper launched: script %s, target %s, new app %s, relaunch %s, log %s",
        script_path,
        current_app,
        prepared.new_app,
        relaunch,
        log_path,
    )
    return script_path


def running_app_version(bundle_path: Path | None) -> str:
    """Return the running app's ``CFBundleShortVersionString`` or the package version."""
    if bundle_path is not None:
        version = read_bundle_info(bundle_path).get("CFBundleShortVersionString")
        if isinstance(version, str) and version:
            return version
    try:
        from importlib.metadata import version as dist_version

        return dist_version("ptarmigan-flow")
    except Exception:
        return "0.0.0-dev"

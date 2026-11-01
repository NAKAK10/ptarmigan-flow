from __future__ import annotations

import io
import json
import logging
import plistlib
import shlex
import subprocess
import urllib.error
from pathlib import Path

import pytest

from ptarmigan_flow import app_updater
from ptarmigan_flow.app_updater import (
    ReleaseAsset,
    ReleaseInfo,
    UpdateCheckResult,
    UpdateError,
    VerificationFacts,
)

TEAM_ID = "35RRH2NH88"
BUNDLE_ID = "com.ptarmiganflow.app"


def _release_payload(**overrides) -> dict:
    payload = {
        "tag_name": "v0.3.22",
        "html_url": "https://github.com/NAKAK10/ptarmigan-flow/releases/tag/v0.3.22",
        "draft": False,
        "prerelease": False,
        "body": "notes",
        "assets": [
            {
                "name": "PtarmiganFlow-macos-arm64.zip",
                "size": 10,
                "browser_download_url": "https://example.com/arm64.zip",
            },
            {
                "name": "PtarmiganFlow-macos-x86_64.zip",
                "size": 20,
                "browser_download_url": "https://example.com/x86_64.zip",
            },
        ],
    }
    payload.update(overrides)
    return payload


def _release(version: str = "0.3.22", assets=None) -> ReleaseInfo:
    if assets is None:
        assets = (
            ReleaseAsset("PtarmiganFlow-macos-arm64.zip", "https://example.com/arm64.zip", 10),
        )
    return ReleaseInfo(
        tag=f"v{version}",
        version=version,
        html_url=f"https://github.com/NAKAK10/ptarmigan-flow/releases/tag/v{version}",
        assets=tuple(assets),
    )


class _FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, headers: dict[str, str] | None = None) -> None:
        super().__init__(data)
        self.headers = headers or {}


def _write_app(path: Path, *, bundle_id: str = BUNDLE_ID, version: str = "0.3.22") -> Path:
    contents = path / "Contents"
    contents.mkdir(parents=True)
    with (contents / "Info.plist").open("wb") as handle:
        plistlib.dump(
            {"CFBundleIdentifier": bundle_id, "CFBundleShortVersionString": version},
            handle,
        )
    return path


# --- versions -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0.3.21", (0, 3, 21)),
        ("v1.2.3", (1, 2, 3)),
        (" 10.0.1 ", (10, 0, 1)),
        ("0.0.0-dev", None),
        ("0.3.21.dev0", None),
        ("0.3", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_release_version(text, expected) -> None:
    assert app_updater.parse_release_version(text) == expected


def test_is_release_version_rejects_dev_and_placeholder_versions() -> None:
    assert app_updater.is_release_version("0.3.21")
    assert not app_updater.is_release_version("0.0.0-dev")
    assert not app_updater.is_release_version("0.0.0")
    assert not app_updater.is_release_version("0.3.21+local")


def test_compare_versions_is_numeric_not_lexical() -> None:
    assert app_updater.compare_versions("0.3.10", "0.3.9") == 1
    assert app_updater.compare_versions("v0.3.21", "0.3.21") == 0
    assert app_updater.compare_versions("0.3.21", "0.4.0") == -1
    with pytest.raises(ValueError):
        app_updater.compare_versions("0.0.0-dev", "0.3.21")


def test_is_newer_version() -> None:
    assert app_updater.is_newer_version("0.3.22", "0.3.21")
    assert not app_updater.is_newer_version("0.3.21", "0.3.21")
    assert not app_updater.is_newer_version("0.3.20", "0.3.21")
    assert app_updater.is_newer_version("0.3.22", "0.0.0-dev")
    assert not app_updater.is_newer_version("garbage", "0.3.21")


# --- releases -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("machine", "expected"),
    [("arm64", "arm64"), ("aarch64", "arm64"), ("x86_64", "x86_64"), ("AMD64", "x86_64"),
     ("ppc", None)],
)
def test_normalize_arch(machine, expected) -> None:
    assert app_updater.normalize_arch(machine) == expected


def test_select_asset_by_arch() -> None:
    release = app_updater.parse_release(_release_payload())

    assert app_updater.select_asset(release, "arm64").url == "https://example.com/arm64.zip"
    assert app_updater.select_asset(release, "x86_64").url == "https://example.com/x86_64.zip"
    assert app_updater.select_asset(release, None) is None
    assert app_updater.select_asset(_release(assets=()), "arm64") is None


def test_parse_release_reads_tag_assets_and_page() -> None:
    release = app_updater.parse_release(_release_payload())

    assert release.tag == "v0.3.22"
    assert release.version == "0.3.22"
    assert release.html_url.endswith("/tag/v0.3.22")
    assert [asset.name for asset in release.assets] == [
        "PtarmiganFlow-macos-arm64.zip",
        "PtarmiganFlow-macos-x86_64.zip",
    ]
    assert release.assets[0].size == 10


def test_parse_release_skips_malformed_and_non_https_assets() -> None:
    release = app_updater.parse_release(
        _release_payload(
            assets=[
                {"name": "PtarmiganFlow-macos-arm64.zip", "browser_download_url": "http://x"},
                {"name": 3},
                "junk",
            ]
        )
    )

    assert release.assets == ()


@pytest.mark.parametrize(
    "overrides",
    [{"draft": True}, {"prerelease": True}, {"tag_name": "nightly"}, {"tag_name": None}],
)
def test_parse_release_rejects_drafts_prereleases_and_bad_tags(overrides) -> None:
    with pytest.raises(UpdateError):
        app_updater.parse_release(_release_payload(**overrides))


def test_fetch_latest_release_sends_user_agent_and_timeout() -> None:
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["agent"] = request.get_header("User-agent")
        seen["timeout"] = timeout
        return _FakeResponse(json.dumps(_release_payload()).encode())

    release = app_updater.fetch_latest_release(urlopen=fake_urlopen)

    assert release.version == "0.3.22"
    assert seen["url"] == app_updater.LATEST_RELEASE_API_URL
    assert seen["agent"] == app_updater.USER_AGENT
    assert seen["timeout"] == app_updater.REQUEST_TIMEOUT_SECONDS


def test_fetch_latest_release_wraps_network_errors() -> None:
    def failing_urlopen(_request, timeout):
        raise urllib.error.URLError("offline")

    with pytest.raises(UpdateError, match="offline"):
        app_updater.fetch_latest_release(urlopen=failing_urlopen)


# --- check_for_update ---------------------------------------------------------


def _check(current: str, **kwargs) -> UpdateCheckResult:
    defaults = {
        "frozen": True,
        "bundle_path": Path("/Applications/PtarmiganFlow.app"),
        "arch": "arm64",
        "fetch": lambda: _release(),
        "is_writable": lambda _path: True,
    }
    defaults.update(kwargs)
    return app_updater.check_for_update(current, **defaults)


def test_check_for_update_reports_installable_update() -> None:
    result = _check("0.3.21")

    assert result.update_available
    assert result.can_install
    assert result.asset.name == "PtarmiganFlow-macos-arm64.zip"
    assert result.release_page_url.endswith("/tag/v0.3.22")


def test_check_for_update_up_to_date() -> None:
    result = _check("0.3.22")

    assert not result.update_available
    assert not result.can_install
    assert result.error is None


def test_check_for_update_reports_fetch_error() -> None:
    def fail():
        raise UpdateError("boom")

    result = _check("0.3.21", fetch=fail)

    assert result.error == "boom"
    assert not result.update_available


@pytest.mark.parametrize(
    ("kwargs", "current", "blocker"),
    [
        ({"frozen": False}, "0.3.21", app_updater.INSTALL_BLOCK_NOT_FROZEN),
        ({}, "0.0.0-dev", app_updater.INSTALL_BLOCK_DEV_BUILD),
        ({"bundle_path": None}, "0.3.21", app_updater.INSTALL_BLOCK_NO_BUNDLE),
        (
            {"bundle_path": Path("/private/var/folders/x/AppTranslocation/y/d/PtarmiganFlow.app")},
            "0.3.21",
            app_updater.INSTALL_BLOCK_TRANSLOCATED,
        ),
        ({"is_writable": lambda _path: False}, "0.3.21", app_updater.INSTALL_BLOCK_NOT_WRITABLE),
        ({"arch": "x86_64"}, "0.3.21", app_updater.INSTALL_BLOCK_NO_ASSET),
    ],
)
def test_check_for_update_install_blockers(kwargs, current, blocker) -> None:
    result = _check(current, **kwargs)

    assert result.update_available
    assert result.install_blocker == blocker
    assert not result.can_install


def test_check_for_update_logs_versions_and_blocker(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="ptarmigan_flow.app_updater"):
        _check("0.3.21", is_writable=lambda _path: False)
        _check("0.3.22")

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "current 0.3.21, latest 0.3.22, update available: yes" in m
        and f"install blocker {app_updater.INSTALL_BLOCK_NOT_WRITABLE}" in m
        for m in messages
    ), messages
    assert any("current 0.3.22, latest 0.3.22, update available: no" in m for m in messages)


def test_check_for_update_logs_fetch_error(caplog) -> None:
    def fail():
        raise UpdateError("boom")

    with caplog.at_level(logging.INFO, logger="ptarmigan_flow.app_updater"):
        _check("0.3.21", fetch=fail)

    assert any(
        record.levelno == logging.WARNING and "boom" in record.getMessage()
        for record in caplog.records
    )


def test_bundle_location_is_writable_logs_result(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="ptarmigan_flow.app_updater"):
        app_updater.bundle_location_is_writable(
            Path("/Applications/PtarmiganFlow.app"), access=lambda _path, _mode: False
        )

    assert any(
        "/Applications/PtarmiganFlow.app writable: False" in record.getMessage()
        for record in caplog.records
    )


def test_dev_build_result_is_marked_dev() -> None:
    assert _check("0.0.0-dev").is_dev_build
    assert not _check("0.3.21").is_dev_build


def test_bundle_location_is_writable_requires_parent_and_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "PtarmiganFlow.app"
    calls: list[str] = []

    def access(path: str, _mode: int) -> bool:
        calls.append(path)
        return path != str(bundle)

    assert not app_updater.bundle_location_is_writable(bundle, access=access)
    assert calls == [str(tmp_path), str(bundle)]
    assert app_updater.bundle_location_is_writable(bundle, access=lambda _p, _m: True)


# --- download / extract -------------------------------------------------------


def test_download_asset_streams_and_reports_progress(tmp_path: Path) -> None:
    data = b"x" * 25
    progress: list[tuple[int, int | None]] = []

    def fake_urlopen(request, timeout):
        assert request.get_header("User-agent") == app_updater.USER_AGENT
        return _FakeResponse(data, {"Content-Length": "25"})

    asset = ReleaseAsset("a.zip", "https://example.com/a.zip", 25)
    out = app_updater.download_asset(
        asset,
        tmp_path / "a.zip",
        urlopen=fake_urlopen,
        progress=lambda done, total: progress.append((done, total)),
        chunk_size=10,
    )

    assert out.read_bytes() == data
    assert progress == [(10, 25), (20, 25), (25, 25)]


def test_download_asset_rejects_truncated_download(tmp_path: Path) -> None:
    asset = ReleaseAsset("a.zip", "https://example.com/a.zip", 100)

    with pytest.raises(UpdateError, match="incomplete"):
        app_updater.download_asset(
            asset,
            tmp_path / "a.zip",
            urlopen=lambda _request, timeout: _FakeResponse(b"abc"),
        )


def test_extract_app_uses_ditto_and_requires_bundle(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(argv):
        calls.append(argv)
        (Path(argv[-1]) / "PtarmiganFlow.app").mkdir(parents=True)
        return subprocess.CompletedProcess(argv, 0, "", "")

    app = app_updater.extract_app(tmp_path / "a.zip", tmp_path / "out", runner=runner)

    assert app == tmp_path / "out" / "PtarmiganFlow.app"
    assert calls == [
        ["/usr/bin/ditto", "-x", "-k", str(tmp_path / "a.zip"), str(tmp_path / "out")]
    ]

    with pytest.raises(UpdateError, match="not found"):
        app_updater.extract_app(
            tmp_path / "a.zip",
            tmp_path / "empty",
            runner=lambda argv: subprocess.CompletedProcess(argv, 0, "", ""),
        )
    with pytest.raises(UpdateError, match="extract"):
        app_updater.extract_app(
            tmp_path / "a.zip",
            tmp_path / "bad",
            runner=lambda argv: subprocess.CompletedProcess(argv, 1, "", "corrupt"),
        )


# --- verification -------------------------------------------------------------


def test_parse_team_identifier() -> None:
    output = "Executable=/x\nIdentifier=com.ptarmiganflow.app\nTeamIdentifier=35RRH2NH88\n"

    assert app_updater.parse_team_identifier(output) == TEAM_ID
    assert app_updater.parse_team_identifier("TeamIdentifier=not set\n") is None
    assert app_updater.parse_team_identifier("Signature=adhoc\n") is None


def _facts(**overrides) -> VerificationFacts:
    values = {
        "codesign_ok": True,
        "spctl_ok": True,
        "new_team_id": TEAM_ID,
        "current_team_id": TEAM_ID,
        "new_bundle_id": BUNDLE_ID,
        "current_bundle_id": BUNDLE_ID,
        "new_version": "0.3.22",
        "expected_version": "0.3.22",
    }
    values.update(overrides)
    return VerificationFacts(**values)


def test_evaluate_verification_accepts_matching_signed_update() -> None:
    result = app_updater.evaluate_verification(_facts())

    assert result.ok
    assert result.failures == ()


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"codesign_ok": False}, "codesign"),
        ({"spctl_ok": False}, "Gatekeeper"),
        ({"current_team_id": None}, "ad-hoc"),
        ({"new_team_id": "OTHERTEAM1"}, "Team ID mismatch"),
        ({"new_team_id": None}, "Team ID mismatch"),
        ({"new_bundle_id": "com.example.other"}, "bundle identifier"),
        ({"current_bundle_id": None, "new_bundle_id": None}, "bundle identifier"),
        ({"new_version": "0.3.21"}, "version mismatch"),
    ],
)
def test_evaluate_verification_rejects_any_failed_check(overrides, fragment) -> None:
    result = app_updater.evaluate_verification(_facts(**overrides))

    assert not result.ok
    assert any(fragment in failure for failure in result.failures)


def _fake_codesign_runner(team_ids: dict[str, str | None], *, verify_rc=0, spctl_rc=0):
    calls: list[list[str]] = []

    def runner(argv):
        calls.append(argv)
        if argv[:2] == ["/usr/bin/codesign", "--verify"]:
            return subprocess.CompletedProcess(argv, verify_rc, "", "")
        if argv[0] == "/usr/sbin/spctl":
            return subprocess.CompletedProcess(argv, spctl_rc, "", "")
        if argv[:2] == ["/usr/bin/codesign", "-dv"]:
            team = team_ids.get(argv[-1])
            line = f"TeamIdentifier={team}" if team else "TeamIdentifier=not set"
            return subprocess.CompletedProcess(argv, 0, "", f"Identifier={BUNDLE_ID}\n{line}\n")
        raise AssertionError(argv)

    return runner, calls


def test_verify_update_with_fake_subprocess_outputs(tmp_path: Path) -> None:
    new_app = _write_app(tmp_path / "new" / "PtarmiganFlow.app", version="0.3.22")
    current_app = _write_app(tmp_path / "cur" / "PtarmiganFlow.app", version="0.3.21")
    runner, calls = _fake_codesign_runner({str(new_app): TEAM_ID, str(current_app): TEAM_ID})

    result = app_updater.verify_update(new_app, current_app, "v0.3.22", runner=runner)

    assert result.ok, result.failures
    assert ["/usr/bin/codesign", "--verify", "--deep", "--strict", str(new_app)] in calls
    assert ["/usr/sbin/spctl", "-a", "-t", "exec", str(new_app)] in calls


def test_verify_update_refuses_ad_hoc_running_app(tmp_path: Path) -> None:
    new_app = _write_app(tmp_path / "new" / "PtarmiganFlow.app")
    current_app = _write_app(tmp_path / "cur" / "PtarmiganFlow.app", version="0.3.21")
    runner, _calls = _fake_codesign_runner({str(new_app): TEAM_ID, str(current_app): None})

    result = app_updater.verify_update(new_app, current_app, "0.3.22", runner=runner)

    assert not result.ok
    assert any("ad-hoc" in failure for failure in result.failures)


def test_verify_update_logs_each_fact(tmp_path: Path, caplog) -> None:
    new_app = _write_app(tmp_path / "new" / "PtarmiganFlow.app", version="0.3.22")
    current_app = _write_app(tmp_path / "cur" / "PtarmiganFlow.app", version="0.3.21")
    runner, _calls = _fake_codesign_runner(
        {str(new_app): TEAM_ID, str(current_app): TEAM_ID}, spctl_rc=3
    )

    with caplog.at_level(logging.INFO, logger="ptarmigan_flow.app_updater"):
        result = app_updater.verify_update(new_app, current_app, "v0.3.22", runner=runner)

    assert not result.ok
    messages = [record.getMessage() for record in caplog.records]
    facts = next(m for m in messages if m.startswith("Update verification facts"))
    assert "codesign ok True" in facts
    assert "spctl ok False" in facts
    assert f"team id new {TEAM_ID} / current {TEAM_ID}" in facts
    assert f"bundle id new {BUNDLE_ID} / current {BUNDLE_ID}" in facts
    assert "version new 0.3.22 / expected 0.3.22" in facts
    assert any(
        record.levelno == logging.WARNING and "Gatekeeper assessment failed" in record.getMessage()
        for record in caplog.records
    )


def test_verify_update_fails_when_tools_cannot_run(tmp_path: Path) -> None:
    new_app = _write_app(tmp_path / "new" / "PtarmiganFlow.app")
    current_app = _write_app(tmp_path / "cur" / "PtarmiganFlow.app", version="0.3.21")

    def runner(_argv):
        raise OSError("missing tool")

    result = app_updater.verify_update(new_app, current_app, "0.3.22", runner=runner)

    assert not result.ok


def test_prepare_update_downloads_extracts_verifies(tmp_path: Path) -> None:
    current_app = _write_app(tmp_path / "Applications" / "PtarmiganFlow.app", version="0.3.21")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    team_ids = {str(current_app): TEAM_ID}
    verify_runner, _calls = _fake_codesign_runner(team_ids)

    def runner(argv):
        if argv[0] == "/usr/bin/ditto":
            app = _write_app(Path(argv[-1]) / "PtarmiganFlow.app", version="0.3.22")
            team_ids[str(app)] = TEAM_ID
            return subprocess.CompletedProcess(argv, 0, "", "")
        return verify_runner(argv)

    stages: list[str] = []
    prepared = app_updater.prepare_update(
        _check("0.3.21"),
        current_app,
        urlopen=lambda _request, timeout: _FakeResponse(b"z" * 10),
        runner=runner,
        on_verify=lambda: stages.append("verify"),
        make_work_dir=lambda: work_dir,
    )

    assert prepared.new_app == work_dir / "extracted" / "PtarmiganFlow.app"
    assert prepared.version == "0.3.22"
    assert stages == ["verify"]
    assert not (work_dir / "PtarmiganFlow-macos-arm64.zip").exists()


def test_prepare_update_removes_work_dir_on_verification_failure(tmp_path: Path) -> None:
    current_app = _write_app(tmp_path / "Applications" / "PtarmiganFlow.app", version="0.3.21")
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    def runner(argv):
        if argv[0] == "/usr/bin/ditto":
            _write_app(Path(argv[-1]) / "PtarmiganFlow.app", version="0.3.22")
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 1, "", "invalid signature")

    with pytest.raises(UpdateError, match="verification failed"):
        app_updater.prepare_update(
            _check("0.3.21"),
            current_app,
            urlopen=lambda _request, timeout: _FakeResponse(b"z" * 10),
            runner=runner,
            make_work_dir=lambda: work_dir,
        )

    assert not work_dir.exists()


def test_prepare_update_refuses_non_installable_result(tmp_path: Path) -> None:
    with pytest.raises(UpdateError):
        app_updater.prepare_update(_check("0.0.0-dev"), tmp_path / "PtarmiganFlow.app")


# --- install helper -----------------------------------------------------------


def test_build_install_script_quotes_paths_and_swaps_bundle(tmp_path: Path) -> None:
    current_app = Path("/Users/me/My Apps/PtarmiganFlow.app")
    new_app = Path("/tmp/it's here/extracted/PtarmiganFlow.app")
    work_dir = Path("/tmp/it's here")
    log_path = Path("/Users/me/Library/Logs/ptarmigan-flow/update.log")

    script = app_updater.build_install_script(
        pid=4242,
        current_app=current_app,
        new_app=new_app,
        work_dir=work_dir,
        log_path=log_path,
    )

    assert script.startswith("#!/bin/sh\n")
    assert "PID=4242\n" in script
    assert f"TARGET={shlex.quote(str(current_app))}\n" in script
    assert f"NEW_APP={shlex.quote(str(new_app))}\n" in script
    assert f"WORK_DIR={shlex.quote(str(work_dir))}\n" in script
    backup = app_updater.backup_path_for(current_app)
    assert backup == Path("/Users/me/My Apps/.PtarmiganFlow.app.update-backup")
    assert f"BACKUP={shlex.quote(str(backup))}\n" in script
    assert 'kill -0 "$PID"' in script
    assert 'mv "$TARGET" "$BACKUP"' in script
    assert '/usr/bin/ditto "$NEW_APP" "$TARGET"' in script
    assert 'mv "$BACKUP" "$TARGET"' in script
    assert 'open "$TARGET"' in script
    # The unquoted path with a space must never appear as a bare word.
    assert "TARGET=/Users/me/My Apps" not in script


def test_build_install_script_is_valid_shell(tmp_path: Path) -> None:
    script = app_updater.build_install_script(
        pid=1,
        current_app=Path("/Applications/PtarmiganFlow.app"),
        new_app=tmp_path / "PtarmiganFlow.app",
        work_dir=tmp_path,
        log_path=tmp_path / "update.log",
    )
    path = tmp_path / "helper.sh"
    path.write_text(script)

    result = subprocess.run(["/bin/sh", "-n", str(path)], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def _run_helper(tmp_path: Path, *, ditto_fails: bool) -> Path:
    """Run the generated helper against a sandbox bundle with stubbed ditto/open."""
    target = _write_app(tmp_path / "Apps" / "PtarmiganFlow.app", version="0.3.21")
    work_dir = tmp_path / "work"
    new_app = _write_app(work_dir / "extracted" / "PtarmiganFlow.app", version="0.3.22")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    opened = tmp_path / "opened.txt"
    (bin_dir / "open").write_text(f'#!/bin/sh\necho "$1" >> {shlex.quote(str(opened))}\n')
    (bin_dir / "open").chmod(0o755)
    script = app_updater.build_install_script(
        pid=999999,
        current_app=target,
        new_app=new_app,
        work_dir=work_dir,
        log_path=tmp_path / "logs" / "update.log",
    )
    fake_ditto = "false" if ditto_fails else "cp -R"
    script = script.replace("/usr/bin/ditto", fake_ditto)
    helper = tmp_path / "helper.sh"
    helper.write_text(script)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin"}
    subprocess.run(["/bin/sh", str(helper)], env=env, check=False, timeout=30)
    assert opened.read_text().strip() == str(target)
    assert not app_updater.backup_path_for(target).exists()
    assert not work_dir.exists()
    return target


def test_install_helper_replaces_bundle_and_relaunches(tmp_path: Path) -> None:
    target = _run_helper(tmp_path, ditto_fails=False)

    assert app_updater.read_bundle_info(target)["CFBundleShortVersionString"] == "0.3.22"


def test_install_helper_restores_old_bundle_when_copy_fails(tmp_path: Path) -> None:
    target = _run_helper(tmp_path, ditto_fails=True)

    assert app_updater.read_bundle_info(target)["CFBundleShortVersionString"] == "0.3.21"


def test_launch_install_helper_writes_script_and_detaches(tmp_path: Path) -> None:
    prepared = app_updater.PreparedUpdate(
        new_app=tmp_path / "extracted" / "PtarmiganFlow.app",
        work_dir=tmp_path,
        version="0.3.22",
    )
    calls: list[tuple[list[str], dict]] = []

    script_path = app_updater.launch_install_helper(
        prepared,
        Path("/Applications/PtarmiganFlow.app"),
        pid=123,
        log_path=tmp_path / "update.log",
        popen=lambda argv, **kwargs: calls.append((argv, kwargs)),
    )

    assert script_path == tmp_path / "install-update.sh"
    assert "PID=123" in script_path.read_text()
    assert calls[0][0] == ["/bin/sh", str(script_path)]
    assert calls[0][1]["start_new_session"] is True


def test_launch_install_helper_wraps_spawn_errors(tmp_path: Path) -> None:
    prepared = app_updater.PreparedUpdate(tmp_path / "x.app", tmp_path, "0.3.22")

    def failing_popen(*_args, **_kwargs):
        raise OSError("no sh")

    with pytest.raises(UpdateError, match="no sh"):
        app_updater.launch_install_helper(
            prepared,
            Path("/Applications/PtarmiganFlow.app"),
            log_path=tmp_path / "update.log",
            popen=failing_popen,
        )


def test_running_app_version_reads_bundle_plist(tmp_path: Path) -> None:
    bundle = _write_app(tmp_path / "PtarmiganFlow.app", version="0.3.21")

    assert app_updater.running_app_version(bundle) == "0.3.21"

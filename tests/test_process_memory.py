from __future__ import annotations

import subprocess
from unittest.mock import Mock

import pytest

from ptarmigan_flow import process_memory


@pytest.fixture
def ps(monkeypatch):
    monkeypatch.setattr(process_memory.sys, "platform", "darwin")
    monkeypatch.setattr(process_memory.os, "getpid", lambda: 100)
    process = Mock(pid=999, returncode=0)
    process.communicate.return_value = ("100 1 7\n", None)
    popen = Mock(return_value=process)
    monkeypatch.setattr(process_memory.subprocess, "Popen", popen)
    return popen, process


def test_current_rss_is_kib_converted_to_bytes_and_can_decrease(ps):
    popen, process = ps
    process.communicate.side_effect = [("100 1 2048\n", None), ("100 1 512\n", None)]

    assert process_memory.sample_process_memory() == 2048 * 1024
    assert process_memory.sample_process_memory() == 512 * 1024
    popen.assert_called_with(
        ["/bin/ps", "-A", "-o", "pid=,ppid=,rss="],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="ascii",
    )
    process.communicate.assert_called_with(timeout=1.0)
    process.kill.assert_not_called()


def test_sums_out_of_order_descendants_excluding_parent_peers_and_sampler(ps):
    _, process = ps
    process.communicate.return_value = (
        "300 200 30\n"  # Grandchild of the speech.cpp child.
        "101 1 800\n"  # Unrelated peer.
        "999 100 900\n"  # The sampler's ps process.
        "998 999 900\n"  # Never follow the sampler's subtree.
        "200 100 20\n"  # speech.cpp child.
        "400 100 40\n"  # Overlay child.
        "1 0 1000\n"  # Parent of the app.
        "0 0 0\n"  # Unrelated kernel row.
        "100 1 10\n",
        None,
    )

    assert process_memory.sample_process_memory() == 100 * 1024
    assert process_memory.sample_process_memory(200) == 50 * 1024


def test_exited_child_is_not_retained_between_samples(ps):
    _, process = ps
    process.communicate.side_effect = [
        ("100 1 10\n200 100 20\n", None),
        ("100 1 10\n300 200 30\n", None),
    ]

    assert process_memory.sample_process_memory() == 30 * 1024
    assert process_memory.sample_process_memory() == 10 * 1024


def test_identical_duplicate_rows_are_counted_once(ps):
    _, process = ps
    process.communicate.return_value = ("100 1 7\n200 100 3\n200 100 3\n", None)

    assert process_memory.sample_process_memory() == 10 * 1024


@pytest.mark.parametrize(
    "output",
    [
        "",
        "\n  \n",
        "200 1 7\n",  # Root exited.
        "100 1 7\n200 100\n",
        "100 1 7 extra\n",
        "PID PPID RSS\n",
        "100 1 unknown\n",
        "100 1 -1\n",
        "100 -1 7\n",
        "-100 1 7\n",
        "100 1 1.5\n",
        "100 1 1_000\n",
        "100 1 ７\n",
        "100 1 7\n100 1 8\n",  # Conflicting RSS.
        "100 1 7\n100 2 7\n",  # Conflicting parent.
        "100 200 7\n200 100 3\n",  # Reachable cycle.
        "100 100 7\n",  # Self cycle.
    ],
)
def test_unavailable_or_malformed_snapshot_returns_none(ps, output):
    _, process = ps
    process.communicate.return_value = (output, None)

    assert process_memory.sample_process_memory() is None


def test_unrelated_cycle_does_not_affect_sample(ps):
    _, process = ps
    process.communicate.return_value = ("100 1 7\n200 300 3\n300 200 4\n", None)

    assert process_memory.sample_process_memory() == 7 * 1024


def test_zero_rss_is_a_valid_sample(ps):
    _, process = ps
    process.communicate.return_value = ("\n 100 1 0 \n\n", None)

    assert process_memory.sample_process_memory() == 0


@pytest.mark.parametrize("pid", [0, -1, False, True, "100", 1.5])
def test_invalid_root_returns_none_without_launching_ps(ps, pid):
    popen, _ = ps

    assert process_memory.sample_process_memory(pid) is None
    popen.assert_not_called()


def test_sampler_cannot_be_the_requested_root(ps):
    _, process = ps
    process.communicate.return_value = ("999 100 7\n", None)

    assert process_memory.sample_process_memory(999) is None


def test_other_platform_is_unavailable(ps, monkeypatch):
    popen, _ = ps
    monkeypatch.setattr(process_memory.sys, "platform", "linux")

    assert process_memory.sample_process_memory() is None
    popen.assert_not_called()


@pytest.mark.parametrize("error", [FileNotFoundError(), PermissionError(), OSError()])
def test_ps_launch_failure_returns_none(ps, error):
    popen, _ = ps
    popen.side_effect = error

    assert process_memory.sample_process_memory() is None


def test_nonzero_exit_returns_none_even_with_partial_output(ps):
    _, process = ps
    process.returncode = 1

    assert process_memory.sample_process_memory() is None


@pytest.mark.parametrize(
    "error",
    [
        subprocess.TimeoutExpired("/bin/ps", 1),
        OSError("read failed"),
        UnicodeDecodeError("ascii", b"\xff", 0, 1, "invalid byte"),
    ],
)
def test_read_failure_kills_and_reaps_with_a_timeout(ps, error):
    _, process = ps
    process.communicate.side_effect = [error, ("", None)]

    assert process_memory.sample_process_memory() is None
    process.kill.assert_called_once_with()
    assert process.communicate.call_count == 2
    for call in process.communicate.call_args_list:
        assert call.kwargs == {"timeout": 1.0}
    process.stdout.close.assert_called_once_with()


@pytest.mark.parametrize("kill_error", [None, ProcessLookupError(), PermissionError()])
def test_cleanup_failure_and_repeated_timeout_remain_bounded(ps, kill_error):
    _, process = ps
    process.communicate.side_effect = subprocess.TimeoutExpired("/bin/ps", 1)
    process.kill.side_effect = kill_error

    assert process_memory.sample_process_memory() is None
    process.kill.assert_called_once_with()
    assert process.communicate.call_count == 2
    assert all(call.kwargs == {"timeout": 1.0} for call in process.communicate.call_args_list)
    process.stdout.close.assert_called_once_with()

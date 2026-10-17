"""Tests for rt-run (src/rt_run.py), without a container runtime or an image.

The stand-ins are in rt_run_fakes.py. A stub `apptainer` takes the real one's
place: `pull` writes a small JSON file
as the image, the scan prints a fixture (or, for one test, runs rt-run's real
scan code against a fake PATH), and any other `exec` prints the arguments it
was given, one per line -- which is what the bind-mount assertions read.
A fake registry answers the digest lookups. build-test-push.yml runs rt-run
for real against every image it builds.

    pip install . pytest && pytest tests/test_rt_run.py
"""

# One suite for one module, as rt_run.py is one file on purpose.
# pylint: disable=too-many-lines

import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from rt_run_fakes import E, HERE, REGISTRY, RT_RUN
from rt_run_fakes import FakeRegistry, World, fake_digest
import rt_run  # pylint: disable=wrong-import-order  # importable once rt_run_fakes ran

# What install writes to RT_HOME/rt-run.
INSTALLED = rt_run.as_launcher(RT_RUN.read_text())

LINKED = ["heimdall", "jupyter", "your_header.py"]
NOT_LINKED = [
    *("python", "python3.10", "python3-config", "pip", "pylint", "xz"),
    *("conda", "psrdada.pc", "nvcc", "ls", "sh"),
]


@pytest.fixture(name="w")
def world():
    # Singularity mounts /tmp, $HOME and $PWD itself, so rt-run rightly binds
    # nothing under them. The bind tests need paths it does have to bind, so
    # the scratch area lives next to this file, with $HOME inside it.
    root = Path(tempfile.mkdtemp(prefix=".rt-run-test.", dir=str(HERE)))
    if str(root).startswith(("/tmp/", "/var/tmp/")):
        shutil.rmtree(str(root))
        pytest.skip("the checkout is under /tmp, which Singularity always mounts")
    # Files the tests make stand for a user's own: private, whatever this
    # machine's umask. Tests of the permission checks loosen them on purpose.
    umask = os.umask(0o022)
    registry = FakeRegistry()
    yield World(root, registry)
    registry.stop()
    os.umask(umask)
    shutil.rmtree(str(root))


@pytest.fixture(name="full")
def installed_full(w):
    result = w.rt_run("install", "full")
    assert result.rc == 0, result.out
    return w


# ---------------------------------------------------------------------------
# Before anything is installed
# ---------------------------------------------------------------------------


def test_running_a_tool_asks_for_an_install_first(w):
    result = w.rt_run("heimdall", "-h")
    assert result.rc == 1 and "rt-run install" in result.out


def test_an_unknown_variant_is_refused_with_status_2(w):
    result = w.rt_run("install", "nonsense")
    assert result.rc == 2 and "unknown variant" in result.out


def test_the_wrong_architecture_is_refused(w):
    if platform.machine() == "aarch64":
        result = w.rt_run("install", "cpu")
        assert result.rc == 1 and "only the arm variant" in result.out
    else:
        result = w.rt_run("install", "arm")
        assert result.rc == 1 and "for aarch64" in result.out
    assert not (w.rt / "image.sif").exists() and not w.pulls()


def test_commands_that_take_no_arguments_refuse_them(w):
    assert w.rt_run("list", "extra").rc == 2


def test_version_and_help(w):
    assert w.rt_run("--version").out.strip() == f"rt-run {rt_run.__version__}"
    assert "VARIANT is one of" in w.rt_run("--help").out


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------


def test_install_pulls_the_tag_and_records_it(full):
    digest = full.registry.digests["latest"]
    assert full.pulls() == [f"{REGISTRY}@{digest}"]  # what was resolved, exactly
    assert (full.rt / "digest").read_text().strip() == digest
    assert (full.rt / "image.sif").is_file()
    assert not (full.rt / "image.sif.new").exists()
    assert (full.rt / "tag").read_text().strip() == "latest"
    assert (full.rt / "built").read_text().strip() == "Monday_7_September_2026"


def test_install_copies_rt_run_so_links_outlive_the_checkout(full):
    script = full.rt / "rt-run"
    assert script.is_file() and not script.is_symlink()
    assert os.access(str(script), os.X_OK)
    assert script.read_text() == INSTALLED


def test_the_installed_copy_is_still_the_module(full):
    """The sh header is only a string to python: same code below it."""
    installed = (full.rt / "rt-run").read_text()
    assert installed.startswith("#!/bin/sh\n")
    head = len(rt_run.LAUNCHER)
    assert installed[head:] == RT_RUN.read_text().split("\n", 1)[1]
    assert rt_run.as_launcher(installed) == installed  # not stacked twice


def test_install_says_how_to_put_the_links_on_path(w):
    result = w.rt_run("install", "full")
    assert f'export PATH="$PATH:{w.bin}"' in result.out
    assert "3 tools linked" in result.out


def test_install_links_the_tools_and_nothing_else(full):
    assert (full.rt / "tools").read_text().split() == LINKED
    for name in LINKED + ["rt-run"]:
        assert full.ours(full.bin / name), name
    for name in NOT_LINKED:
        assert not (full.bin / name).exists(), name


def test_every_command_in_the_image_is_known(full):
    commands = (full.rt / "commands").read_text().split()
    assert {"python", "ls", "nvcc", "psrdada.pc"} <= set(commands)
    assert commands.count("your_header.py") == 1


def test_list_shows_the_linked_tools(full):
    result = full.tool("rt-run", "list")
    assert result.rc == 0 and "heimdall" in result.out and "nvcc" not in result.out
    assert "a name with a command" not in result.out


WHEN = r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d UTC"


def history(w):
    lines = (w.rt / "history").read_text().splitlines()
    return [line.split("\t") for line in lines]


def test_list_says_when_the_image_was_pulled_and_from_where(full):
    out = full.tool("rt-run", "list").out
    assert "  built      Monday_7_September_2026\n" in out
    assert re.search(rf"  pulled     {WHEN}, from {re.escape(REGISTRY)}:latest\n", out)
    assert f"  history    {full.rt / 'history'}\n" in out


def test_history_keeps_a_line_per_install_and_update(full):
    first = full.registry.digests["latest"]
    second = full.registry.move()
    full.built("Monday_14_September_2026")
    assert full.tool("rt-run", "update").rc == 0
    rows = history(full)
    assert [row[1] for row in rows] == ["install", "update"]
    assert rows[0][2:] == [
        "radio_transients:latest",
        "built Monday_7_September_2026",
        f"{REGISTRY}:latest",
        first,
        "unchecked: no cosign",
    ]
    assert rows[1][5] == second
    assert rows[1][3] == "built Monday_14_September_2026"
    assert all(re.fullmatch(WHEN, row[0]) for row in rows)


def test_an_install_from_a_file_records_the_file(w):
    mine = w.root / "cpu.sif"
    mine.write_text("{}")
    assert w.rt_run("install", "cpu", "--from", mine).rc == 0
    assert history(w)[-1][1:] == [
        "install",
        "radio_transients:cpu",
        "built unknown",
        str(mine),
        "-",  # no registry digest for a file
        "unchecked: local file",
    ]
    out = w.tool("rt-run", "list").out
    assert re.search(rf"  installed  {WHEN}, from {re.escape(str(mine))}\n", out)


def test_a_failed_update_leaves_no_history(full):
    full.registry.move()
    (full.state / "pull_fails").write_text("1")
    assert full.tool("rt-run", "update").rc == 1
    assert [row[1] for row in history(full)] == ["install"]


def test_list_gives_the_digest_and_how_to_install_it_again(full):
    digest = full.registry.digests["latest"]
    out = full.tool("rt-run", "list").out
    assert f"  digest     {digest}\n" in out
    assert f"    rt-run install latest --digest {digest}\n" in out


def test_install_digest_pulls_that_exact_build(w):
    old = w.registry.digests["gpu"]
    w.registry.move("gpu")  # the tag has moved on since
    result = w.rt_run("install", "gpu", "--digest", old)
    assert result.rc == 0, result.out
    assert w.pulls() == [f"{REGISTRY}@{old}"]
    assert (w.rt / "tag").read_text().strip() == "gpu"
    assert history(w)[-1][5] == old


@pytest.mark.parametrize(
    "args, complaint",
    [
        (["gpu", "--digest", "sha256:abc"], "isn't a digest"),
        (["--digest", fake_digest("gpu", 0)], "name the variant"),
        (["gpu", "--digest", fake_digest("gpu", 0), "--from", "x.sif"], "--from"),
    ],
)
def test_install_digest_is_checked(w, args, complaint):
    (w.home / "x.sif").write_text("{}")
    result = w.rt_run("install", *args)
    assert result.rc == 2 and complaint in result.out
    assert not w.pulls()


def test_without_the_registry_it_pulls_by_tag(w):
    result = w.rt_run("install", "full", RT_REGISTRY_API="http://127.0.0.1:9")
    assert result.rc == 0, result.out
    assert "no digest to record" in result.out
    assert w.pulls() == [f"{REGISTRY}:latest"]
    assert history(w)[-1][5] == "-"
    assert "--digest" not in w.tool("rt-run", "list").out


def test_resolve_digest(w, monkeypatch):
    monkeypatch.setenv("RT_REGISTRY_API", w.registry.url)
    assert rt_run.resolve_digest("cpu") == w.registry.digests["cpu"]
    assert rt_run.resolve_digest("no-such-tag") == ""


def host_command(w, name):
    """A command of the user's, in a directory of its own; returns the dir."""
    directory = w.root / "host-bin"
    directory.mkdir(exist_ok=True)
    (directory / name).write_text("#!/bin/sh\n")
    (directory / name).chmod(0o755)
    return directory


def test_install_names_the_host_commands_that_run_first(w):
    mine = host_command(w, "jupyter")
    result = w.rt_run("install", "full", PATH=f"{w.path}:{mine}")
    assert result.rc == 0, result.out
    assert "1 of the tools shares a name with a command of yours" in result.out
    assert str(mine / "jupyter") in result.out
    assert "over from a command" not in result.out
    assert full_listing_unchanged(w)


def full_listing_unchanged(w):
    """A name clash is reported, never acted on: every tool is still linked."""
    return (w.rt / "tools").read_text().split() == LINKED and all(
        w.ours(w.bin / name) for name in LINKED
    )


def test_install_names_the_host_commands_it_takes_over_from(w):
    mine = host_command(w, "heimdall")
    result = w.rt_run("install", "full", PATH=f"{w.bin}:{w.path}:{mine}")
    assert result.rc == 0, result.out
    assert "1 of the tools takes over from a command of yours" in result.out
    assert str(mine / "heimdall") in result.out
    assert "export PATH" not in result.out  # it is already on PATH
    assert full_listing_unchanged(w)


def test_list_names_the_overlaps_on_the_current_path(full):
    mine = host_command(full, "heimdall")
    result = full.tool("rt-run", "list", PATH=f"{mine}:{full.path}:{full.bin}")
    assert "shares a name with a command of yours" in result.out
    assert str(mine / "heimdall") in result.out


def test_a_host_without_python3_gets_a_sentence(full):
    empty = full.root / "empty"
    empty.mkdir()
    result = full.tool("heimdall", PATH=empty)
    assert result.rc == 127
    assert result.out.strip() == (
        "rt-run: needs python3 (3.8 or newer) on this machine"
    )


def test_path_overlaps(tmp_path):
    ours, a, b = tmp_path / "ours", tmp_path / "a", tmp_path / "b"
    for directory in (ours, a, b):
        directory.mkdir()
    for path in (a / "x", b / "x", b / "y", a / "not-executable"):
        path.write_text("")
    for path in (a / "x", b / "x", b / "y"):
        path.chmod(0o755)
    tools = ["x", "y", "z", "not-executable"]
    path = f"{a}:{ours}:{b}"
    assert rt_run.path_overlaps(tools, str(ours), path) == [
        ("x", str(a / "x"), False),  # the first match on PATH is the one
        ("y", str(b / "y"), True),
    ]
    # Not on PATH yet: it will be appended, so everything else comes first.
    assert rt_run.path_overlaps(["y"], str(ours), f"{a}:{b}") == [
        ("y", str(b / "y"), False)
    ]


# ---------------------------------------------------------------------------
# Running a tool
# ---------------------------------------------------------------------------


def test_a_link_runs_its_tool_with_its_arguments_intact(full):
    result = full.tool("heimdall", "-f", "x.fil", "-dm", "0", "two words")
    sif = str(full.rt / "image.sif")
    assert result.rc == 0
    assert result.tail(7) == [sif, "heimdall", "-f", "x.fil", "-dm", "0", "two words"]


def test_an_unlinked_command_runs_through_rt_run(full):
    result = full.tool("rt-run", "python", "-c", "1")
    assert result.rc == 0 and "python" in result.lines()


def test_a_command_the_image_lacks_is_refused_up_front(full):
    result = full.tool("rt-run", "fetch.py")
    assert result.rc == 1 and "not in the installed latest image" in result.out


def test_the_drivers_tools_are_not_checked(full):
    """--nv mounts the host's nvidia-smi into the image at run time."""
    result = full.tool("rt-run", "nvidia-smi", "-L", RT_NV=1)
    assert result.rc == 0 and "nvidia-smi" in result.lines()


def test_a_command_given_by_path_is_not_checked(full):
    assert full.tool("rt-run", "/usr/local/bin/heimdall").rc == 0


def test_binds_come_from_the_arguments(full):
    """Each path mounts the filesystem it is on (see World.mounts)."""
    data = full.root / "data"
    for directory in (data / "obs", data / "out", data / "nested/obs"):
        directory.mkdir(parents=True)
    (full.root / "other").mkdir()
    (data / "obs/x.fil").touch()
    (data / "nested/obs/y.fil").touch()
    (full.home / "a.fil").touch()
    (data / "lnk").symlink_to(full.root / "other")
    result = full.tool(
        "heimdall",
        "-f",
        data / "obs/x.fil",
        f"-output_dir={data / 'out'}",
        "-dm",
        "5",
        "10",
        full.home / "a.fil",
        "/tmp/y",
        data / "new.cand",
        full.root / "nope/deeper/f",
        data / "lnk/z",
        "a.fil",
        "/usr/lib/x86_64-linux-gnu/out",
        "/nonexistent-top-level",
        data / "nested/obs/y.fil",
    )
    assert sorted(result.binds()) == sorted(
        str(p)
        for p in (
            # data/obs/x.fil, -output_dir=data/out, data/new.cand (an output
            # yet to be written) and data/lnk/z (through a symlink, as
            # spelled): all on the data filesystem.
            data,
            full.root / "other",  # the symlink's target, on a filesystem of its own
            data / "nested",  # a filesystem mounted inside data: the deepest wins
        )
    )
    # Not $HOME, /tmp, a path with no parent, /usr, nor / itself.


def test_binds_work_from_the_root_directory(full):
    (full.root / "data").mkdir()
    result = full.tool("heimdall", full.root / "data", cwd="/")
    assert result.binds() == [str(full.root / "data")]


def test_a_path_on_the_root_filesystem_mounts_its_top_directory(full):
    """No filesystem of its own: the directory under / it is in, as /data."""
    plain = full.root / "plain/obs"
    plain.mkdir(parents=True)
    top = "/" + plain.parts[1]  # where this checkout is: /home, say
    if rt_run.under(top, rt_run.CONTAINER_OWNED):
        pytest.skip(f"the checkout is under {top}, which is never mounted over")
    assert full.tool("heimdall", "-f", plain / "x.fil").binds() == [top]


def test_rt_bind_adds_its_binds(full):
    result = full.tool("heimdall", RT_BIND="/a,/b:/mnt")
    assert result.binds() == ["/a", "/b:/mnt"]


def test_nv_on_a_gpu_image_when_nvidia_smi_is_there(full):
    assert "--nv" in full.tool("heimdall", PATH=full.gpu_path).lines()


def test_no_nv_without_nvidia_smi(full):
    if shutil.which("nvidia-smi"):
        pytest.skip("this machine has an nvidia-smi")
    assert "--nv" not in full.tool("heimdall").lines()


def test_rt_nv_overrides(full):
    assert "--nv" not in full.tool("heimdall", PATH=full.gpu_path, RT_NV=0).lines()
    assert "--nv" in full.tool("heimdall", RT_NV=1).lines()


def test_rt_debug_prints_the_command_instead(full):
    result = full.tool("heimdall", "-f", "x.fil", RT_DEBUG=1)
    assert len(result.lines()) == 1
    assert result.out.startswith(f"{full.root}/stub/apptainer exec")
    assert "heimdall -f x.fil" in result.out


def python_only_path(w):
    """A PATH with python3 on it, for the links, and no container runtime."""
    directory = w.root / "python-only"
    directory.mkdir()
    (directory / "python3").symlink_to(sys.executable)
    return directory


def test_the_runtime_install_found_is_used_when_not_on_path(full):
    """As in a job script that never ran `module load apptainer`."""
    assert (full.rt / "engine").read_text().strip() == f"{full.root}/stub/apptainer"
    result = full.tool("heimdall", "-f", "x.fil", PATH=python_only_path(full))
    assert result.rc == 0, result.out
    assert result.tail(4) == [str(full.rt / "image.sif"), "heimdall", "-f", "x.fil"]


def test_no_runtime_anywhere_says_to_load_the_module(w):
    result = w.rt_run("heimdall", PATH=python_only_path(w))
    assert result.rc == 1 and "module load apptainer" in result.out


def test_links_find_a_custom_rt_home_without_being_told(w):
    """Install with RT_HOME set; then use the links from an environment
    without it, as a job's may be."""
    home = w.root / "project/rt"
    assert w.rt_run("install", "full", RT_HOME=home).rc == 0
    link = home / "bin/heimdall"
    result = w.run([link, "-f", "x.fil"])
    assert result.rc == 0, result.out
    assert result.tail(4) == [str(home / "image.sif"), "heimdall", "-f", "x.fil"]
    assert "radio_transients:latest" in w.run([home / "bin/rt-run", "list"]).out
    assert not w.rt.exists()  # nothing went to the default RT_HOME


def test_a_job_with_no_path_at_all_gets_a_standard_one(w, monkeypatch, capsys):
    """srun --export=NONE leaves no PATH. The runtime needs one to find the
    host's nvidia-smi for --nv; rt-run sets the one sh would assume."""
    image = w.root / "x.sif"
    image.write_text("{}")
    (w.home / "engine").write_text(f"{w.root}/stub/apptainer\n")
    monkeypatch.delenv("PATH")
    for key, value in (("RT_HOME", w.home), ("RT_SIF", image), ("RT_DEBUG", 1)):
        monkeypatch.setenv(key, str(value))
    assert rt_run.cmd_run("heimdall", ["-h"]) == 0
    assert os.environ["PATH"] == rt_run.DEFAULT_PATH
    assert capsys.readouterr().out.rstrip().endswith("heimdall -h")


def test_tmpdir_is_bound(full):
    """Slurm may point TMPDIR at a job's own disk, outside /tmp."""
    job_tmp = full.root / "job-tmp"
    (job_tmp / "1234").mkdir(parents=True)
    result = full.tool("heimdall", TMPDIR=job_tmp / "1234")
    assert result.binds() == [str(job_tmp)]  # the local disk it is on
    assert full.tool("heimdall", TMPDIR="/tmp").binds() == []


def test_rt_sif_runs_another_image_unchecked(full):
    other = full.root / "local.sif"
    other.write_text("{}")
    result = full.tool("rt-run", "anything", RT_SIF=other)
    assert result.rc == 0 and str(other) in result.lines()


def test_a_deleted_image_is_reported(full):
    (full.rt / "image.sif").unlink()
    result = full.tool("heimdall")
    assert result.rc == 1 and "rt-run update" in result.out


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


def test_update_relinks_and_never_replaces_a_users_file(full):
    first, second = full.registry.digests["latest"], full.registry.move()
    (full.bin / "mine").write_text("a user's own file")
    (full.bin / "riptide").write_text("a user's own riptide")
    full.listing(
        [
            [E, "your_header.py", "x"],
            ["/usr/local/bin", "heimdall", "x"],
            ["/usr/local/bin", "riptide", "x"],
            ["/usr/local/bin", "rffa", "x"],
        ]
    )
    result = full.tool("rt-run", "update")
    assert result.rc == 0, result.out
    assert full.pulls() == [f"{REGISTRY}@{first}", f"{REGISTRY}@{second}"]
    assert not (full.bin / "jupyter").exists()  # dropped by the image
    assert full.ours(full.bin / "rffa")  # gained
    assert (full.bin / "riptide").read_text() == "a user's own riptide"
    assert "skipping" in result.out and "riptide" in result.out
    assert (full.bin / "mine").exists()


def test_update_downloads_nothing_while_the_tag_has_not_moved(full):
    result = full.tool("rt-run", "update")
    assert result.rc == 0 and "already up to date" in result.out
    assert len(full.pulls()) == 1 and len(history(full)) == 1
    full.registry.move()
    full.built("Monday_14_September_2026")
    result = full.tool("rt-run", "update")
    assert result.rc == 0 and "already up to date" not in result.out
    assert len(full.pulls()) == 2


def test_without_a_digest_update_compares_build_dates(full):
    offline = {"RT_REGISTRY_API": "http://127.0.0.1:9"}
    assert "already up to date" in full.tool("rt-run", "update", **offline).out
    full.built("Monday_14_September_2026")
    assert "already up to date" not in full.tool("rt-run", "update", **offline).out


def test_update_takes_rt_run_from_the_image_when_newer(full):
    full.registry.move()
    (full.state / "image_rt_run").write_text("#!/usr/bin/env python3\n# newer\n")
    full.listing(module=["/x/rt_run.py", "999.0.0"])
    assert full.tool("rt-run", "update").rc == 0
    assert (full.rt / "rt-run").read_text().endswith("# newer\n")


def test_update_keeps_its_own_rt_run_when_the_images_is_older(full):
    full.registry.move()
    (full.state / "image_rt_run").write_text("# older\n")
    full.listing(module=["/x/rt_run.py", "0.0.1"])
    assert full.tool("rt-run", "update").rc == 0
    assert (full.rt / "rt-run").read_text() == INSTALLED


def test_a_failed_pull_leaves_the_install_as_it_was(full):
    full.registry.move()
    (full.state / "pull_fails").write_text("1")
    result = full.tool("rt-run", "update")
    assert result.rc == 1 and "could not pull" in result.out
    assert not (full.rt / "image.sif.new").exists()
    assert full.tool("heimdall").rc == 0


def test_an_image_with_no_tools_is_refused(full):
    full.registry.move()
    full.listing([])
    result = full.tool("rt-run", "update")
    assert result.rc == 1 and "found no tools" in result.out
    assert not (full.rt / "image.sif.new").exists()
    assert full.tool("heimdall").rc == 0


# ---------------------------------------------------------------------------
# install --from
# ---------------------------------------------------------------------------


def test_install_from_hard_links_the_image_and_moves_the_links(full):
    (full.bin / "mine").write_text("a user's own file")
    mine = full.root / "cpu.sif"
    mine.write_text("{}")
    result = full.tool("rt-run", "install", "cpu", "--from", mine, "--bin-dir", "~x")
    assert result.rc == 0, result.out
    assert len(full.pulls()) == 1  # the first install's only
    assert (full.rt / "image.sif").stat().st_ino == mine.stat().st_ino
    new_bin = full.home / "~x"
    assert (full.rt / "bin-dir").read_text().strip() == str(new_bin)
    assert os.readlink(new_bin / "heimdall") == str(full.rt / "rt-run")
    assert not (full.bin / "heimdall").exists() and (full.bin / "mine").exists()


def test_a_reinstall_keeps_the_earlier_bin_dir(w):
    mine = w.root / "cpu.sif"
    mine.write_text("{}")
    other = w.home / "bin"
    assert w.rt_run("install", "cpu", "--from", mine, "--bin-dir", other).rc == 0
    result = w.rt_run("install", "cpu", "--from", mine)  # the same file again
    assert result.rc == 0, result.out
    assert (other / "heimdall").is_symlink() and not w.bin.exists()
    assert not (w.rt / "image.sif.new").exists()


def test_install_from_reads_the_variant_from_the_image(w):
    w.listing(tag="gpu")
    mine = w.root / "x.sif"
    mine.write_text("{}")
    result = w.rt_run("install", "--from", mine)
    assert result.rc == 0, result.out
    assert (w.rt / "tag").read_text().strip() == "gpu"


def test_install_from_an_unlabelled_image_needs_the_variant(w):
    mine = w.root / "x.sif"
    mine.write_text("{}")
    result = w.rt_run("install", "--from", mine)
    assert result.rc == 2 and "can't tell which variant" in result.out
    assert "VARIANT is one of" not in result.out  # the message says what to type
    assert not (w.rt / "image.sif.new").exists() and not (w.rt / "tag").exists()


def test_install_from_another_filesystem_uses_the_image_in_place(w):
    shm = Path("/dev/shm")
    if not shm.is_dir() or os.stat(str(shm)).st_dev == os.stat(str(w.root)).st_dev:
        pytest.skip("needs /dev/shm on a filesystem of its own")
    mine = Path(tempfile.mkdtemp(dir=str(shm))) / "x.sif"
    try:
        mine.write_text("{}")
        result = w.rt_run("install", "cpu", "--from", mine)
        assert result.rc == 0 and "where it is" in result.out
        assert (w.rt / "image").read_text().strip() == str(mine)
        assert str(mine) in w.tool("heimdall").lines()
        result = w.tool("rt-run", "uninstall")
        assert mine.exists() and "Left" in result.out
    finally:
        shutil.rmtree(str(mine.parent))


def test_no_nv_for_the_cpu_image_even_with_a_gpu(w):
    mine = w.root / "cpu.sif"
    mine.write_text("{}")
    assert w.rt_run("install", "cpu", "--from", mine).rc == 0
    result = w.tool("heimdall", PATH=w.gpu_path)
    assert result.rc == 0 and "--nv" not in result.lines()


# ---------------------------------------------------------------------------
# Inside an image: singularity run --app install IMAGE
# ---------------------------------------------------------------------------


def fake_image_path(w):
    """Directories standing in for an image's PATH, for an in-process scan."""
    local, env = w.root / "usr/local/bin", w.root / "envs/RT/bin"
    for directory in (local, env):
        directory.mkdir(parents=True)
    (local / "heimdall").write_bytes(b"\x7fELF...")
    (local / "psrdada.pc").write_text("prefix=/usr/local\n")
    (env / "jupyter").write_text("#!/usr/bin/env python\n")
    (env / "pylint").write_text("#!/usr/bin/env python\n")
    for path in (local / "heimdall", local / "psrdada.pc", env / "jupyter"):
        path.chmod(0o755)
    (env / "pylint").chmod(0o755)
    return f"{local}:{env}"


def test_install_inside_the_image_needs_no_runtime(w):
    image = w.home / "rt.sif"
    image.write_text("{}")
    result = w.rt_run(
        "install", "cpu", SINGULARITY_CONTAINER=image, PATH=fake_image_path(w)
    )
    assert result.rc == 0, result.out
    assert (w.rt / "tools").read_text().split() == ["heimdall", "jupyter"]
    assert (w.rt / "image.sif").stat().st_ino == image.stat().st_ino
    assert (w.rt / "rt-run").read_text() == INSTALLED
    assert not w.pulls()
    assert "the links need python3" in result.out
    assert "a name with a command" not in result.out  # this PATH is the image's
    # And back on the host, the links run the tool in that image.
    assert str(w.rt / "image.sif") in w.tool("heimdall").lines()


def test_inside_the_image_update_and_from_are_refused(w):
    image = w.home / "rt.sif"
    image.write_text("{}")
    result = w.rt_run("update", APPTAINER_CONTAINER=image)
    assert result.rc == 1 and "on the host" in result.out
    result = w.rt_run("install", "--from", image, APPTAINER_CONTAINER=image)
    assert result.rc == 2


def test_inside_the_image_an_unmounted_image_gets_a_hint(w):
    result = w.rt_run("install", "cpu", SINGULARITY_CONTAINER="/elsewhere/rt.sif")
    assert result.rc == 1 and "-B /elsewhere" in result.out


# ---------------------------------------------------------------------------
# uninstall
# ---------------------------------------------------------------------------


def test_uninstall_removes_only_what_rt_run_made(full):
    (full.bin / "mine").write_text("a user's own file")
    result = full.tool("rt-run", "uninstall")
    assert result.rc == 0
    assert not any(full.ours(p) for p in full.bin.iterdir())
    assert (full.bin / "mine").exists()
    assert sorted(p.name for p in full.rt.iterdir()) == ["bin"]


def test_uninstall_keeps_the_file_installed_from(w):
    mine = w.root / "cpu.sif"
    mine.write_text("{}")
    assert w.rt_run("install", "cpu", "--from", mine).rc == 0
    assert w.tool("rt-run", "uninstall").rc == 0
    assert mine.exists() and not w.rt.exists()


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------


def test_without_cosign_the_signature_goes_unchecked(w):
    result = w.rt_run("install", "gpu")
    assert result.rc == 0, result.out
    assert "not checking the image's signature" in result.out
    assert "install cosign" in result.out
    assert history(w)[-1][6] == "unchecked: no cosign"
    assert "  signature  unchecked: no cosign\n" in w.tool("rt-run", "list").out


def test_with_cosign_the_signature_is_checked_first(w):
    w.cosign()
    digest = w.registry.digests["gpu"]
    result = w.rt_run("install", "gpu")
    assert result.rc == 0, result.out
    assert w.cosign_calls() == [
        ["verify", "--certificate-identity", rt_run.SIGNER]
        + ["--certificate-oidc-issuer", rt_run.SIGNER_ISSUER]
        + [f"ghcr.io/josephwkania/radio_transients@{digest}"]
    ]
    assert rt_run.SIGNER.endswith("/build-test-push.yml@refs/heads/main")
    assert w.pulls() == [f"{REGISTRY}@{digest}"]
    assert history(w)[-1][6] == rt_run.SIGNED
    assert f"  signature  {rt_run.SIGNED}\n" in w.tool("rt-run", "list").out


def test_an_unsigned_image_is_never_pulled(w):
    w.cosign()
    w.unsign(w.registry.digests["gpu"])
    result = w.rt_run("install", "gpu")
    assert result.rc == 1
    assert "no valid signature" in result.out and "no signatures found" in result.out
    assert "RT_VERIFY=0" in result.out
    assert not w.pulls() and not (w.rt / "tag").exists()
    assert not w.bin.exists() or not any(w.bin.iterdir())


def test_install_digest_checks_that_digest(w):
    w.cosign()
    old = w.registry.digests["gpu"]
    w.registry.move("gpu")
    assert w.rt_run("install", "gpu", "--digest", old).rc == 0
    assert w.cosign_calls()[-1][-1].endswith(f"@{old}")


def test_rt_verify_0_skips_the_check(w):
    w.cosign()
    w.unsign(w.registry.digests["gpu"])
    result = w.rt_run("install", "gpu", RT_VERIFY=0)
    assert result.rc == 0, result.out
    assert not w.cosign_calls()
    assert history(w)[-1][6] == "unchecked: RT_VERIFY=0"


@pytest.mark.parametrize(
    "setup, complaint",
    [
        ("no cosign", "cosign, which checks it, isn't installed"),
        ("no registry", "didn't say which image the tag is"),
        ("from a file", "an image from a file has no signature"),
    ],
)
def test_rt_verify_require_refuses_what_it_cannot_check(w, setup, complaint):
    args, env = ["install", "gpu"], {"RT_VERIFY": "require"}
    if setup != "no cosign":
        w.cosign()
    if setup == "no registry":
        env["RT_REGISTRY_API"] = "http://127.0.0.1:9"
    if setup == "from a file":
        (w.root / "gpu.sif").write_text("{}")
        args += ["--from", w.root / "gpu.sif"]
    result = w.rt_run(*args, **env)
    assert result.rc == 1 and complaint in result.out
    assert not w.pulls() and not (w.rt / "tag").exists()


def test_rt_verify_takes_only_its_three_values(w):
    result = w.rt_run("install", "gpu", RT_VERIFY="yes")
    assert result.rc == 2 and "it takes auto, require or 0" in result.out
    assert not w.pulls()


def test_update_checks_a_new_image_and_keeps_the_old_one_if_unsigned(full):
    full.cosign()
    assert full.tool("rt-run", "update").rc == 0  # the tag hasn't moved
    assert not full.cosign_calls()
    signed = full.registry.move()
    assert full.tool("rt-run", "update").rc == 0
    assert full.cosign_calls()[-1][-1].endswith(f"@{signed}")
    full.unsign(full.registry.move())
    result = full.tool("rt-run", "update")
    assert result.rc == 1 and "no valid signature" in result.out
    assert (full.rt / "digest").read_text().strip() == signed
    assert [row[1] for row in history(full)] == ["install", "update"]
    assert full.pulls()[-1] == f"{REGISTRY}@{signed}"


# ---------------------------------------------------------------------------
# The scan code that runs inside an image
# ---------------------------------------------------------------------------


def test_the_scan_sent_to_an_image_runs_there(w):
    """The real code, under `python -I`, against a fake PATH."""
    (w.state / "real_scan_path").write_text(fake_image_path(w))
    result = w.rt_run("install", "cpu")
    assert result.rc == 0, result.out
    assert (w.rt / "tools").read_text().split() == ["heimdall", "jupyter"]
    assert "psrdada.pc" in (w.rt / "commands").read_text().split()


class Dist:
    """Enough of importlib.metadata.Distribution for installed_from_source."""

    def __init__(self, installer, direct_url, files=(), scripts=()):
        self.texts = {"INSTALLER": installer, "direct_url.json": direct_url}
        self.files = list(files)
        self.entry_points = [Entry("console_scripts", s) for s in scripts]
        self.entry_points.append(Entry("gui_scripts", "not-a-script"))

    def read_text(self, name):
        return self.texts.get(name)


class Entry:
    def __init__(self, group, name):
        self.group, self.name = group, name


def test_installed_from_source_picks_the_recipes_own_packages():
    dists = [
        Dist(
            "pip\n",
            '{"url": "file:///root/your"}',
            files=[
                "../../../bin/your_header.py",
                "../../../bin/__pycache__/your_header.cpython-310.pyc",
                "your/__init__.py",
            ],
            scripts=["your_entry"],
        ),
        Dist("conda", '{"url": "file:///home/conda/..."}', scripts=["httpx"]),
        Dist("pip", None, scripts=["pylint"]),  # from PyPI: no direct_url.json
    ]
    found = rt_run.installed_from_source(dists)
    assert found == ["your_entry", "your_header.py"]


def test_scan_in_process(w, monkeypatch):
    monkeypatch.setenv("PATH", fake_image_path(w) + ":/nonexistent")
    labels = w.root / "labels.json"
    labels.write_text('{"RT_TAG": "gpu", "org.label-schema.build-date": "today"}')
    found = rt_run.scan(str(labels), dists=[])
    assert found["tag"] == "gpu" and found["built"] == "today"
    names = {(Path(d).name, n, k) for d, n, k in found["commands"]}
    assert ("bin", "heimdall", "x") in names and ("bin", "psrdada.pc", "-") in names


@pytest.mark.parametrize(
    "directory, name, kind, from_source, linked",
    [
        ("/usr/local/bin", "heimdall", "x", False, True),
        ("/usr/local/bin", "psrdada.pc", "-", False, False),
        ("/usr/local/presto/bin", "prepfold", "x", False, True),
        ("/usr/bin", "htop", "x", False, False),
        ("/usr/local/cuda/bin", "nvcc", "x", False, False),
        ("/usr/local/nvidia/bin", "nvidia-smi", "x", False, False),
        ("/scif/apps/install/bin", "x", "x", False, False),
        ("/usr/local/miniconda/condabin", "conda", "x", False, False),
        (E, "your_header.py", "x", True, True),
        (E, "pylint", "x", False, False),
        (E, "jupyter", "x", False, True),
        (E, "rt-run", "x", True, False),
        (E, "python3.10", "x", True, False),
    ],
)
def test_should_link(directory, name, kind, from_source, linked):
    assert rt_run.should_link(directory, name, kind, from_source) is linked


def test_the_first_directory_on_path_wins():
    found = {
        "commands": [[E, "x", "x"], ["/usr/local/bin", "x", "x"]],
        "from_source": [],
    }
    assert rt_run.sort_out(found) == (["x"], [])


def test_version_key():
    assert rt_run.version_key("0.10.0") > rt_run.version_key("0.9.1")
    assert not rt_run.version_key("garbage")


def test_filesystem_root():
    points = {"/", "/lustre", "/lustre/project", "/scratch"}
    assert rt_run.filesystem_root("/lustre/project/a/b", points) == "/lustre/project"
    assert rt_run.filesystem_root("/lustre/other", points) == "/lustre"
    assert rt_run.filesystem_root("/scratch", points) == "/scratch"
    assert rt_run.filesystem_root("/data/obs1", points) == "/data"  # on /
    assert rt_run.filesystem_root("/lustrex/a", points) == "/lustrex"  # not /lustre


def test_mount_points_reads_the_kernels_table(tmp_path):
    table = tmp_path / "mountinfo"
    table.write_text(
        "22 1 8:1 / / rw - ext4 /dev/sda1 rw\n"
        "23 22 0:5 / /data/my\\040disk rw - nfs host:/x rw\n"
    )
    assert rt_run.mount_points(str(table)) == {"/", "/data/my disk"}
    assert rt_run.mount_points(str(tmp_path / "missing")) == set()


# ---------------------------------------------------------------------------
# Safeguards
# ---------------------------------------------------------------------------


def test_install_refuses_an_rt_home_others_can_write(w):
    home = w.root / "shared/rt"
    home.mkdir(parents=True)
    home.chmod(0o777)
    result = w.rt_run("install", "full", RT_HOME=home)
    assert result.rc == 1 and f"chmod go-w {home}" in result.out
    assert not w.pulls() and not os.listdir(home)


def test_install_refuses_an_image_others_can_write(w):
    mine = w.root / "cpu.sif"
    mine.write_text("{}")
    mine.chmod(0o666)
    result = w.rt_run("install", "cpu", "--from", mine)
    assert result.rc == 1 and "writable by others" in result.out


@pytest.mark.parametrize("name", ["image.sif", "rt-run", "bin", "engine", "image"])
def test_a_tool_wont_run_once_others_can_change_its_parts(full, name):
    path = full.rt / name
    path.chmod(path.stat().st_mode | 0o002)
    result = full.tool("heimdall")
    assert result.rc == 1 and f"chmod go-w {path}" in result.out
    assert "exec" not in result.lines()  # never reached the runtime


def test_a_private_group_may_write(full):
    """umask 002 under a user private group gives no one else access."""
    if not rt_run.private_group(os.getgid()):
        pytest.skip("this user's group isn't private")
    image = full.rt / "image.sif"
    image.chmod(0o664)
    assert full.tool("heimdall").rc == 0


def test_rt_runs_own_files_are_private_whatever_the_umask(w):
    old = os.umask(0o002)  # the subprocess inherits it
    try:
        assert w.rt_run("install", "full").rc == 0
    finally:
        os.umask(old)
    for path in [w.rt, w.bin, *w.rt.iterdir()]:
        if not path.is_symlink():
            assert not path.stat().st_mode & 0o022, path
    assert w.tool("heimdall").rc == 0


def test_install_wont_use_a_directory_holding_other_files(w):
    elsewhere = w.root / "papers"
    elsewhere.mkdir()
    (elsewhere / "tools").write_text("my list of tools")
    result = w.rt_run("install", "full", RT_HOME=elsewhere)
    assert result.rc == 1 and "directory of its own" in result.out
    assert sorted(os.listdir(elsewhere)) == ["tools"] and not w.pulls()


def test_uninstall_leaves_an_unmarked_directory_alone(full):
    (full.rt / rt_run.MARKER).unlink()
    (full.rt / "rt-run").unlink()  # nor an install from before the marker
    result = full.rt_run("uninstall")
    assert result.rc == 1 and "isn't marked" in result.out
    assert (full.rt / "image.sif").exists()


def test_an_install_from_before_the_marker_is_taken_as_ours(full):
    (full.rt / rt_run.MARKER).unlink()
    assert full.rt_run("install", "full").rc == 0
    assert (full.rt / rt_run.MARKER).exists()


def test_a_second_install_waits_for_the_first(full):
    fcntl = rt_run.fcntl
    with open(full.rt / ".lock", "a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)  # an install already under way
        with subprocess.Popen(
            [sys.executable, str(RT_RUN), "install", "full"],
            cwd=str(full.home),
            env=full.env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ) as waiting:
            with pytest.raises(subprocess.TimeoutExpired):
                waiting.wait(timeout=2)  # still blocked
            fcntl.flock(handle, fcntl.LOCK_UN)  # the first one finishes
            out, _ = waiting.communicate(timeout=60)
    assert waiting.returncode == 0, out
    assert "waiting for another install" in out


def test_install_checks_there_is_room_first(w):
    w.registry.size = 10**18  # an exabyte
    result = w.rt_run("install", "full")
    assert result.rc == 1 and "GB free" in result.out
    assert not w.pulls()


def test_update_checks_there_is_room_first(full):
    full.registry.move()
    full.registry.size = 10**18
    result = full.tool("rt-run", "update")
    assert result.rc == 1 and "GB free" in result.out
    assert len(full.pulls()) == 1 and full.tool("heimdall").rc == 0


def test_the_images_own_directories_are_never_mounted_over(full):
    plain = full.root / "plain/obs"
    plain.mkdir(parents=True)
    top = "/" + plain.parts[1]  # say /home: here, it stands in for /fetch
    if rt_run.under(top, rt_run.CONTAINER_OWNED):
        pytest.skip(f"the checkout is under {top}, which is never mounted over")
    (full.rt / "top").write_text(top + "\n")
    assert full.tool("heimdall", "-f", plain / "x.fil").binds() == []
    result = full.tool("heimdall", "-f", plain / "x.fil", RT_BIND=top)
    assert result.binds() == [top]  # unless asked for by name


def test_image_top_dirs(tmp_path):
    for name in ("fetch/bin", "opt/x", "mnt", "home/me", "data/proj"):
        (tmp_path / name).mkdir(parents=True)
    (tmp_path / "environment").write_text("")  # a file, not a directory
    mounted = {str(tmp_path / "home/me"), str(tmp_path / "data/proj")}
    # mnt is empty; home and data hold mounts Singularity made.
    assert rt_run.image_top_dirs(mounted, root=str(tmp_path)) == ["/fetch", "/opt"]


def test_rt_autobind_0_mounts_only_what_rt_bind_names(full):
    (full.root / "data/obs").mkdir(parents=True)
    args = ("-f", full.root / "data/obs/x.fil")
    assert full.tool("heimdall", *args, RT_AUTOBIND=0).binds() == []
    result = full.tool("heimdall", *args, RT_AUTOBIND=0, RT_BIND="/a")
    assert result.binds() == ["/a"]

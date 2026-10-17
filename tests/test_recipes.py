"""Checks on the Singularity recipes that need no build and no image.

pip install . pytest && pytest tests/test_recipes.py
"""

import os
import re
import shlex
import subprocess
import sys

import pytest

from rt_run_fakes import HERE, RT_RUN
import rt_run  # pylint: disable=wrong-import-order  # importable once rt_run_fakes ran

RECIPES = ["Singularity", "Singularity.gpu", "Singularity.cpu", "Singularity.arm"]


@pytest.mark.parametrize("recipe", RECIPES)
def test_recipe_labels_its_image_with_rt_runs_version(recipe, tmp_path):
    text = (HERE.parent / recipe).read_text()
    labels = text.split("%labels\n", 1)[1].split("\n%", 1)[0]
    assert "Version" not in labels and "Build-date" not in labels
    # The lines of %post that write them, run here against this rt-run.
    lines = [
        line
        for line in text.splitlines()
        if "RT_VERSION" in line or "Build-date" in line
    ]
    assert len(lines) == 3
    script = "\n".join(lines).replace(
        "/usr/local/miniconda/envs/RT/bin/rt-run",
        f"{shlex.quote(sys.executable)} {shlex.quote(str(RT_RUN))}",
    )
    out = tmp_path / "labels"
    env = {**os.environ, "SINGULARITY_LABELS": str(out)}
    subprocess.run(["bash", "-euc", script], env=env, check=True)
    version, built = out.read_text().splitlines()
    assert version == f"Version v{rt_run.__version__}"
    assert re.fullmatch(r"Build-date \d\d-[A-Z][a-z]{2}-\d{4}", built)


@pytest.mark.parametrize("recipe", RECIPES)
def test_recipe_exports_no_compiler_flags(recipe):
    # An exported CFLAGS reaches every later build in %post, and autoconf
    # drops its default -O2 whenever CFLAGS is set: it left psrdada, Heimdall
    # and sigproc unoptimised. Give flags to the one build that needs them.
    text = (HERE.parent / recipe).read_text()
    exported = re.findall(
        r"^\s*export\s+(?:C|CXX|CPP|F|NVCC_|CUDA_NVCC_)FLAGS\b", text, re.M
    )
    assert not exported


@pytest.mark.parametrize("recipe", RECIPES)
def test_recipe_clones_over_https(recipe):
    assert "git://" not in (HERE.parent / recipe).read_text()


@pytest.mark.parametrize(
    "recipe", ["Singularity", "Singularity.cpu", "Singularity.arm"]
)
@pytest.mark.parametrize(
    "march, flag, label",
    [
        ("none", "", "none"),  # the default, as CI builds it
        ("", "", "none"),
        ("x86-64-v3", "-march=x86-64-v3", "x86-64-v3"),
    ],
)
def test_march_is_off_unless_asked_for(recipe, march, flag, label, tmp_path):
    text = (HERE.parent / recipe).read_text()
    arguments = text.split("%arguments\n", 1)[1].split("\n%", 1)[0]
    assert re.search(r"^\s*MARCH=none$", arguments, re.M)
    # The three lines at the top of %post that turn MARCH into MARCH_FLAG
    # and a label, run here as the build would run them.
    start = re.compile(r'\s*(MARCH="\{\{ MARCH \}\}"|case "\$MARCH"|echo "Build-march)')
    setup = [line for line in text.splitlines() if start.match(line)]
    assert len(setup) == 3
    script = "\n".join(setup).replace("{{ MARCH }}", march)
    script += '\nprintf "%s" "$MARCH_FLAG"'
    out = tmp_path / "labels"
    env = {**os.environ, "SINGULARITY_LABELS": str(out)}
    result = subprocess.run(
        ["bash", "-euc", script],
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )
    assert result.stdout == flag
    assert out.read_text() == f"Build-march {label}\n"

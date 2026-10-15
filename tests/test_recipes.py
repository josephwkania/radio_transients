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

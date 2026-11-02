# Development: tests, CI and releases

## Testing

Each recipe has a test suite in `tests/`, written to run inside the built
image. There are two, split by what they need:

`container_test.sh` needs **no GPU**, so it runs on an ordinary CI runner. It
checks that the binaries exist, that their shared libraries resolve, that they
start, that every Python module imports, that the pinned versions the recipes
depend on are the ones actually installed, and that a real filterbank makes it
through a `fake` -> `rfifind` -> `prepdata` -> `realfft` ->
`single_pulse_search.py` pipeline.

```sh
singularity exec -B "$PWD/tests:/tests" radio_transients.sif \
    bash /tests/container_test.sh --variant full
```

`gpu_test.sh` needs an NVIDIA GPU. It runs every CUDA code path against a CPU
reference and requires them to agree, then injects pulses at a known DM into a
synthetic filterbank and asserts Heimdall recovers them at that DM.

```sh
singularity exec --nv -B "$PWD/tests:/tests" radio_transients_gpu.sif \
    bash /tests/gpu_test.sh --dm 100
```

The variant is auto-detected if you leave `--variant` off. `--quick` skips the
slow parts, `--keep` leaves the scratch directory for inspection, and
`--list` / `--only NAME` on `gpu_numerics.py` run a subset of the GPU checks.
Output is TAP 13, and the exit status is 0 only when every non-skipped
assertion passed -- so the suites can gate a build.

Results from a rebuild of every image (10-Sep-2026, Tesla T4):

| Image | Tag | Size | `container_test.sh` | `gpu_test.sh` |
|---|---|---|---|---|
| radio_transients     | `latest`          | 6.7 G | 151 passed, 0 failed, 18 skipped | 18 / 0 / 0 |
| radio_transients     | `latest-cuda12.6` | 7.0 G | 151 / 0 / 18 | 18 / 0 / 0 |
| radio_transients_gpu | `gpu`             | 6.5 G | 50 / 0 / 4   | 18 / 0 / 0 |
| radio_transients_gpu | `gpu-cuda12.6`    | 6.8 G | 50 / 0 / 4   | 18 / 0 / 0 |
| radio_transients_cpu | `cpu`             | 1.1 G | 113 / 0 / 18 | n/a |
| arm (aarch64)        | `arm`             | 994 M | 113 / 0 / 18 | n/a |

Skips are expected: a variant is not asked for tools it does not ship, and the
GPU-less suite skips assertions that need real hardware. The arm figures come
from a native aarch64 build.

`test_rt_run.py` tests `rt-run` itself, against a stub runtime, so it needs
neither Singularity nor an image: `pip install . pytest && pytest`.
`test_recipes.py`, run the same way, checks the recipes without building
them.

[`tests/README.md`](../tests/README.md) explains what each layer catches and
why, including the checks that exist specifically to catch a
passing-but-wrong result -- such as the negative control that fails a
dedisperser which ignores its DM argument.

## Continuous integration

- `build-test-push.yml` builds all six images on every push and pull request
  that touches a recipe or the tests, runs `container_test.sh` against each,
  and pushes to GHCR only from `main` and only if the tests pass -- an image
  that fails is never tagged. It then signs each image it pushed with cosign.
  The tests include `singularity run`, and installing `rt-run` from the image
  just built, both with its `install` app and from the host.
- `rt-run.yml` lints, tests and builds `rt-run` on its own, in a minute, when
  only it changes.
- `pypi.yml` uploads a `.dev` build to TestPyPI and installs it back from
  there, as a dry run, on every pull request (except from forks) and push to
  `main` that changes the package, and when run by hand from the Actions tab.
- `gpu-test.yml` runs the GPU suite on a self-hosted runner with a real card.

A `lint` job runs `black`, `flake8`, `pylint` and `shellcheck` over `tests/`.
The linter configs live in `tests/.flake8` and `tests/.pylintrc`, and CI passes
no flags of its own, so running the tools by hand gives the same answer CI
does.

### Weekly rebuilds

`build-test-push.yml` also runs Mondays at 06:00 UTC, and `gpu-test.yml` at
09:00 UTC against what it published. The recipes do not pin the repositories
they clone, so this is what catches upstream breakage -- and why the tags move
even when no recipe changed.

## Releasing

1. Bump `__version__` in `src/rt_run.py` and merge that to `main`. The images
   take their `Version` label from it, so the merge puts the new number on
   them too.
2. Publish a GitHub release tagged `v<version>`. `pypi.yml` checks the tag
   matches, re-runs the TestPyPI dry run on that source, then uploads to PyPI.

Both indexes use trusted publishing, so no token is stored; the header of
`pypi.yml` says how they were set up.

## Sylabs Cloud (legacy)

Images were built on an E5 v3 machine and uploaded to Sylabs Cloud at
https://cloud.sylabs.io/library/josephwkania/radio_transients/radio_transients
until 27-Nov-2021. They are no longer updated, and predate the CUDA 11.8 /
Ubuntu 22.04 rebuild, the meson PRESTO build and the test suite.

# radio_transients


[![Issues](https://img.shields.io/github/issues/josephwkania/radio_transients?style=flat-square)]()
[![Forks](https://img.shields.io/github/forks/josephwkania/radio_transients?style=flat-square)]()
[![Stars](https://img.shields.io/github/stars/josephwkania/radio_transients?style=flat-square)]()
[![License](https://img.shields.io/github/license/josephwkania/radio_transients?style=flat-square)]()
[![GHCR](https://img.shields.io/badge/Hosted-GHCR-Blue.svg)](https://github.com/josephwkania/radio_transients/pkgs/container/radio_transients)


## Overview

These are my Singularity Recipes for common radio transient software.
There are four containers. The two with GPU support are published twice, once
on CUDA 11.8 and once on CUDA 12.6, so there are six images to pull from.

### radio_transients

Contains everything (CPU+GPU)
 
    CUDA 11.8 or 12.6
    FETCH          https://github.com/devanshkv/fetch
    heimdall       https://sourceforge.net/p/heimdall-astro/wiki/Use/
    - dedisp       https://github.com/ajameson/dedisp
    htop           https://htop.dev/
    iqrm_apollo    https://gitlab.com/kmrajwade/iqrm_apollo
    jess           https://github.com/josephwkania/jess
    jupyterlab     https://jupyter.org/
    PRESTO         https://www.cv.nrao.edu/~sransom/presto/
    psrdada        http://psrdada.sourceforge.net/
    psrdada-python https://github.com/TRASAL/psrdada-python
    psrcat         https://www.atnf.csiro.au/people/pulsar/psrcat/download.html
    pysigproc      https://github.com/devanshkv/pysigproc
    riptide        https://github.com/v-morello/riptide
    sigproc        https://github.com/SixByNine/sigproc
    Tempo          http://tempo.sourceforge.net/
    RFIClean       https://github.com/ymaan4/RFIClean
    YAPP           https://github.com/jayanthc/yapp
    your           https://github.com/thepetabyteproject/your

Get with
`singularity pull radio_transients.sif oras://ghcr.io/josephwkania/radio_transients:latest`

or, for the CUDA 12.6 build,
`singularity pull radio_transients.sif oras://ghcr.io/josephwkania/radio_transients:latest-cuda12.6`

### radio_transients_cpu

Contains CPU based programs

    htop
    iqrm_apollo
    jupyterlab   
    PRESTO
    psrcat
    pysigproc
    riptide
    sigproc
    Tempo 
    RFIClean
    YAPP  
    your

Get with
`singularity pull radio_transients_cpu.sif oras://ghcr.io/josephwkania/radio_transients:cpu`

### radio_transients arm

The CPU container built for arm64 (aarch64), from `Singularity.arm`. Same
programs as `radio_transients_cpu`.

Get with
`singularity pull radio_transients_arm.sif oras://ghcr.io/josephwkania/radio_transients:arm`

### radio_transients_gpu

Contains gpu based programs

    CUDA 11.8 or 12.6
    FETCH
    jess
    jupyterlab
    heimdall
    - dedisp
    htop 
    psrdada 
    psrdada-python
    your

Get with
`singularity pull radio_transients_gpu.sif oras://ghcr.io/josephwkania/radio_transients:gpu`

or, for the CUDA 12.6 build,
`singularity pull radio_transients_gpu.sif oras://ghcr.io/josephwkania/radio_transients:gpu-cuda12.6`

### CUDA versions

`Singularity` and `Singularity.gpu` take the CUDA version as a build argument,
defaulting to 11.8:

    singularity build radio_transients.sif Singularity
    singularity build --build-arg CUDA_VERSION=12.6.3 radio_transients.sif Singularity

TensorFlow and cupy follow it: 2.14 is the last release for CUDA 11.8, 2.15.1
the first that works on 12, and the recipe reads `nvcc --version` to choose.

Any driver from 525 up runs either image, so 11.8 is the safe default. 12.9
would add Blackwell support and ~1.5 G per image with it; CUDA 13 does not
build, as dedisp does not compile against its Thrust.

### Running the tools from the host

`rt-run` puts the container's tools on your `PATH`, so `heimdall ...` works
without a `singularity exec` line, and it bind-mounts the directories your
arguments name. It needs `python3` (3.8 or newer) on the host, and nothing
else.

Install it with pip, then let it pull an image:

    pipx install git+https://github.com/josephwkania/radio_transients    # or pip install --user
    rt-run install gpu    # or full, cpu, arm, full-cuda12.6, gpu-cuda12.6

Or, with an image you already have, let the image install it:

    singularity run --app install radio_transients_gpu.sif

Either way, then:

    export PATH="$PATH:$HOME/.local/share/radio_transients/bin"    # and in ~/.bashrc

    heimdall -f /data/obs/x.fil -dm 0 1000
    your_header.py -f /data/obs/x.fil

With no variant, `rt-run install` picks arm on aarch64, full when `nvidia-smi`
is found, and cpu otherwise. `rt-run install --from radio_transients_gpu.sif`
installs an image you have from the host side. Images built before `rt-run`
existed have neither the `install` app nor the label saying which variant they
are, so for those, name it: `rt-run install gpu --from old.sif`.

- `rt-run update` pulls the image again; the tags are rebuilt every week. It
  also takes the copy of `rt-run` inside the new image when that is at least as
  new as yours, so the launcher keeps pace with the images by itself.
- `rt-run list` shows the tools it linked: the programs the recipes build, and
  the scripts of the Python packages they install from source. Anything else in
  the image runs as `rt-run COMMAND`, e.g. `rt-run python script.py`.
- `rt-run uninstall` removes the links, and the image if `rt-run` pulled it.

Each link runs `apptainer exec` (or `singularity exec`) on the installed image,
with `--nv` for the GPU images when the host has `nvidia-smi`. An argument
naming a path outside `$HOME`, `/tmp` and the current directory -- the value in
`--opt=/path` included -- has its directory bind-mounted at the same place, so
paths mean the same inside and out. Paths under the container's own system
directories (`/usr`, `/etc`, `/opt`, ...) are never bound over them. A path it
can't see, such as one inside a config file, needs `RT_BIND=/data,/scratch`.
`RT_DEBUG=1` prints the command instead of running it; `rt-run --help` and the
top of `src/rt_run.py` list the other settings.

Appending to `PATH`, as above, keeps any host command of the same name ahead
of the container's. `install` names any tool that shares a name with a command
already on your `PATH`, and says which of the two will run; `rt-run list` says
the same for the `PATH` you have now. It never replaces a file it didn't make.
On a host with no `python3`, the links say so rather than failing obscurely.

### Under Slurm

The links work in batch jobs as they do on a login node. On a two-GPU node
with Slurm's cgroup device limits, each job saw only the GPU it was given --
as GPU 0, so `-gpu_id 0` is right -- two jobs ran side by side on different
GPUs, and `scancel` stopped the container and everything in it.

    #!/bin/bash
    #SBATCH --gres=gpu:1
    #SBATCH --mem=8G
    #SBATCH --time=1:00:00

    heimdall -f /data/obs/x.fil -dm 0 1000 -gpu_id 0

What depends on the site:

- `--nv` is decided on the node a tool runs on, so CPU nodes don't get it.
- If apptainer is a module, a job that never runs `module load apptainer`
  still works: `rt-run` falls back to the apptainer that `install` or
  `update` last ran with.
- If jobs don't inherit your environment (`--export=NONE`), add the
  `export PATH=...` line to the script, or call the links by full path.
- A per-job `TMPDIR` outside `/tmp` is bind-mounted automatically.
- The image is 1 to 7 G. If your home quota can't take it, set `RT_HOME` to
  storage every node sees and that isn't purged, such as a project
  directory, for `rt-run install`. The links find it from then on without
  `RT_HOME` being set.

### How to use

Your `$HOME` automatically gets mounted.
You can mount a directory with `-B /dir/on/host:/mnt`, which will mount `/dir/on/host` to `/mnt` in the container. 

For the gpu processes, you must pass `--nv` when running singularity.

`singularity shell --nv -B /data:/mnt radio_transients_gpu.sif` 
will mount `/data` to `/mnt`, give you GPU access, and drop you into the interactive shell. 

`singularity exec --nv -B /data:/mnt radio_transients_gpu.sif your_heimdall.py -f /mnt/data.fil` 
will mount `/data` to `/mnt`, give you GPU access, and run your_heimdall.py without entering the container.

`singularity run IMAGE COMMAND ...` does the same as `exec`. With no command,
it opens a shell -- except in the full `radio_transients` image, where it runs
`your_heimdall.py`, so `singularity run radio_transients.sif -f /mnt/data.fil`
works too.

All the Python scripts are installed in a Conda environment `RT`, this environment is automatically loaded.

You can see the commits and corresponding dates by running `singularity inspect radio_transients.sif`

### Testing

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

`tests/README.md` explains what each layer catches and why, including the
checks that exist specifically to catch a passing-but-wrong result -- such as
the negative control that fails a dedisperser which ignores its DM argument.

### Continuous integration

`.github/workflows/build-test-push.yml` builds all four variants on every push
that touches a recipe or the tests, runs `container_test.sh` against each, and
pushes to GHCR only if the tests pass -- an image that fails is never tagged.
The tests include `singularity run`, and installing `rt-run` from the image
just built, both with its `install` app and from the host. `rt-run.yml` lints,
tests and builds `rt-run` on its own, in a minute, when only it changes.
`pypi.yml` publishes it to PyPI when a GitHub release is published; run by
hand from the Actions tab, it uploads a `.dev` build to TestPyPI instead and
installs it back from there, as a dry run. Both need PyPI to trust this
repository first -- the workflow's header says how to set that up.
`gpu-test.yml` runs the GPU suite on a self-hosted runner with a real card.

A `lint` job runs `black`, `flake8`, `pylint` and `shellcheck` over `tests/`.
The linter configs live in `tests/.flake8` and `tests/.pylintrc`, and CI passes
no flags of its own, so running the tools by hand gives the same answer CI
does.

#### Weekly rebuilds

Both workflows also run Mondays: `build-test-push.yml` at 06:00 UTC,
`gpu-test.yml` at 09:00 UTC against what it published. The recipes do not pin
the repositories they clone, so this is what catches upstream breakage.

The tags therefore move -- `:latest` today is not what you pulled last month,
even if no recipe changed. `singularity inspect` reports the build date and
commits. To pin a build, pull by digest instead of tag:

    singularity pull rt.sif \
        oras://ghcr.io/josephwkania/radio_transients@sha256:<digest>

Digests are on the package page linked from the badge above.

### Sylabs Cloud (legacy)

**These images are no longer updated.** They were last built on 27-Nov-2021,
so they predate the CUDA 11.8 / Ubuntu 22.04 rebuild, the meson PRESTO build,
and the test suite. Pull from GHCR instead, using the commands above. The rest
of this section is kept for reference.

These are built on a E5 v3 family machine and uploaded to Sylabs Cloud at 
https://cloud.sylabs.io/library/josephwkania/radio_transients/radio_transients
They where last built on 27-Nov-2021

If your processor your processor is significantly older than this, you may run into problems with 
the older processor not having the whole instruction set needed. In this case, you should build
use singularity to build the image locally. 

### Improvements

If you come across bug or have suggestions for improvements, let me know or submit a pull request.

### Thanks

To Kshitij Aggarwal for bug reports and suggestions.

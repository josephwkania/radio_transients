# radio_transients


[![Issues](https://img.shields.io/github/issues/josephwkania/radio_transients?style=flat-square)]()
[![Forks](https://img.shields.io/github/forks/josephwkania/radio_transients?style=flat-square)]()
[![Stars](https://img.shields.io/github/stars/josephwkania/radio_transients?style=flat-square)]()
[![License](https://img.shields.io/github/license/josephwkania/radio_transients?style=flat-square)]()
[![GHCR](https://img.shields.io/badge/Hosted-GHCR-Blue.svg)](https://github.com/josephwkania/radio_transients/pkgs/container/radio_transients)
[![PyPI](https://img.shields.io/pypi/v/radio-transients?style=flat-square)](https://pypi.org/project/radio-transients/)


Singularity/Apptainer containers with common radio transient software, and
`rt-run`, which runs their tools from the host as if they were installed
there.

## Images

| Tag | Contents | Size |
|---|---|---|
| `latest` | everything, CUDA 11.8 | 6.7 G |
| `latest-cuda12.6` | everything, CUDA 12.6 | 7.0 G |
| `gpu` | GPU tools, CUDA 11.8 | 6.5 G |
| `gpu-cuda12.6` | GPU tools, CUDA 12.6 | 6.8 G |
| `cpu` | CPU tools, x86-64 | 1.1 G |
| `arm` | CPU tools, aarch64 | 1.0 G |

    singularity pull radio_transients.sif oras://ghcr.io/josephwkania/radio_transients:latest

| Tool | full | gpu | cpu / arm |
|---|:-:|:-:|:-:|
| [FETCH](https://github.com/devanshkv/fetch) | ✓ | ✓ | |
| [heimdall](https://sourceforge.net/p/heimdall-astro/wiki/Use/) + [dedisp](https://github.com/ajameson/dedisp) | ✓ | ✓ | |
| [psrdada](http://psrdada.sourceforge.net/), [psrdada-python](https://github.com/TRASAL/psrdada-python) | ✓ | ✓ | |
| [jess](https://github.com/josephwkania/jess) | ✓ | ✓ | ✓ |
| [your](https://github.com/thepetabyteproject/your) | ✓ | ✓ | ✓ |
| [jupyterlab](https://jupyter.org/), [htop](https://htop.dev/) | ✓ | ✓ | ✓ |
| [PRESTO](https://www.cv.nrao.edu/~sransom/presto/) | ✓ | | ✓ |
| [TEMPO](http://tempo.sourceforge.net/), [tempo2](https://bitbucket.org/psrsoft/tempo2) | ✓ | | ✓ |
| [sigproc](https://github.com/SixByNine/sigproc), [pysigproc](https://github.com/devanshkv/pysigproc) | ✓ | | ✓ |
| [psrcat](https://www.atnf.csiro.au/people/pulsar/psrcat/download.html) | ✓ | | ✓ |
| [riptide](https://github.com/v-morello/riptide) | ✓ | | ✓ |
| [iqrm_apollo](https://gitlab.com/kmrajwade/iqrm_apollo) | ✓ | | ✓ |
| [RFIClean](https://github.com/ymaan4/RFIClean) | ✓ | | ✓ |
| [YAPP](https://github.com/jayanthc/yapp) | ✓ | | ✓ |
| [will](https://github.com/josephwkania/will) | ✓ | | ✓ |

Pick a CUDA 12.6 tag if your driver branch is CUDA 12; any driver from 525 up
runs either. The images are rebuilt and re-tested every Monday, so tags move;
pull by digest (`...radio_transients@sha256:...`) to pin a build. Every
published image is signed with cosign
([details](https://github.com/josephwkania/radio_transients/blob/main/docs/rt-run.md#signatures)).

## rt-run: the tools on your PATH

    pip install radio-transients        # or pipx install radio-transients
    rt-run install                      # full, gpu, cpu or arm; picks one if not given
    export PATH="$PATH:$HOME/.local/share/radio_transients/bin"    # and in ~/.bashrc

    heimdall -f /data/obs/x.fil -dm 0 1000
    your_header.py -f /data/obs/x.fil

Each tool runs in the container, with `--nv` on GPU hosts and the
directories its arguments name bind-mounted. It needs Apptainer or
Singularity and Python 3.8+ on the host.

- `rt-run update` -- pull the latest build
- `rt-run list` -- linked tools, image digest, signature status
- `rt-run uninstall` -- remove the links, and the image if rt-run pulled it
- `rt-run COMMAND` -- run anything else in the image, e.g. `rt-run python script.py`
- `rt-run install --from FILE.sif` -- use an image you already have

It works in Slurm jobs as on a login node:

    #!/bin/bash
    #SBATCH --gres=gpu:1
    heimdall -f /data/obs/x.fil -dm 0 1000 -gpu_id 0

Settings (`RT_BIND`, `RT_HOME`, `RT_VERIFY`, ...), signatures, digests,
safeguards and site notes:
[docs/rt-run.md](https://github.com/josephwkania/radio_transients/blob/main/docs/rt-run.md).

## Using the images directly

    singularity shell --nv -B /data:/mnt radio_transients_gpu.sif
    singularity exec --nv -B /data:/mnt radio_transients_gpu.sif your_heimdall.py -f /mnt/data.fil

- `$HOME` is mounted automatically; `-B /host/dir:/mnt` mounts more.
- GPU tools need `--nv`.
- The Python tools live in the conda env `RT`, which is activated for you.
- `singularity run IMAGE COMMAND` is the same as `exec`. With no command it
  opens a shell, except in the full image, where it runs `your_heimdall.py`.
- `singularity inspect IMAGE` shows the version, build date and the commit
  each tool was built from.

## More

- [Building the images](https://github.com/josephwkania/radio_transients/blob/main/docs/building.md) --
  CUDA versions, build flags, CPU-specific (`MARCH`) builds
- [Development](https://github.com/josephwkania/radio_transients/blob/main/docs/development.md) --
  tests, CI, weekly rebuilds, releasing, the legacy Sylabs images

Bugs and suggestions: open an issue or a pull request.

Thanks to Kshitij Aggarwal for bug reports and suggestions.

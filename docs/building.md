# Building the images

    apptainer build radio_transients.sif Singularity          # full
    apptainer build radio_transients_gpu.sif Singularity.gpu
    apptainer build radio_transients_cpu.sif Singularity.cpu
    apptainer build radio_transients_arm.sif Singularity.arm  # on an aarch64 machine

Building needs root (or `sudo`); `--fakeroot` works where the site allows it.
A built image installs into `rt-run` with `rt-run install --from FILE.sif`.

## CUDA versions

`Singularity` and `Singularity.gpu` take the CUDA version as a build argument,
defaulting to 11.8:

    apptainer build radio_transients.sif Singularity
    apptainer build --build-arg CUDA_VERSION=12.6.3 radio_transients.sif Singularity

TensorFlow and cupy follow it: 2.14 is the last release for CUDA 11.8, 2.15.1
the first that works on 12, and the recipe reads `nvcc --version` to choose.

Any driver from 525 up runs either image, so 11.8 is the safe default. 12.9
would add Blackwell support and ~1.5 G per image with it; CUDA 13 does not
build, as dedisp does not compile against its Thrust.

## Build flags

Each program is built at its upstream's own optimisation level -- mostly
`-O2`, `-O3` where the upstream chose it. Passing `-O3` to everything buys
nothing measurable: Heimdall, for one, ran as fast at `-O2` as at `-O3`.
The recipes never export `CFLAGS`. An exported `CFLAGS` once left psrdada,
sigproc and Heimdall's CUDA host code with no `-O` at all, Heimdall 3.7x
slower for it. Each autoconf build now checks it got `-O2`, and the build
stops if not.

GPU code is built for every major GPU generation (`-arch=all-major`), so one
image runs on any card. Images built before October 2026 ran Heimdall's GPU
code from sm_52 PTX that the driver recompiled at startup. Against them, the
candidates are the same, but slightly more of them differ in the last digit
of S/N: different GPU code rounds differently.

## CPU-specific builds (`MARCH`)

The published images run on any x86-64 (or aarch64) CPU: nothing is built
with `-march`. Four programs whose upstreams use it -- PRESTO, iqrm_apollo,
riptide and RFIClean -- can be built for a particular CPU instead. On a
Xeon E5-2699 v3, `accelsearch` ran 2.6x faster single-threaded with
`x86-64-v3`:

    apptainer build --build-arg MARCH=native rt_cpu.sif Singularity.cpu
    apptainer build --build-arg MARCH=x86-64-v3 rt_cpu.sif Singularity.cpu

`Singularity`, `Singularity.cpu` and `Singularity.arm` take it, and the image
records it in its `Build-march` label. Such an image is only for CPUs that
have what it was built for -- elsewhere it dies with "illegal instruction" --
and fused multiply-adds change floating-point results in the last bits.
Install it with `rt-run install --from`. `rt-run update` pulls the published,
portable image in its place. CI never publishes one: each image built from
these three recipes must say `Build-march: none`.

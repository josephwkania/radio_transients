#!/usr/bin/env python
"""Numerical checks that need a real GPU.

The GPU-less suite proves the CUDA binaries linked and were compiled for the
right architectures. This proves they compute the right answers, by running
each GPU code path against a CPU reference and requiring them to agree.

Prints one pipe-delimited line per check for gpu_test.sh to turn into TAP,
and always exits 0 -- the caller decides what a failure means.

    OK|cupy fft matches numpy|max rel err 3.2e-07
    FAIL|jess GPU dedispersion matches CPU|max abs diff 4.0

Run a subset with --only NAME [NAME...]; list names with --list.
"""

import argparse
import sys
import traceback
import warnings

import numpy as np

warnings.filterwarnings("ignore")

CHECKS = []


def check(name):
    """Register a function as a named check."""

    def decorator(func):
        CHECKS.append((name, func))
        return func

    return decorator


# --------------------------------------------------------------------------
# The device itself
# --------------------------------------------------------------------------


@check("gpu is visible to cupy")
def gpu_visible():
    import cupy as cp

    count = cp.cuda.runtime.getDeviceCount()
    assert count >= 1, "no CUDA devices found"
    device = cp.cuda.Device(0)
    name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    return f"{count} device(s), device 0 is {name}, cc {device.compute_capability}"


@check("cuda runtime and driver are compatible")
def runtime_driver():
    import cupy as cp

    runtime = cp.cuda.runtime.runtimeGetVersion()
    driver = cp.cuda.runtime.driverGetVersion()
    # The driver must be at least as new as the runtime it is asked to serve,
    # unless forward compatibility packages are installed.
    assert driver >= runtime, (
        f"driver {driver} is older than runtime {runtime}; "
        "the container's CUDA is newer than the host driver"
    )
    return f"runtime {runtime}, driver {driver}"


@check("this gpu's compute capability was compiled into libdedisp")
def dedisp_covers_this_gpu():
    """Tie the GPU-less cuobjdump check to the hardware actually present.

    An image whose libdedisp only carries sm_50 will run here via PTX JIT,
    slowly, or not at all. Either way the operator wants to know.
    """
    import glob
    import re
    import subprocess

    import cupy as cp

    libs = [
        path
        for pattern in ("/usr/local/lib/libdedisp.so*", "/usr/lib/libdedisp.so*")
        for path in glob.glob(pattern)
    ]
    assert libs, "no libdedisp found"

    result = subprocess.run(
        ["cuobjdump", "--list-elf", libs[0]],
        capture_output=True,
        text=True,
        check=False,
    )
    # cuobjdump names each embedded cubin, e.g. "libdedisp.4.sm_70.cubin",
    # so the architecture is inside the token rather than at the start of it.
    arches = sorted({m.group(0) for m in re.finditer(r"sm_\d+", result.stdout)})
    assert arches, f"cuobjdump reported no SM architectures in {libs[0]}"

    capability = cp.cuda.Device(0).compute_capability
    major, minor = int(capability[:-1]), int(capability[-1])
    this_gpu = f"sm_{major}{minor}"

    # CUDA guarantees cubin compatibility forward across minor revisions
    # within a major generation, so sm_70 SASS runs on an sm_75 Turing card.
    # Requiring this GPU's exact architecture would fail a container that is
    # in fact perfectly able to run -- as `-arch=all-major` builds are, since
    # they emit only the major architectures.
    usable = sorted(
        arch
        for arch in arches
        if int(arch[3:-1] or 0) == major and int(arch[-1]) <= minor
    )
    assert usable, (
        f"this GPU is {this_gpu} but libdedisp carries only {' '.join(arches)}; "
        "none of those cubins can run on it, so it will fall back to PTX JIT "
        "or fail"
    )
    return f"{this_gpu} runs {' '.join(usable)} (library has {' '.join(arches)})"


@check("libNVVM is present for numba.cuda")
def libnvvm_present():
    """`your`'s GPU path compiles kernels with numba.cuda, which needs
    libNVVM -- a separate file from libcudart. A container can have a
    perfectly good CUDA runtime and still fail here, and every CPU-side
    test will pass while it does.
    """
    from numba import cuda

    assert cuda.is_available(), "numba.cuda reports no usable device"

    @cuda.jit
    def _add_one(out):
        i = cuda.grid(1)
        if i < out.size:
            out[i] += 1

    device_array = cuda.to_device(np.zeros(64, dtype=np.float32))
    _add_one[1, 64](device_array)
    result = device_array.copy_to_host()
    assert np.all(result == 1), "numba.cuda kernel produced the wrong result"
    return "numba.cuda compiled and ran a kernel"


# --------------------------------------------------------------------------
# CuPy kernels against NumPy
# --------------------------------------------------------------------------


@check("cupy elementwise and reductions match numpy")
def cupy_elementwise():
    import cupy as cp

    rng = np.random.default_rng(0)
    a = rng.normal(size=(512, 512)).astype(np.float32)
    b = rng.normal(size=(512, 512)).astype(np.float32)

    expected = np.sqrt(np.abs(a * b + a)).sum(axis=1)
    got = cp.asnumpy(
        cp.sqrt(cp.abs(cp.asarray(a) * cp.asarray(b) + cp.asarray(a))).sum(axis=1)
    )

    err = np.abs(expected - got).max() / np.abs(expected).max()
    assert err < 1e-5, f"max relative error {err:.3g}"
    return f"max rel err {err:.2g}"


@check("cupy fft matches numpy")
def cupy_fft():
    """cuFFT is a separate library from the CUDA runtime, and a common
    casualty of a mismatched CUDA installation."""
    import cupy as cp

    rng = np.random.default_rng(1)
    signal = rng.normal(size=8192).astype(np.float64)

    expected = np.fft.rfft(signal)
    got = cp.asnumpy(cp.fft.rfft(cp.asarray(signal)))

    err = np.abs(expected - got).max() / np.abs(expected).max()
    assert err < 1e-9, f"max relative error {err:.3g}"
    return f"max rel err {err:.2g}"


@check("cupy compiles and runs a raw kernel")
def cupy_raw_kernel():
    """Exercises NVRTC, the runtime compiler. A container can ship a working
    cuFFT and still be unable to compile a kernel at runtime."""
    import cupy as cp

    kernel = cp.RawKernel(
        r"""
        extern "C" __global__
        void scale_add(const float* x, float* y, float a, int n) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i < n) { y[i] = a * x[i] + y[i]; }
        }
        """,
        "scale_add",
    )

    n = 4096
    x = cp.arange(n, dtype=cp.float32)
    y = cp.ones(n, dtype=cp.float32)
    kernel((32,), (128,), (x, y, cp.float32(2.0), n))

    expected = 2.0 * np.arange(n, dtype=np.float32) + 1.0
    err = np.abs(cp.asnumpy(y) - expected).max()
    assert err == 0, f"max abs diff {err}"
    return "NVRTC compiled and ran a kernel exactly"


# --------------------------------------------------------------------------
# The science: does GPU dedispersion recover the injected DM?
# --------------------------------------------------------------------------


def _synthetic():
    """The same synthetic filterbank the rest of the suite uses."""
    from make_filterbank import DEFAULTS, channel_freqs, dispersed_pulse, pulse_samples

    rng = np.random.default_rng(DEFAULTS["seed"])
    data = dispersed_pulse(rng)
    freqs = channel_freqs()
    arrival = pulse_samples(DEFAULTS["nsamples"], DEFAULTS["pulses"])[0]
    return data, freqs, arrival, DEFAULTS


@check("jess GPU dedispersion matches its CPU implementation")
def jess_dedisperse_agrees():
    import cupy as cp
    from jess import dispersion, dispersion_cupy

    data, freqs, _, defaults = _synthetic()

    cpu = dispersion.dedisperse(
        data.astype(np.float32), defaults["dm"], defaults["tsamp"], chan_freqs=freqs
    )
    gpu = cp.asnumpy(
        dispersion_cupy.dedisperse(
            cp.asarray(data, dtype=cp.float32),
            defaults["dm"],
            defaults["tsamp"],
            chan_freqs=cp.asarray(freqs),
        )
    )

    assert cpu.shape == gpu.shape, f"shape mismatch: {cpu.shape} vs {gpu.shape}"
    diff = np.abs(cpu - gpu).max()
    # Both implementations do integer sample shifts, so they should agree
    # exactly rather than merely closely.
    assert diff == 0, f"max abs diff {diff}"
    return f"identical over {cpu.shape} array"


@check("jess GPU dedispersion recovers the injected pulse")
def jess_dedisperse_recovers():
    """Agreement with the CPU version is not enough -- both could be wrong
    in the same way. Assert the pulse actually lands where it was injected.
    """
    import cupy as cp
    from jess import dispersion_cupy

    data, freqs, arrival, defaults = _synthetic()

    dedispersed = cp.asnumpy(
        dispersion_cupy.dedisperse(
            cp.asarray(data, dtype=cp.float32),
            defaults["dm"],
            defaults["tsamp"],
            chan_freqs=cp.asarray(freqs),
        )
    )

    timeseries = dedispersed.sum(axis=1)
    peak = int(np.argmax(timeseries))
    snr = (timeseries.max() - np.median(timeseries)) / timeseries.std()

    assert abs(peak - arrival) <= 2, f"peak at sample {peak}, injected at {arrival}"
    assert snr > 10, f"peak S/N only {snr:.1f}"
    return f"peak at sample {peak} (injected {arrival}), S/N {snr:.0f}"


@check("gpu dedispersion at the wrong dm does not find the pulse")
def jess_wrong_dm_negative_control():
    """A dedisperser that ignores its DM argument would pass every test
    above. Dedispersing at a badly wrong DM must degrade the detection.
    """
    import cupy as cp
    from jess import dispersion_cupy

    data, freqs, _arrival, defaults = _synthetic()

    def peak_snr(dm):
        dedispersed = cp.asnumpy(
            dispersion_cupy.dedisperse(
                cp.asarray(data, dtype=cp.float32),
                dm,
                defaults["tsamp"],
                chan_freqs=cp.asarray(freqs),
            )
        )
        timeseries = dedispersed.sum(axis=1)
        return (timeseries.max() - np.median(timeseries)) / timeseries.std()

    correct = peak_snr(defaults["dm"])
    wrong = peak_snr(defaults["dm"] + 500.0)

    assert correct > 2 * wrong, (
        f"S/N at the correct DM ({correct:.1f}) is not clearly better than "
        f"at a wrong DM ({wrong:.1f}); is the DM argument being used?"
    )
    return (
        f"S/N {correct:.0f} at DM {defaults['dm']} "
        f"vs {wrong:.1f} at DM {defaults['dm'] + 500}"
    )


@check("your GPU dedispersion matches its CPU implementation")
def your_candidate_gpu():
    """This is the code path FETCH's candmaker uses, so it is the one that
    matters for the container's headline workflow."""
    import tempfile

    from your.candidate import Candidate

    from make_filterbank import DEFAULTS, dispersed_pulse, pulse_samples, write

    rng = np.random.default_rng(DEFAULTS["seed"])
    data = dispersed_pulse(rng)
    arrival = pulse_samples(DEFAULTS["nsamples"], DEFAULTS["pulses"])[0]

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/candidate.fil"
        write(
            path,
            data,
            DEFAULTS["nchans"],
            DEFAULTS["tsamp"],
            DEFAULTS["fch1"],
            DEFAULTS["foff"],
        )

        candidate = Candidate(
            fp=path,
            dm=DEFAULTS["dm"],
            tcand=arrival * DEFAULTS["tsamp"],
            width=2,
            label=-1,
            snr=20,
            min_samp=256,
            device=0,
        )
        candidate.get_chunk()

        candidate.dedisperse(target="CPU")
        cpu = np.array(candidate.dedispersed, dtype=np.float64)
        candidate.dedisperse(target="GPU")
        gpu = np.array(candidate.dedispersed, dtype=np.float64)
        dedisp_diff = np.abs(cpu - gpu).max()

        # your's dmtime() takes dmsteps but its GPU branch calls
        # gpu_dmt(self, device=...) without passing it on, so the GPU plane is
        # always gpu_dmt's own default of 256 trials however this is called.
        # Asking for anything else compares planes of different shapes and the
        # comparison dies on a broadcast error rather than on the numbers.
        dmsteps = 256
        candidate.dmtime(dmsteps=dmsteps, target="CPU")
        cpu_dmt = np.array(candidate.dmt, dtype=np.float64)
        candidate.dmtime(dmsteps=dmsteps, target="GPU")
        gpu_dmt = np.array(candidate.dmt, dtype=np.float64)
        dmt_diff = np.abs(cpu_dmt - gpu_dmt).max()

    assert dedisp_diff == 0, f"dedispersed planes differ by {dedisp_diff}"
    assert dmt_diff == 0, f"DM-time planes differ by {dmt_diff}"
    return f"dedispersed and {cpu_dmt.shape} DM-time planes identical"


# --------------------------------------------------------------------------
# TensorFlow, which FETCH runs on
# --------------------------------------------------------------------------


@check("tensorflow sees the gpu")
def tensorflow_sees_gpu():
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    assert gpus, (
        f"tensorflow {tf.__version__} reports no GPU; "
        "check that its CUDA and cuDNN versions match the image"
    )
    return f"tensorflow {tf.__version__} sees {len(gpus)} GPU(s)"


@check("tensorflow computes correctly on the gpu")
def tensorflow_matmul():
    import tensorflow as tf

    rng = np.random.default_rng(2)
    a = rng.normal(size=(256, 256)).astype(np.float32)
    b = rng.normal(size=(256, 256)).astype(np.float32)

    with tf.device("/GPU:0"):
        got = tf.matmul(tf.constant(a), tf.constant(b)).numpy()

    expected = a @ b
    err = np.abs(expected - got).max() / np.abs(expected).max()
    assert err < 1e-5, f"max relative error {err:.3g}"
    return f"256x256 matmul, max rel err {err:.2g}"


# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", nargs="+", metavar="NAME", help="run only these checks"
    )
    parser.add_argument("--list", action="store_true", help="list check names and exit")
    args = parser.parse_args()

    if args.list:
        for name, _ in CHECKS:
            print(name)
        return

    selected = CHECKS
    if args.only:
        wanted = set(args.only)
        selected = [(name, func) for name, func in CHECKS if name in wanted]
        # A misspelled --only would otherwise run nothing and report nothing,
        # which every caller reads as "no failures".
        for name in sorted(wanted - {n for n, _ in CHECKS}):
            print(f"FAIL|{name}|no such check; run --list for the names", flush=True)

    for name, func in selected:
        try:
            detail = func()
        except BaseException:  # noqa: BLE001 -- some libraries raise SystemExit
            exc_type, exc, _ = sys.exc_info()
            detail = (
                "".join(traceback.format_exception_only(exc_type, exc))
                .strip()
                .replace("\n", " ")
            )
            print(f"FAIL|{name}|{detail}", flush=True)
        else:
            print(f"OK|{name}|{detail or ''}", flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

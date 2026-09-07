# Container tests

Verifies that a `radio_transients` image built correctly, on a machine with
**no GPU** — so it can run on an ordinary CI runner.

```sh
apptainer exec -B "$PWD/tests:/tests" radio_transients.sif \
    bash /tests/container_test.sh --variant full
```

The variant is auto-detected if you leave `--variant` off. `--quick` skips the
end-to-end pipeline and the slow TensorFlow/CuPy imports; `--keep` leaves the
scratch directory behind for inspection.

Output is TAP 13. Exit status is 0 only when every non-skipped assertion
passed.

## What it checks

| Layer | Method | Catches |
|---|---|---|
| Binaries exist | `command -v` | a `make install` that silently no-oped |
| Shared libraries resolve | `ldd \| grep 'not found'` | the `apt-get purge` at the end of `%post` taking a runtime library with it |
| Binaries start | run them, treat only loader failures as errors | `undefined symbol` from a version-mismatched library |
| CUDA architectures | `cuobjdump --list-elf libdedisp.so` | the `sm_30` → `all-major` rewrite no longer matching, leaving an image that only runs on Kepler |
| Heimdall linkage | `ldd heimdall` for `libdedisp`, `libcudart`, `libpsrdada` | Heimdall built against the wrong dedisp |
| Python imports | one interpreter, all modules | a compiled extension that built but cannot load |
| Pinned versions | compare installed to what the recipe pins | numpy drifting to 2.x and breaking FETCH; TensorFlow moving off 2.14 and breaking CUDA 11.8; the CuPy wheel not matching the image's CUDA |
| Environment | `$PGPLOT_DIR`, `$TEMPO`, `$PRESTO`, `$PSRCAT_FILE` point at real files | a runtime variable set to a path that got cleaned up |
| End-to-end | `fake` → `header` → `readfile` → `rfifind` → `prepdata` → `realfft` → `single_pulse_search.py` → `your` | tools that load fine but cannot actually process data |
| psrdada | create and destroy a `/dev/shm` ring buffer | a psrdada that links but does not work — no GPU needed |

## What it cannot check

Whether CUDA kernels compute the right numbers. On a GPU-less runner,
`heimdall` failing with "no CUDA capable device" counts as a pass: it proves
the binary loaded and reached `main()`, and nothing more. That is what
`gpu_test.sh` is for.

# GPU tests

Requires an NVIDIA GPU. Run against the published image on a self-hosted
runner, or by hand:

```sh
apptainer exec --nv -B "$PWD/tests:/tests" radio_transients_gpu.sif \
    bash /tests/gpu_test.sh --dm 100
```

`--quick` skips the Heimdall search, which dominates the runtime. `--gpu ID`
picks a device.

## What it checks

Every GPU code path runs against a CPU reference and must agree:

| Check | Assertion |
|---|---|
| Device visible | CuPy sees a device; driver is no older than the runtime |
| Architecture match | the *present* GPU's compute capability is in `libdedisp`'s compiled arch list — ties the GPU-less `cuobjdump` check to real hardware |
| libNVVM | `numba.cuda` can compile and run a kernel. `your`'s GPU path needs libNVVM, which is a separate file from `libcudart` — a container can have a perfect CUDA runtime, pass every CPU test, and still fail here |
| CuPy | elementwise, reductions and `rfft` match NumPy; a `RawKernel` compiles through NVRTC and gives an exact answer |
| jess dedispersion | GPU output is bit-identical to the CPU implementation, **and** recovers the injected pulse at the right sample |
| Negative control | S/N at the correct DM is more than twice that at DM+500 — a dedisperser that ignored its DM argument would pass every other check |
| `your` | `Candidate.dedisperse` and `Candidate.dmtime` agree between `target="CPU"` and `target="GPU"`. This is the path FETCH's candmaker uses |
| TensorFlow | sees the GPU and computes a matmul correctly — the real test of the TF 2.14 / CUDA 11.8 pairing that the CPU suite only version-checks |
| **Heimdall** | three pulses injected at a known DM into 33.5 s of data come back as candidates at that DM and those arrival times |
| Runscript | `your_heimdall.py`, the container's `%runscript`, survives a real file |

The Heimdall check is the headline. A dedispersion kernel with a scaling
error still finds *a* pulse — just at the wrong DM — so asserting that the
brightest candidate sits at the injected DM is the assertion that catches it.
The synthetic file is unambiguous: measured on a T4, the three pulses come
back at S/N ~100 at DM 100 and nothing above S/N 8 at DM 0, 50 or 150.

## Files

- `container_test.sh` — the GPU-less suite
- `gpu_test.sh` — the GPU suite
- `imports.py` — imports every module in one interpreter, so the sweep costs
  seconds rather than minutes
- `gpu_numerics.py` — the CPU-reference comparisons, one named check each;
  `--list` shows the names, `--only NAME` runs a subset
- `make_filterbank.py` — writes a filterbank with pulses dispersed at a
  chosen DM. Fallback for the CPU suite when sigproc's `fake` is
  unavailable, ground truth for the GPU suite
- `check_heimdall_cands.py` — parses Heimdall `.cand` files and asserts the
  injected pulses were recovered

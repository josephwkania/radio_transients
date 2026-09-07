#!/bin/bash
# container_test.sh -- verify a radio_transients container built correctly,
# on a machine that has no GPU.
#
# What this can and cannot prove:
#   CAN  -- every expected binary exists, every shared library it needs
#           resolves, every Python module imports, the CUDA binaries were
#           compiled for the right SM architectures, and the CPU tools
#           actually process a filterbank end to end.
#   CANNOT -- that CUDA kernels produce correct numbers. That needs a GPU.
#
# Usage, from outside the container:
#   apptainer exec radio_transients.sif bash /path/to/container_test.sh
#   apptainer exec -B "$PWD/tests:/tests" rt_gpu.sif bash /tests/container_test.sh -v gpu
#
# Output is TAP version 13, so CI can parse it. Exit status is 0 only when
# every non-skipped assertion passed.

VARIANT=""
QUICK=0
KEEP=0

usage() {
    cat <<'EOF'
Usage: container_test.sh [options]

  -v, --variant NAME   full | cpu | gpu | arm   (default: auto-detect)
  -q, --quick          skip the end-to-end pipeline and slow imports
  -k, --keep           keep the scratch directory for inspection
  -h, --help           this message
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -v|--variant) VARIANT="$2"; shift 2 ;;
        -q|--quick)   QUICK=1; shift ;;
        -k|--keep)    KEEP=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# The %environment block activates the RT conda env for `apptainer exec`, but
# not for `docker run`. Activate it ourselves if it is not already live.
if [ -z "${CONDA_DEFAULT_ENV:-}" ] && [ -f /usr/local/miniconda/bin/activate ]; then
    # shellcheck disable=SC1091
    . /usr/local/miniconda/bin/activate RT
fi

# Singularity bind-mounts $HOME by default, and python puts
# ~/.local/lib/pythonX.Y/site-packages on sys.path ahead of the image's own
# packages. A stray pip install on the host therefore silently shadows what the
# container ships, and the suite reports on the host's package instead of the
# image's. Disable the user site so every import here comes from the image.
export PYTHONNOUSERSITE=1

SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/rt-test.XXXXXX")
# shellcheck disable=SC2317  # reached via `trap cleanup EXIT` below
cleanup() {
    # psrdada leaves shared memory segments behind if a test dies mid-run.
    if command -v dada_db >/dev/null 2>&1; then
        dada_db -k "$DADA_KEY" -d >/dev/null 2>&1
    fi
    if [ "$KEEP" -eq 1 ]; then
        echo "# scratch directory kept at $SCRATCH"
    else
        rm -rf "$SCRATCH"
    fi
}
DADA_KEY=beef
trap cleanup EXIT

# --------------------------------------------------------------------------
# TAP plumbing
# --------------------------------------------------------------------------

TESTS=0; PASSED=0; FAILED=0; SKIPPED=0
FAILURES=()

pass() {
    TESTS=$((TESTS + 1)); PASSED=$((PASSED + 1))
    printf 'ok %d - %s\n' "$TESTS" "$1"
}

fail() {
    TESTS=$((TESTS + 1)); FAILED=$((FAILED + 1))
    FAILURES+=("$1")
    printf 'not ok %d - %s\n' "$TESTS" "$1"
    if [ -n "${2:-}" ]; then
        printf '  ---\n  diagnostic: |\n'
        printf '%s\n' "$2" | head -20 | sed 's/^/    /'
        printf '  ...\n'
    fi
}

skip() {
    TESTS=$((TESTS + 1)); SKIPPED=$((SKIPPED + 1))
    printf 'ok %d - %s # SKIP %s\n' "$TESTS" "$1" "$2"
}

section() { printf '\n# %s\n' "$1"; }

# --------------------------------------------------------------------------
# Assertions
# --------------------------------------------------------------------------

# t_bin NAME -- NAME is on PATH
t_bin() {
    if command -v "$1" >/dev/null 2>&1; then
        pass "binary present: $1"
    else
        fail "binary present: $1" "not found on PATH ($PATH)"
    fi
}

# t_ldd NAME -- every shared library NAME needs actually resolves.
# This is the test that catches the `apt-get purge` at the end of %post
# taking a runtime library with it.
t_ldd() {
    local name="$1" path out missing
    path=$(command -v "$name" 2>/dev/null) || path="$name"
    if [ ! -e "$path" ]; then
        skip "links resolve: $name" "binary not present"
        return
    fi
    out=$(ldd "$path" 2>&1)
    if printf '%s' "$out" | grep -q 'not a dynamic executable'; then
        skip "links resolve: $name" "not a dynamic executable"
        return
    fi
    missing=$(printf '%s\n' "$out" | grep 'not found')
    if [ -n "$missing" ]; then
        fail "links resolve: $name" "$missing"
    else
        pass "links resolve: $name"
    fi
}

# t_runs DESC CMD [ARGS...] -- CMD starts and gets past the dynamic loader.
# A non-zero exit is fine: many of these tools exit 1 after printing usage,
# and a GPU tool on a GPU-less host exits with a CUDA error, which still
# proves the binary loaded. A loader failure or a hang is not fine.
t_runs() {
    local desc="$1"; shift
    local out rc
    out=$(timeout 120 "$@" </dev/null 2>&1); rc=$?
    if [ "$rc" -eq 124 ]; then
        fail "$desc" "timed out after 120s"
    elif [ "$rc" -eq 127 ]; then
        fail "$desc" "exit 127 (command or loader failure)
$out"
    elif printf '%s' "$out" | grep -qE 'error while loading shared libraries|cannot open shared object|symbol lookup error|undefined symbol'; then
        fail "$desc" "$out"
    else
        pass "$desc"
    fi
}

# t_ok DESC CMD [ARGS...] -- CMD must exit 0. For things that really should
# succeed, like a header parse.
t_ok() {
    local desc="$1"; shift
    local out rc
    out=$(timeout 300 "$@" </dev/null 2>&1); rc=$?
    if [ "$rc" -eq 0 ]; then
        pass "$desc"
    else
        fail "$desc" "exit $rc
$out"
    fi
}

# t_file PATH DESC -- PATH exists and is non-empty
t_file() {
    if [ -s "$1" ]; then
        pass "${2:-file exists: $1}"
    else
        fail "${2:-file exists: $1}" "missing or empty: $1"
    fi
}

# t_env VAR [EXPECTED_PATH] -- VAR is set, and if given, points somewhere real
t_env() {
    local var="$1" val
    val=$(eval "printf '%s' \"\${$var:-}\"")
    if [ -z "$val" ]; then
        fail "environment: \$$var" "unset"
    elif [ -n "${2:-}" ] && [ ! -e "$val" ]; then
        fail "environment: \$$var" "set to '$val' which does not exist"
    else
        pass "environment: \$$var = $val"
    fi
}

# --------------------------------------------------------------------------
# Which variant are we in?
# --------------------------------------------------------------------------

if [ -z "$VARIANT" ]; then
    if command -v heimdall >/dev/null 2>&1 && command -v rfifind >/dev/null 2>&1; then
        VARIANT=full
    elif command -v heimdall >/dev/null 2>&1; then
        VARIANT=gpu
    elif [ "$(uname -m)" != "x86_64" ]; then
        VARIANT=arm
    else
        VARIANT=cpu
    fi
fi

case "$VARIANT" in
    full|cpu|gpu|arm) ;;
    *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac

HAS_CPU_STACK=0; HAS_GPU_STACK=0
case "$VARIANT" in
    full)     HAS_CPU_STACK=1; HAS_GPU_STACK=1 ;;
    cpu|arm)  HAS_CPU_STACK=1 ;;
    gpu)      HAS_GPU_STACK=1 ;;
esac

printf 'TAP version 13\n'
printf '# radio_transients container test\n'
printf '# variant  : %s\n' "$VARIANT"
printf '# arch     : %s\n' "$(uname -m)"
printf '# host GPU : %s\n' "$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | head -1 || echo 'none visible (expected)')"

# --------------------------------------------------------------------------
section "environment"
# --------------------------------------------------------------------------

# The arm recipe has its conda block commented out and uses the distribution's
# own python instead, so there is no RT env to find there. Asserting one would
# report a container that is behaving exactly as its recipe intends as broken.
if [ "$VARIANT" = "arm" ]; then
    skip "environment: \$CONDA_DEFAULT_ENV" "the arm variant uses system python, not conda"
    skip "conda env RT is active"            "the arm variant uses system python, not conda"
else
    t_env CONDA_DEFAULT_ENV
    if [ "${CONDA_DEFAULT_ENV:-}" = "RT" ]; then
        pass "conda env RT is active"
    else
        fail "conda env RT is active" "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-<unset>}"
    fi
fi

PYVER=$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)
if [ "$PYVER" = "3.10" ]; then
    pass "python is 3.10 (got $PYVER)"
else
    fail "python is 3.10" "got '${PYVER:-none}'"
fi

if [ "$HAS_CPU_STACK" -eq 1 ]; then
    t_env PGPLOT_DIR check
    t_env TEMPO check
    t_env PRESTO check
    t_env PSRCAT_FILE check
    t_file "${PGPLOT_DIR:-/nonexistent}/grfont.dat" "PGPLOT font file present"
    t_file "${TEMPO:-/nonexistent}/obsys.dat"       "tempo observatory table present"
fi

# The %environment block puts the dedisp libraries on LD_LIBRARY_PATH.
# Note that %post line 52 writes `LD_LIBRARY_PATH=LD_LIBRARY_PATH:...`
# with a missing `$`, so this checks the runtime value, not the build one.
if [ "$HAS_GPU_STACK" -eq 1 ]; then
    case ":${LD_LIBRARY_PATH:-}:" in
        *:/usr/local/lib:*|*:/usr/local/lib/:*)
            pass "LD_LIBRARY_PATH includes /usr/local/lib (dedisp)" ;;
        *)
            fail "LD_LIBRARY_PATH includes /usr/local/lib (dedisp)" \
                 "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}" ;;
    esac
fi

# --------------------------------------------------------------------------
# `nvcc` is not in these images: %post purges build-essential and the
# autoremove that follows takes cuda-nvcc with it -- the same purge the recipe
# already works around for libNVVM. Nothing at runtime needs the compiler
# driver (CuPy compiles through NVRTC, numba through libNVVM), so read the
# toolkit version from the CUDA runtime's own soname instead:
# libcudart.so.11.8.89 -> 11.8.89.
cuda_version() {
    local -a found
    found=(/usr/local/cuda/lib64/libcudart.so.*.*.*)
    [ -e "${found[0]}" ] || found=(/usr/local/cuda/lib64/libcudart.so.*.*)
    [ -e "${found[0]}" ] || return 1
    printf '%s\n' "${found[0]##*/libcudart.so.}"
}

section "binaries on PATH and their shared libraries"
# --------------------------------------------------------------------------

BINS=()

if [ "$HAS_CPU_STACK" -eq 1 ]; then
    # PRESTO
    BINS+=(readfile rfifind prepdata prepfold accelsearch realfft exploredat
           prepsubband single_pulse_search.py DDplan.py)
    # tempo / psrcat
    BINS+=(tempo psrcat)
    # sigproc (SixByNine)
    BINS+=(header filterbank dedisperse fake decimate reader)
    # RFI mitigation
    BINS+=(rficlean iqrm_apollo_cli)
    # your
    BINS+=(your_header.py your_writer.py your_rfimask.py your_h5plotter.py)
    # misc
    BINS+=(htop jupyter)
fi

if [ "$HAS_GPU_STACK" -eq 1 ]; then
    BINS+=(heimdall dada_db dada_junkdb dada_dbdisk dada_diskdb)
    # cuobjdump, not nvcc: the recipe's apt purge removes the compiler driver
    # but leaves the CUDA binary utilities, and cuobjdump is what the
    # architecture check below actually depends on.
    BINS+=(cuobjdump)
    BINS+=(your_heimdall.py predict.py)
    # FETCH's candidate maker ships as candmaker.py in older releases and as
    # your_candmaker.py in newer ones. Either satisfies the requirement, so
    # resolve the name once rather than asserting on a name that moved.
    CANDMAKER=""
    for c in candmaker.py your_candmaker.py; do
        if command -v "$c" >/dev/null 2>&1; then CANDMAKER="$c"; break; fi
    done
    BINS+=("${CANDMAKER:-candmaker.py}")
fi

for b in "${BINS[@]}"; do t_bin "$b"; done

section "shared library resolution"
for b in "${BINS[@]}"; do t_ldd "$b"; done

# YAPP installs a family of binaries whose exact names drift between commits,
# so glob rather than hard-code them.
if [ "$HAS_CPU_STACK" -eq 1 ]; then
    section "YAPP"
    mapfile -t YAPP_BINS < <(compgen -c 'yapp_' 2>/dev/null | sort -u)
    if [ "${#YAPP_BINS[@]}" -ge 5 ]; then
        pass "YAPP binaries installed (${#YAPP_BINS[@]} found)"
        for b in "${YAPP_BINS[@]}"; do t_ldd "$b"; done
    else
        fail "YAPP binaries installed" "found only ${#YAPP_BINS[@]}: ${YAPP_BINS[*]:-none}"
    fi
fi

# --------------------------------------------------------------------------
section "binaries start and print usage"
# --------------------------------------------------------------------------

if [ "$HAS_CPU_STACK" -eq 1 ]; then
    t_runs "readfile runs"          readfile
    t_runs "rfifind runs"           rfifind
    t_runs "prepdata runs"          prepdata
    t_runs "accelsearch runs"       accelsearch
    t_runs "tempo runs"             tempo -h
    t_runs "rficlean runs"          rficlean
    t_runs "iqrm_apollo_cli runs"   iqrm_apollo_cli --help
    t_runs "your_header.py runs"    your_header.py --help
    t_ok   "psrcat queries the catalogue" psrcat -c "name p0 dm" J0534+2200
    t_ok   "jupyter lab reports a version" jupyter lab --version
fi

if [ "$HAS_GPU_STACK" -eq 1 ]; then
    # heimdall on a GPU-less host will fail to find a device. That is a pass:
    # it means the binary loaded, resolved libdedisp/libcudart, and ran main().
    t_runs "heimdall loads and reaches main()" heimdall
    t_runs "dada_db runs"                     dada_db -h
    CUDAVER=$(cuda_version)
    if [ -n "$CUDAVER" ]; then
        pass "cuda runtime reports a version (got $CUDAVER)"
    else
        fail "cuda runtime reports a version" \
             "no libcudart.so.X.Y.Z under /usr/local/cuda/lib64"
    fi
    t_ok   "cuobjdump reports a version"      cuobjdump --version
fi

# --------------------------------------------------------------------------
section "CUDA objects compiled for the expected architectures"
# --------------------------------------------------------------------------

# The Singularity recipe rewrites dedisp's Makefile.inc from `sm_30` to
# `all-major`. Without a GPU we cannot run a kernel, but we can read the
# fatbin and confirm the right SM targets are embedded -- which is the thing
# that silently breaks when that sed stops matching.
if [ "$HAS_GPU_STACK" -eq 1 ]; then
    DEDISP_LIB=""
    for cand in /usr/local/lib/libdedisp.so /usr/local/lib/libdedisp.a \
                /usr/lib/libdedisp.so; do
        [ -e "$cand" ] && { DEDISP_LIB="$cand"; break; }
    done

    if [ -z "$DEDISP_LIB" ]; then
        fail "libdedisp installed" "no libdedisp.{so,a} under /usr/local/lib or /usr/lib"
    else
        pass "libdedisp installed at $DEDISP_LIB"

        if command -v cuobjdump >/dev/null 2>&1; then
            ARCHES=$(cuobjdump --list-elf "$DEDISP_LIB" 2>/dev/null \
                     | grep -oE 'sm_[0-9]+' | sort -u | tr '\n' ' ')
            NARCH=$(printf '%s' "$ARCHES" | wc -w)
            if [ "$NARCH" -ge 3 ]; then
                pass "libdedisp targets multiple SM architectures: $ARCHES"
            else
                fail "libdedisp targets multiple SM architectures" \
                     "found only: ${ARCHES:-none} -- did the sm_30 -> all-major rewrite still apply?"
            fi
            case " $ARCHES " in
                *" sm_80 "*) pass "libdedisp includes sm_80 (Ampere)" ;;
                *) fail "libdedisp includes sm_80 (Ampere)" "found: ${ARCHES:-none}" ;;
            esac
        else
            skip "libdedisp SM architectures" "cuobjdump not available"
        fi
    fi

    if command -v heimdall >/dev/null 2>&1; then
        HD_LIBS=$(ldd "$(command -v heimdall)" 2>/dev/null)
        for lib in libdedisp libcudart libpsrdada; do
            if printf '%s' "$HD_LIBS" | grep -q "$lib"; then
                pass "heimdall links against $lib"
            else
                fail "heimdall links against $lib" "$HD_LIBS"
            fi
        done
    fi
fi

# --------------------------------------------------------------------------
section "Python imports"
# --------------------------------------------------------------------------

PYMODS=(numpy scipy matplotlib astropy pandas h5py numba your jess)

if [ "$HAS_CPU_STACK" -eq 1 ]; then
    PYMODS+=(presto presto.presto presto.filterbank presto.psrfits
             pysigproc riptide will)
fi

if [ "$HAS_GPU_STACK" -eq 1 ]; then
    PYMODS+=(psrdada)
    if [ "$QUICK" -eq 0 ]; then
        # tensorflow and cupy are slow to import; both must work on CPU.
        PYMODS+=(tensorflow cupy)
    fi
fi

IMPORT_HELPER="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/imports.py"
if [ -f "$IMPORT_HELPER" ]; then
    # One interpreter for all modules, rather than ~20 cold starts. The cost
    # of sharing an interpreter is that a module which takes the process down
    # with it -- a segfault in a compiled extension is the realistic case --
    # silently takes every module after it too, so reconcile what came back
    # against what was asked for.
    IMPORT_OUT="$SCRATCH/imports.out"
    IMPORT_ERR="$SCRATCH/imports.err"
    python "$IMPORT_HELPER" "${PYMODS[@]}" >"$IMPORT_OUT" 2>"$IMPORT_ERR"
    IMPORT_RC=$?

    while IFS='|' read -r status mod detail; do
        case "$status" in
            OK)   pass "import $mod${detail:+ ($detail)}" ;;
            FAIL) fail "import $mod" "$detail" ;;
        esac
    done < "$IMPORT_OUT"

    IMPORT_REPORTED="$SCRATCH/imports.names"
    cut -d'|' -f2 "$IMPORT_OUT" > "$IMPORT_REPORTED"
    for m in "${PYMODS[@]}"; do
        if ! grep -qxF "$m" "$IMPORT_REPORTED"; then
            fail "import $m" "the import helper exited $IMPORT_RC without reporting this module
$(tail -20 "$IMPORT_ERR")"
        fi
    done
else
    for m in "${PYMODS[@]}"; do
        t_ok "import $m" python -c "import $m"
    done
fi

# --------------------------------------------------------------------------
section "pinned version constraints from the recipe"
# --------------------------------------------------------------------------

# These are invariants the recipe deliberately establishes, each of which
# breaks quietly when an upstream release moves.

# FETCH needs numpy < 2; %post pins it as the very last pip install, so any
# later package that drags numpy 2 back in silently breaks FETCH.
#
# The constraint belongs to FETCH, not to the image, so only assert it where
# FETCH is actually installed. The cpu variant ships no FETCH and its recipe
# deliberately installs numpy unpinned, where demanding <2 is a test asserting
# a requirement that nothing in the container has.
NPVER=$(python -c 'import numpy; print(numpy.__version__)' 2>/dev/null)
if ! python -c 'import fetch' >/dev/null 2>&1; then
    skip "numpy is <2 as FETCH requires" "FETCH not installed in this variant (numpy ${NPVER:-unknown})"
elif [ -n "$NPVER" ]; then
    case "$NPVER" in
        1.*) pass "numpy is <2 as FETCH requires (got $NPVER)" ;;
        *)   fail "numpy is <2 as FETCH requires" "got $NPVER" ;;
    esac
else
    fail "numpy is <2 as FETCH requires" "numpy did not import"
fi

if [ "$HAS_GPU_STACK" -eq 1 ] && [ "$QUICK" -eq 0 ]; then
    # The recipe pins tensorflow 2.14, the last release built against CUDA 11.8.
    TFVER=$(python -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null)
    case "$TFVER" in
        2.14*) pass "tensorflow is 2.14.x, matching CUDA 11.8 (got $TFVER)" ;;
        "")    fail "tensorflow is 2.14.x" "tensorflow did not import" ;;
        *)     fail "tensorflow is 2.14.x, matching CUDA 11.8" "got $TFVER" ;;
    esac

    # %post derives the CuPy wheel name from `nvcc --version` with a fragile
    # grep. Confirm the installed wheel matches the CUDA in the image. nvcc is
    # gone by this point in the build, so use the runtime soname (see
    # cuda_version) -- otherwise this check silently degrades to a skip.
    CUDA_MAJOR=$(cuda_version | cut -d. -f1)
    CUPY_DIST=$(python -m pip list 2>/dev/null | grep -oE '^cupy[-a-z0-9]*' | head -1)
    if [ -n "$CUDA_MAJOR" ] && [ -n "$CUPY_DIST" ]; then
        if [ "$CUPY_DIST" = "cupy-cuda${CUDA_MAJOR}x" ]; then
            pass "cupy wheel matches CUDA major version ($CUPY_DIST)"
        else
            fail "cupy wheel matches CUDA major version" \
                 "CUDA $CUDA_MAJOR but installed $CUPY_DIST"
        fi
    else
        skip "cupy wheel matches CUDA major version" "could not determine both versions"
    fi

    # cupy imports without a driver; it only needs one to run a kernel.
    t_ok "cupy imports without a GPU present" python -c 'import cupy'
fi

# --------------------------------------------------------------------------
section "end-to-end CPU pipeline"
# --------------------------------------------------------------------------

if [ "$QUICK" -eq 1 ]; then
    skip "end-to-end pipeline" "--quick"
elif [ "$HAS_CPU_STACK" -eq 0 ]; then
    skip "end-to-end pipeline" "no CPU search stack in the $VARIANT variant"
else
    cd "$SCRATCH" || exit 1
    FIL="$SCRATCH/test.fil"

    # Prefer sigproc's own generator: it exercises sigproc as a side effect.
    if command -v fake >/dev/null 2>&1 &&
       timeout 120 fake -nbits 8 -nchans 128 -tsamp 128 -tobs 8 \
                        -period 250.0 -snrpeak 20 -dm 25 > "$FIL" 2>/dev/null &&
       [ -s "$FIL" ]; then
        pass "sigproc fake generated a filterbank"
    else
        # Fall back to writing one with `your` so a broken `fake` does not
        # take the whole pipeline down with it.
        rm -f "$FIL"
        GEN="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/make_filterbank.py"
        if [ -f "$GEN" ] && timeout 300 python "$GEN" "$FIL" >/dev/null 2>&1 && [ -s "$FIL" ]; then
            fail "sigproc fake generated a filterbank" "fake failed; fell back to your"
            pass "fallback filterbank written by your"
        else
            fail "sigproc fake generated a filterbank" "and the your fallback also failed"
        fi
    fi

    if [ -s "$FIL" ]; then
        t_ok "sigproc header parses the filterbank"  header "$FIL"
        t_ok "your_header.py parses the filterbank"  your_header.py -f "$FIL"
        t_ok "PRESTO readfile parses the filterbank" readfile "$FIL"

        t_ok "PRESTO rfifind produces a mask" \
            rfifind -time 0.5 -o "$SCRATCH/rfi" "$FIL"
        t_file "$SCRATCH/rfi_rfifind.mask" "rfifind mask written"

        t_ok "PRESTO prepdata dedisperses to a time series" \
            prepdata -nobary -dm 25 -o "$SCRATCH/dm25" "$FIL"
        t_file "$SCRATCH/dm25.dat" "prepdata time series written"

        t_ok "PRESTO realfft transforms the time series" realfft "$SCRATCH/dm25.dat"
        t_file "$SCRATCH/dm25.fft" "realfft output written"

        t_ok "PRESTO single_pulse_search.py runs" \
            single_pulse_search.py -p "$SCRATCH/dm25.dat"

        t_ok "your reads the filterbank through its Python API" \
            python -c "
import sys, your
y = your.Your('$FIL')
d = y.get_data(0, 128)
assert d.shape[0] == 128, d.shape
assert d.shape[1] == y.your_header.nchans, d.shape
"
        if command -v yapp_viewmetadata >/dev/null 2>&1; then
            t_ok "YAPP reads the filterbank metadata" yapp_viewmetadata "$FIL"
        else
            skip "YAPP reads the filterbank metadata" "yapp_viewmetadata not installed"
        fi
    else
        skip "pipeline stages" "no test filterbank could be generated"
    fi
    cd / || true
fi

# --------------------------------------------------------------------------
section "psrdada ring buffer (CPU only, no GPU needed)"
# --------------------------------------------------------------------------

if [ "$HAS_GPU_STACK" -eq 0 ]; then
    skip "psrdada ring buffer" "psrdada not in the $VARIANT variant"
elif [ ! -w /dev/shm ]; then
    skip "psrdada ring buffer" "/dev/shm is not writable in this container"
elif ! command -v dada_db >/dev/null 2>&1; then
    skip "psrdada ring buffer" "dada_db not installed"
else
    # Creating and destroying a shared-memory ring buffer exercises the whole
    # psrdada library without touching a GPU.
    if timeout 60 dada_db -k "$DADA_KEY" -b 16384 -n 4 >/dev/null 2>&1; then
        pass "dada_db created a ring buffer"
        t_ok "dada_db destroyed the ring buffer" dada_db -k "$DADA_KEY" -d
    else
        fail "dada_db created a ring buffer" \
             "$(dada_db -k "$DADA_KEY" -b 16384 -n 4 2>&1 | head -10)"
    fi

    if python -c 'import psrdada' >/dev/null 2>&1; then
        t_ok "psrdada-python connects to a ring buffer" python -c "
import subprocess, psrdada
subprocess.run(['dada_db','-k','$DADA_KEY','-b','16384','-n','4'], check=True,
               capture_output=True)
try:
    r = psrdada.Reader()
    r.connect(0x$DADA_KEY)
    r.disconnect()
finally:
    subprocess.run(['dada_db','-k','$DADA_KEY','-d'], capture_output=True)
"
    else
        skip "psrdada-python connects to a ring buffer" "psrdada module not importable"
    fi
fi

# --------------------------------------------------------------------------
section "FETCH (CPU inference path)"
# --------------------------------------------------------------------------

if [ "$HAS_GPU_STACK" -eq 0 ]; then
    skip "FETCH CLI" "FETCH not in the $VARIANT variant"
elif [ "$QUICK" -eq 1 ]; then
    skip "FETCH CLI" "--quick (loading tensorflow is slow)"
else
    # predict.py imports tensorflow, which must fall back to CPU cleanly.
    t_runs "FETCH predict.py loads"   predict.py --help
    if [ -n "${CANDMAKER:-}" ]; then
        t_runs "FETCH $CANDMAKER loads" "$CANDMAKER" --help
    else
        fail "FETCH candmaker loads" \
             "neither candmaker.py nor your_candmaker.py is on PATH"
    fi
fi

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

printf '\n1..%d\n' "$TESTS"
printf '# passed  %d\n' "$PASSED"
printf '# failed  %d\n' "$FAILED"
printf '# skipped %d\n' "$SKIPPED"

if [ "$FAILED" -gt 0 ]; then
    printf '#\n# failures:\n'
    for f in "${FAILURES[@]}"; do printf '#   - %s\n' "$f"; done
    exit 1
fi

exit 0

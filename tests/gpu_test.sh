#!/bin/bash
# gpu_test.sh -- the half of the container's behaviour that container_test.sh
# cannot reach: whether the CUDA code produces correct numbers.
#
# container_test.sh treats "no CUDA capable device" as a pass, because on a
# GPU-less runner that only proves the binary loaded. This suite requires a
# GPU and asserts on the science: a pulse injected at a known DM must come
# back out of Heimdall at that DM, and every GPU code path must agree with a
# CPU reference.
#
# Usage, on a machine with an NVIDIA GPU:
#   apptainer exec --nv -B "$PWD/tests:/tests" rt_gpu.sif bash /tests/gpu_test.sh
#
# Output is TAP version 13. Exit status is 0 only when every non-skipped
# assertion passed.

QUICK=0
KEEP=0
DM=100
GPU_ID=0

usage() {
    cat <<'EOF'
Usage: gpu_test.sh [options]

  -d, --dm VALUE     DM to inject and search for (default: 100)
  -g, --gpu ID       GPU to run on (default: 0)
  -q, --quick        skip the Heimdall search, which dominates the runtime
  -k, --keep         keep the scratch directory for inspection
  -h, --help         this message
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -d|--dm)    DM="$2"; shift 2 ;;
        -g|--gpu)   GPU_ID="$2"; shift 2 ;;
        -q|--quick) QUICK=1; shift ;;
        -k|--keep)  KEEP=1; shift ;;
        -h|--help)  usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

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

TESTS_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/rt-gpu-test.XXXXXX")
# shellcheck disable=SC2317  # reached via `trap cleanup EXIT` below
cleanup() {
    if [ "$KEEP" -eq 1 ]; then
        echo "# scratch directory kept at $SCRATCH"
    else
        rm -rf "$SCRATCH"
    fi
}
trap cleanup EXIT

# --------------------------------------------------------------------------
# TAP plumbing, same shape as container_test.sh
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
        printf '%s\n' "$2" | head -30 | sed 's/^/    /'
        printf '  ...\n'
    fi
}

skip() {
    TESTS=$((TESTS + 1)); SKIPPED=$((SKIPPED + 1))
    printf 'ok %d - %s # SKIP %s\n' "$TESTS" "$1" "$2"
}

section() { printf '\n# %s\n' "$1"; }

t_ok() {
    local desc="$1"; shift
    local out rc
    out=$(timeout 1800 "$@" </dev/null 2>&1); rc=$?
    if [ "$rc" -eq 0 ]; then
        pass "$desc"
    else
        fail "$desc" "exit $rc
$out"
    fi
}

printf 'TAP version 13\n'
printf '# radio_transients GPU test\n'

# --------------------------------------------------------------------------
section "a GPU is actually present"
# --------------------------------------------------------------------------

# Everything below is meaningless without one, so bail rather than emit a
# hundred confusing failures. This is the case where someone forgot --nv.
if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi is available" \
         "no nvidia-smi in the container -- was it run without --nv?"
    printf '\n1..%d\n# passed  %d\n# failed  %d\n# skipped %d\n' \
           "$TESTS" "$PASSED" "$FAILED" "$SKIPPED"
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total \
                      --format=csv,noheader 2>&1)
if [ -n "$GPU_INFO" ] && ! printf '%s' "$GPU_INFO" | grep -qi 'failed\|error'; then
    pass "nvidia-smi sees a GPU: $GPU_INFO"
else
    fail "nvidia-smi sees a GPU" "$GPU_INFO"
    printf '\n1..%d\n# passed  %d\n# failed  %d\n# skipped %d\n' \
           "$TESTS" "$PASSED" "$FAILED" "$SKIPPED"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"

# --------------------------------------------------------------------------
section "GPU numerics: every CUDA path against a CPU reference"
# --------------------------------------------------------------------------

# gpu_numerics.py imports make_filterbank, so it has to run from tests/.
NUMERICS_OUT="$SCRATCH/gpu_numerics.out"
NUMERICS_ERR="$SCRATCH/gpu_numerics.err"

# Ask first which checks it intends to run. Reconciling that list against the
# lines it actually prints is what turns an interpreter that died halfway --
# a segfault inside a CUDA library, say -- into a visible failure rather than
# a section that quietly asserts nothing and lets the suite exit 0.
EXPECTED_CHECKS=$(cd "$TESTS_DIR" && timeout 300 python gpu_numerics.py --list \
                      2>"$NUMERICS_ERR")

if [ -z "$EXPECTED_CHECKS" ]; then
    fail "gpu_numerics.py can list its checks" \
         "$(tail -20 "$NUMERICS_ERR")"
else
    (cd "$TESTS_DIR" && timeout 1800 python gpu_numerics.py) \
        >"$NUMERICS_OUT" 2>"$NUMERICS_ERR"
    NUMERICS_RC=$?

    while IFS='|' read -r status name detail; do
        case "$status" in
            OK)   pass "$name${detail:+ -- $detail}" ;;
            FAIL) fail "$name" "$detail" ;;
        esac
    done < "$NUMERICS_OUT"

    # A check that never reported is a check that never ran. Blame it on the
    # interpreter's exit status and its stderr, which is the only place the
    # real cause -- an OOM kill, a missing shared library -- shows up.
    REPORTED="$SCRATCH/gpu_numerics.names"
    cut -d'|' -f2 "$NUMERICS_OUT" > "$REPORTED"
    while IFS= read -r name; do
        [ -n "$name" ] || continue
        if ! grep -qxF "$name" "$REPORTED"; then
            fail "$name" "gpu_numerics.py exited $NUMERICS_RC without reporting this check
$(tail -20 "$NUMERICS_ERR")"
        fi
    done <<< "$EXPECTED_CHECKS"
fi

# --------------------------------------------------------------------------
section "Heimdall recovers an injected pulse at the right DM"
# --------------------------------------------------------------------------

if [ "$QUICK" -eq 1 ]; then
    skip "Heimdall search" "--quick"
elif ! command -v heimdall >/dev/null 2>&1; then
    skip "Heimdall search" "heimdall not installed in this variant"
else
    FIL="$SCRATCH/heimdall.fil"
    CANDS="$SCRATCH/cands"
    mkdir -p "$CANDS"

    # 131072 samples at 256 us is 33.5 s of data -- long enough for Heimdall's
    # default gulp and its baseline estimation, small enough to stay fast.
    #
    # --width 2 matters. A single-sample delta per channel leaves each channel
    # holding one isolated 10-sigma sample, which Heimdall's narrow-band RFI
    # excision (-rfi_tol 5, on by default) removes -- the search then finds
    # nothing and the failure looks like broken dedispersion. Sigma 2 samples
    # keeps the per-channel excursion near 3.75 sigma, under that threshold,
    # and still recovers at S/N ~65. Testing against Heimdall's defaults is
    # the point: it is what an operator actually runs.
    GEN_OUT=$(timeout 600 python "$TESTS_DIR/make_filterbank.py" "$FIL" \
                  --dm "$DM" --nsamples 131072 --pulses 3 \
                  --amplitude 30 --width 2 2>&1)
    if [ -s "$FIL" ]; then
        pass "generated a 33.5 s filterbank with 3 pulses at DM $DM"
    else
        fail "generated a 33.5 s filterbank with 3 pulses at DM $DM" "$GEN_OUT"
    fi

    PULSE_TIMES=$(printf '%s\n' "$GEN_OUT" | sed -n 's/^PULSE_TIMES=//p')

    if [ -s "$FIL" ]; then
        # Search a DM range that straddles the injected value, so recovering
        # the right DM is a real result rather than the only option.
        DM_LO=0
        DM_HI=$(python -c "print(int($DM * 3))")

        HD_OUT=$(cd "$CANDS" && timeout 1800 heimdall \
                    -f "$FIL" \
                    -dm "$DM_LO" "$DM_HI" \
                    -gpu_id 0 \
                    -nsamps_gulp 65536 \
                    -output_dir "$CANDS" 2>&1)
        HD_RC=$?

        if [ "$HD_RC" -eq 0 ]; then
            pass "heimdall ran to completion on the GPU"
        else
            fail "heimdall ran to completion on the GPU" "exit $HD_RC
$HD_OUT"
        fi

        # The headline assertion. A dedispersion kernel with a scaling error
        # still finds a pulse -- just at the wrong DM.
        t_ok "heimdall recovered the pulses at DM $DM" \
            python "$TESTS_DIR/check_heimdall_cands.py" "$CANDS" \
                --dm "$DM" --times "$PULSE_TIMES" --min-snr 8
    else
        skip "heimdall search" "no test filterbank could be generated"
    fi
fi

# --------------------------------------------------------------------------
section "the container's own runscript"
# --------------------------------------------------------------------------

# %runscript is `exec your_heimdall.py`, so this is what a user gets by
# running the .sif directly. It is worth knowing it survives a real file.
if [ "$QUICK" -eq 1 ]; then
    skip "your_heimdall.py end to end" "--quick"
elif ! command -v your_heimdall.py >/dev/null 2>&1; then
    skip "your_heimdall.py end to end" "not installed in this variant"
elif [ ! -s "$SCRATCH/heimdall.fil" ]; then
    skip "your_heimdall.py end to end" "no test filterbank available"
else
    YH_DIR="$SCRATCH/your_heimdall"
    mkdir -p "$YH_DIR"
    t_ok "your_heimdall.py runs the full search" \
        timeout 1800 your_heimdall.py -f "$SCRATCH/heimdall.fil" \
            -dm 0 "$(python -c "print(int($DM * 3))")" \
            -g 0 -o "$YH_DIR"
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

#!/usr/bin/env python
"""Assert that Heimdall recovered the pulses make_filterbank.py injected.

This is the test the GPU-less suite cannot do. There, Heimdall failing with
"no CUDA capable device" counts as a pass, because it only proves the binary
loaded. Here we give it a filterbank with a pulse at a known DM and known
arrival time and require the candidate to come back in the right place.

Heimdall writes whitespace-separated `.cand` files with the columns:

    S/N  sample  time(s)  filter  dm_trial  DM  members  begin  end

    python check_heimdall_cands.py CANDDIR --dm 100 --times 8.4,16.8,25.2
"""

import argparse
import glob
import os
import sys

# Column positions in a Heimdall .cand file.
COL_SNR = 0
COL_TIME = 2
COL_FILTER = 3
COL_DM = 5
MIN_COLUMNS = 6


class Candidate:
    __slots__ = ("snr", "time", "width_log2", "dm")

    def __init__(self, fields):
        self.snr = float(fields[COL_SNR])
        self.time = float(fields[COL_TIME])
        self.width_log2 = int(float(fields[COL_FILTER]))
        self.dm = float(fields[COL_DM])

    def __str__(self):
        return (
            f"S/N {self.snr:6.1f}  t {self.time:9.4f}s  "
            f"DM {self.dm:8.2f}  boxcar {2 ** self.width_log2:4d} samples"
        )


def load(directory):
    candidates = []
    paths = sorted(glob.glob(os.path.join(directory, "*.cand")))
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) < MIN_COLUMNS:
                    continue
                try:
                    candidates.append(Candidate(fields))
                except ValueError:
                    # A malformed row is not worth failing the whole run over;
                    # the assertions below will catch a genuinely empty result.
                    continue
    return paths, candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="directory containing Heimdall .cand files")
    parser.add_argument("--dm", type=float, required=True, help="injected DM")
    parser.add_argument(
        "--dm-tol",
        type=float,
        default=None,
        help="allowed DM error; defaults to 10%% of --dm or 5, whichever is larger",
    )
    parser.add_argument(
        "--times",
        default="",
        help="comma-separated injected arrival times in seconds",
    )
    parser.add_argument(
        "--time-tol",
        type=float,
        default=0.5,
        help="allowed arrival time error, seconds",
    )
    parser.add_argument("--min-snr", type=float, default=8.0)
    args = parser.parse_args()

    dm_tol = args.dm_tol if args.dm_tol is not None else max(5.0, 0.1 * args.dm)

    paths, candidates = load(args.directory)
    print(
        f"# read {len(candidates)} candidates "
        f"from {len(paths)} file(s) in {args.directory}"
    )

    if not candidates:
        print(f"FAIL: Heimdall produced no candidates (files: {paths or 'none'})")
        return 1

    failures = []

    brightest = max(candidates, key=lambda c: c.snr)
    print(f"# brightest: {brightest}")

    if brightest.snr < args.min_snr:
        failures.append(f"brightest candidate S/N {brightest.snr:.1f} < {args.min_snr}")

    # The headline assertion: the strongest thing Heimdall found is at the DM
    # we injected. A dedisperser with a scaling error still finds *a* pulse,
    # just at the wrong DM, so this is the check that catches it.
    if abs(brightest.dm - args.dm) > dm_tol:
        failures.append(
            f"brightest candidate is at DM {brightest.dm:.2f}, "
            f"injected at DM {args.dm:.2f} (tolerance {dm_tol:.2f})"
        )

    if args.times:
        wanted = [float(t) for t in args.times.split(",") if t.strip()]
        on_dm = [c for c in candidates if abs(c.dm - args.dm) <= dm_tol]
        for arrival in wanted:
            matched = [c for c in on_dm if abs(c.time - arrival) <= args.time_tol]
            if matched:
                best = max(matched, key=lambda c: c.snr)
                print(f"# pulse at {arrival:.4f}s recovered: {best}")
            else:
                failures.append(
                    f"no candidate within {args.time_tol}s of the pulse injected "
                    f"at {arrival:.4f}s (at DM {args.dm} +/- {dm_tol})"
                )

    if failures:
        for message in failures:
            print(f"FAIL: {message}")
        print("# all candidates:")
        for candidate in sorted(candidates, key=lambda c: -c.snr)[:20]:
            print(f"#   {candidate}")
        return 1

    print(
        f"PASS: brightest candidate at DM {brightest.dm:.2f} "
        f"(injected {args.dm:.2f}), S/N {brightest.snr:.1f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

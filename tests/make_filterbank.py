#!/usr/bin/env python
"""Write a SIGPROC filterbank containing pulses dispersed at a known DM.

Used two ways:

  * by container_test.sh, as a fallback when sigproc's own `fake` is
    unavailable, so a failure in one package does not take the whole
    end-to-end pipeline test down with it;
  * by gpu_test.sh, as ground truth -- the GPU tests assert that Heimdall
    and the GPU dedispersion routines recover the DM injected here.

Because it writes through `your`, it also doubles as a test of your's
filterbank writer.

    python make_filterbank.py test.fil
    python make_filterbank.py heimdall.fil --dm 100 --nsamples 131072 --pulses 3
"""

import argparse
import os
import sys

import numpy as np
from your.formats.filwriter import make_sigproc_object

# Dispersion delay in seconds for frequencies in MHz and DM in pc/cm^3.
DISPERSION_CONSTANT = 4148.808

DEFAULTS = dict(
    nchans=128,
    nsamples=4096,
    tsamp=256e-6,
    fch1=1500.0,
    foff=-1.0,
    dm=25.0,
    amplitude=60.0,
    width=0.0,
    pulses=1,
    seed=1234,
)


def channel_freqs(
    nchans=DEFAULTS["nchans"], fch1=DEFAULTS["fch1"], foff=DEFAULTS["foff"]
):
    """Centre frequency of each channel, in MHz, highest first."""
    return fch1 + foff * np.arange(nchans)


def pulse_samples(nsamples, pulses):
    """Sample index at which each pulse arrives in the top channel.

    Spread evenly through the file, but kept clear of the ends so the
    dispersion sweep does not run off the bottom of the band.
    """
    return [int(nsamples * (i + 1) / (pulses + 1)) for i in range(pulses)]


def dispersed_pulse(
    rng,
    nchans=DEFAULTS["nchans"],
    nsamples=DEFAULTS["nsamples"],
    tsamp=DEFAULTS["tsamp"],
    fch1=DEFAULTS["fch1"],
    foff=DEFAULTS["foff"],
    dm=DEFAULTS["dm"],
    amplitude=DEFAULTS["amplitude"],
    pulses=DEFAULTS["pulses"],
    width=DEFAULTS["width"],
):
    """Gaussian noise with pulses smeared across the band by `dm`.

    `width` is the Gaussian sigma of the pulse in samples. Zero injects a
    single-sample delta into each channel, which is the cheapest thing to
    assert on but is not what a real pulse looks like: at a DM where the
    sweep runs at about one channel per sample, a delta leaves every channel
    holding one isolated high-sigma sample. That is indistinguishable from
    impulsive narrow-band RFI, and Heimdall's narrow-band excision -- on by
    default at -rfi_tol 5 -- removes it, so the search finds nothing. Giving
    the pulse a finite width keeps the per-channel, per-sample excursion
    below that threshold while preserving the integrated signal.
    """
    data = rng.normal(loc=128.0, scale=8.0, size=(nsamples, nchans))

    freqs = channel_freqs(nchans, fch1, foff)
    # Delay of each channel relative to the top of the band.
    delays = DISPERSION_CONSTANT * dm * (freqs**-2 - fch1**-2)
    sample_offsets = np.round(delays / tsamp).astype(int)

    for arrival in pulse_samples(nsamples, pulses):
        for channel, offset in enumerate(sample_offsets):
            centre = arrival + offset
            if width <= 0:
                if 0 <= centre < nsamples:
                    data[centre, channel] += amplitude
                continue
            lo = max(0, int(centre - 4 * width))
            hi = min(nsamples, int(centre + 4 * width) + 1)
            if hi > lo:
                samples = np.arange(lo, hi)
                data[lo:hi, channel] += amplitude * np.exp(
                    -0.5 * ((samples - centre) / width) ** 2
                )

    return np.clip(data, 0, 255).astype(np.uint8)


def write(path, data, nchans, tsamp, fch1, foff):
    sigproc_object = make_sigproc_object(
        # SIGPROC's read_header reads strings into a fixed 80 character
        # buffer, so store the bare filename rather than the full path --
        # a long scratch directory would otherwise corrupt the header for
        # every reader except `your`.
        rawdatafile=os.path.basename(path),
        source_name="TEST_SOURCE",
        nchans=nchans,
        foff=foff,
        fch1=fch1,
        tsamp=tsamp,
        tstart=59000.0,
        src_raj=112233.44,
        src_dej=112233.44,
        machine_id=0,
        nbeams=1,
        ibeam=0,
        nbits=8,
        nifs=1,
        barycentric=0,
        pulsarcentric=0,
        telescope_id=6,
        data_type=0,
        az_start=-1,
        za_start=-1,
    )
    sigproc_object.write_header(path)
    sigproc_object.append_spectra(data, path)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="path of the filterbank to write")
    parser.add_argument("--dm", type=float, default=DEFAULTS["dm"])
    parser.add_argument("--nchans", type=int, default=DEFAULTS["nchans"])
    parser.add_argument("--nsamples", type=int, default=DEFAULTS["nsamples"])
    parser.add_argument("--tsamp", type=float, default=DEFAULTS["tsamp"])
    parser.add_argument("--fch1", type=float, default=DEFAULTS["fch1"])
    parser.add_argument("--foff", type=float, default=DEFAULTS["foff"])
    parser.add_argument("--amplitude", type=float, default=DEFAULTS["amplitude"])
    parser.add_argument(
        "--width",
        type=float,
        default=DEFAULTS["width"],
        help="Gaussian pulse sigma in samples; 0 gives a single-sample delta",
    )
    parser.add_argument("--pulses", type=int, default=DEFAULTS["pulses"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(seed=args.seed)
    data = dispersed_pulse(
        rng,
        nchans=args.nchans,
        nsamples=args.nsamples,
        tsamp=args.tsamp,
        fch1=args.fch1,
        foff=args.foff,
        dm=args.dm,
        amplitude=args.amplitude,
        pulses=args.pulses,
        width=args.width,
    )
    write(args.output, data, args.nchans, args.tsamp, args.fch1, args.foff)

    arrivals = pulse_samples(args.nsamples, args.pulses)
    print(
        f"wrote {args.output}: {args.nsamples} samples x {args.nchans} channels, "
        f"DM={args.dm}, {args.pulses} pulse(s)"
    )
    # gpu_test.sh reads these back to know what Heimdall ought to recover.
    print(f"DM={args.dm}")
    print("PULSE_TIMES=" + ",".join(f"{s * args.tsamp:.6f}" for s in arrivals))


if __name__ == "__main__":
    sys.exit(main())

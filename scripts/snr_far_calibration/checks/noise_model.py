"""Measure the peak-search excess of bayestar-realize-coincs (PEAK_OFFSET).

threshold.network_snr draws each detector's SNR as |rho_opt + n|.
bayestar-realize-coincs instead adds the noise to a 0.1 s SNR time series and
keeps its maximum, which is slightly larger. This script runs ligo.skymap's
simulate_snr at fixed optimal SNR and prints the mean difference.

Templates follow bayestar-inject (o2-uberbank: TaylorF2 below 4 Msun total
mass, SEOBNRv4_ROM above), which needs SEOBNRv4ROM_v3.0.hdf5 in LAL_DATA_PATH.

    python checks/noise_model.py ../../data/runs/O4a/psds.xml --detector L1
"""

import argparse
import logging

import lal
import lal.series
import numpy as np
from igwn_ligolw import utils as ligolw_utils
from ligo.skymap.bayestar import filter as bayestar_filter
from ligo.skymap.tool.bayestar_realize_coincs import simulate_snr

# Template masses, detector frame [Msun]: a 1.4 + 1.4 Msun BNS at z = 0.07, an
# NSBH and a BBH. They only set the template shape; the distance is chosen to
# give each optimal SNR, so the frame does not affect the result.
SOURCES = ((1.5, 1.5), (10.0, 1.5), (30.0, 25.0))
OPTIMAL_SNRS = (6.0, 8.0)
F_LOW = 25.0
SKY = dict(ra=1.0, dec=0.3, psi=0.2, inc=0.0)
DETECTORS = {
    "H1": lal.LHO_4K_DETECTOR,
    "L1": lal.LLO_4K_DETECTOR,
    "V1": lal.VIRGO_DETECTOR,
}

logging.getLogger("BAYESTAR").setLevel(logging.ERROR)


def load_psd(path, detector):
    xmldoc = ligolw_utils.load_filename(
        path, contenthandler=lal.series.PSDContentHandler
    )
    psd = lal.series.read_psd_xmldoc(xmldoc, root_name=None)[detector]
    return bayestar_filter.InterpolatedPSD(bayestar_filter.abscissa(psd), psd.data.data)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("psd_xml")
    parser.add_argument("--detector", default="L1", choices=list(DETECTORS))
    parser.add_argument("--draws", type=int, default=400)
    args = parser.parse_args()

    psd = load_psd(args.psd_xml, args.detector)
    detector = lal.CachedDetectors[DETECTORS[args.detector]]
    epoch = lal.LIGOTimeGPS(1380000000)
    gmst = lal.GreenwichMeanSiderealTime(epoch)
    fplus, fcross = lal.ComputeDetAMResponse(
        detector.response, SKY["ra"], SKY["dec"], SKY["psi"], gmst
    )
    cosi = np.cos(SKY["inc"])
    antenna = abs(0.5 * (1 + cosi**2) * fplus + 1j * cosi * fcross)
    rng = np.random.default_rng(0)

    def simulate(template, distance, mode):
        return simulate_snr(
            SKY["ra"],
            SKY["dec"],
            SKY["psi"],
            SKY["inc"],
            distance,
            epoch,
            gmst,
            template,
            psd,
            detector.response,
            detector.location,
            mode,
        )

    print(
        f"{'masses':>11} {'rho_opt':>7} {'bayestar':>9} {'|rho+n|':>8} "
        f"{'excess':>7} {'error':>6}"
    )
    for mass1, mass2 in SOURCES:
        template = bayestar_filter.sngl_inspiral_psd(
            "o2-uberbank", mass1=mass1, mass2=mass2, f_min=F_LOW
        )
        horizon = simulate(template, 1.0, "zero-noise")[0]
        for rho in OPTIMAL_SNRS:
            peaks = []
            for seed in range(args.draws):
                np.random.seed(seed)  # bayestar draws from numpy's global generator
                peaks.append(
                    simulate(template, horizon * antenna / rho, "gaussian-noise")[1]
                )
            formula = np.hypot(
                rho + rng.standard_normal(200_000), rng.standard_normal(200_000)
            )
            print(
                f"{mass1:5.1f}+{mass2:<5.1f} {rho:7.1f} {np.mean(peaks):9.3f} "
                f"{formula.mean():8.3f} {np.mean(peaks) - formula.mean():+7.3f} "
                f"{np.std(peaks) / np.sqrt(len(peaks)):6.3f}"
            )


if __name__ == "__main__":
    main()

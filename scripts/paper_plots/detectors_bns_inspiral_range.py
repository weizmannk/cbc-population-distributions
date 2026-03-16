# coding: utf-8
"""
---------------------------------------------------------------------------------------------------
                                    ABOUT
@author         : Ramodgwendé Weizmann KIENDREBEOGO
@email          : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
@repo           : https://github.com/weizmannk/bns-inspiral-range
@createdOn      : February 2026
@description    : Computes the Binary Neutron Star (BNS) inspiral range for
                  observing runs O4 and O5 (a/b/c), using SNR=8 and a
                  1.4 solar mass binary system.
---------------------------------------------------------------------------------------------------
"""

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gwpy.astro import inspiral_range
from gwpy.frequencyseries import FrequencySeries

# ============================================================================
# Logging / warning configuration
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="No LaTeX-compatible font")
logging.getLogger("matplotlib").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class BNSInspiralRangeCalculator:
    """BNS inspiral range calculator for O4 and O5 observing runs.

    Parameters
    ----------
    run_names : list[str]
        Observing run identifiers (e.g., ``["O4-HL", "O5a-HLV"]``).
    ifos : list[str]
        Interferometer names (e.g., ``["H1", "L1", "V1"]``).
    path : str or pathlib.Path
        Directory containing the sensitivity data files.
    data_type : str, optional
        ``"ASD"`` or ``"PSD"``. Defaults to ``"ASD"``.
    """

    def __init__(self, run_names, ifos, path, data_type="ASD"):
        self.run_names = run_names
        self.ifos = ifos
        self.path = Path(path)
        self.data_type = data_type

        # Format: (filename, column_name or None)
        self.sensitivity_files = {
            "O4-HL": {
                "H1": ("o4b_h1_ref.txt", None),
                "L1": ("o4b_l1_ref.txt", None),
            },
            "O4-HLV": {
                "H1": ("o4b_h1_ref.txt", None),
                "L1": ("o4b_l1_ref.txt", None),
                "V1": ("o4b_v1_ref.txt", None),
            },
            "O5a-HL": {
                "H1": ("O5StrainCurves_freqabc.txt", "O5aStrain"),
                "L1": ("O5StrainCurves_freqabc.txt", "O5aStrain"),
            },
            "O5a-HLV": {
                "H1": ("O5StrainCurves_freqabc.txt", "O5aStrain"),
                "L1": ("O5StrainCurves_freqabc.txt", "O5aStrain"),
                "V1": ("25115_O5Tier1LowSensASD.txt", None),
            },
            "O5b-HLV": {
                "H1": ("O5StrainCurves_freqabc.txt", "O5bStrain"),
                "L1": ("O5StrainCurves_freqabc.txt", "O5bStrain"),
                "V1": ("25115_O5Tier1LowSensASD.txt", None),
            },
            "O5c-HLV": {
                "H1": ("O5StrainCurves_freqabc.txt", "O5cStrain"),
                "L1": ("O5StrainCurves_freqabc.txt", "O5cStrain"),
                "V1": ("25115_O5Tier1HighSensASD.txt", None),
            },
        }

    def load_data(self, run_name, ifo):
        """Load frequency and PSD arrays for a given run and interferometer.

        Parameters
        ----------
        run_name : str
            Run identifier (e.g., ``"O5b-HLV"``).
        ifo : str
            Interferometer name (e.g., ``"H1"``).

        Returns
        -------
        freq : numpy.ndarray
            Frequency array in Hz.
        psd : numpy.ndarray
            Power spectral density in strain^2/Hz.

        Raises
        ------
        KeyError
            If ``run_name`` or ``ifo`` is not found in ``sensitivity_files``.
        ValueError
            If ``data_type`` is neither ``"ASD"`` nor ``"PSD"``.
        """
        filename, column = self.sensitivity_files[run_name][ifo]
        filepath = self.path / filename

        if column is None:
            freq, asd = np.loadtxt(filepath, unpack=True)
        else:
            data = np.genfromtxt(filepath, names=True)
            freq = data[data.dtype.names[0]]
            asd = data[column]

        if self.data_type.lower() == "asd":
            psd = asd**2
        elif self.data_type.lower() == "psd":
            psd = asd
        else:
            raise ValueError("Invalid data type. Please specify 'PSD' or 'ASD'.")

        return freq, psd

    def calculate_bns_range(self, snr=8, mass=1.4):
        """Print the BNS inspiral range for each run and interferometer.

        Parameters
        ----------
        snr : int or float, optional
            Signal-to-noise ratio threshold. Defaults to 8.
        mass : float, optional
            Neutron star mass in solar masses. Defaults to 1.4.
        """
        for run_name in self.run_names:
            print("\n======================================\n")
            print(f"The BNS range in Run {run_name},")
            psd_label = "Measured" if run_name.startswith("O4") else "Ideal"
            print(f"with the {psd_label} PSD\n")

            for ifo in self.ifos:
                if ifo not in self.sensitivity_files.get(run_name, {}):
                    continue
                freq, psd = self.load_data(run_name, ifo)
                fs = FrequencySeries(psd, f0=freq[0], df=freq[1] - freq[0])
                bns_range = inspiral_range(
                    fs, snr=snr, fmin=10, mass1=mass, mass2=mass
                ).value
                print(f"{ifo} BNS Inspiral Range: {round(bns_range, 0)} Mpc")

        print("\n======================================\n")

    def plot_sensitivity_curves(self, run_names, outdir=None):
        """Plot ASD sensitivity curves for O5 runs on a single figure.

        One color per run, one linestyle per detector type:
        LIGO (L1, H1) = ``-``, Virgo (V1) = ``:``.

        Parameters
        ----------
        run_names : list[str]
            O5 run names (e.g., ``["O5a-HLV", "O5b-HLV", "O5c-HLV"]``).
        outdir : str or pathlib.Path
            Output directory. Figure saved as ``Strain_O5_HLV.png``.
        """

        colors = ["#0072B2", "#009E73", "#CC0000"]
        ifo_linestyles = {"H1": "-", "L1": "-", "V1": ":"}
        run_labels = [r.split("-")[0] for r in run_names]

        fig, ax = plt.subplots(figsize=(7, 4.5))

        # Fix 1: avoid LaTeX syntax when LaTeX is not available
        ax.set_title("O5 Sensitivity Curves", fontsize=14, fontweight="bold")

        # --- Plot curves first ---
        for irun, run_name in enumerate(run_names):
            color = colors[irun]
            for ifo in self.ifos:
                if ifo not in self.sensitivity_files.get(run_name, {}):
                    continue
                if ifo == "L1":  # already represented by H1
                    continue
                freq, psd = self.load_data(run_name, ifo)
                asd = np.sqrt(psd)
                ax.loglog(
                    freq,
                    asd,
                    color=color,
                    linestyle=ifo_linestyles[ifo],
                    linewidth=1.5,
                    label="_nolegend_",
                )

        # --- Legend: linestyle = detector type, color = run ---
        linestyle_handles = [
            plt.Line2D([], [], color="black", linestyle="-", linewidth=1.5),
            plt.Line2D([], [], color="black", linestyle=":", linewidth=1.5),
        ]
        color_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=color) for color in colors
        ]
        ax.legend(
            linestyle_handles + color_handles,
            ["LIGO (L1-H1)", "Virgo (V1)"] + run_labels,
            loc="upper right",
            fontsize=10,
        )

        ax.set_xlabel(
            r"$\mathrm{Frequency}\,\mathrm{[Hz]}$", fontsize=14, fontweight="bold"
        )
        ax.set_ylabel(
            r"$\mathrm{ASD}\,[1/\sqrt{\mathrm{Hz}}]$", fontsize=14, fontweight="bold"
        )
        ax.tick_params(labelsize=10)
        ax.set_xlim([8, 4000])
        ax.set_ylim(1e-24, 1e-19)
        ax.grid(True)

        plt.tight_layout(pad=0.5)

        if outdir is not None:
            outdir = Path(outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            for ext in ("pdf", "png", "svg"):
                out_path = outdir / f"Strain_O5_HLV.{ext}"
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                print(f"[INFO] Saved: {out_path.resolve()}")
        else:
            plt.show()
        plt.close()


# # ---------------------------------------------------------------------------
# # Entry point
# # ---------------------------------------------------------------------------
# def main():
#     outdir = Path("outdir")
#     outdir.mkdir(exist_ok=True)

#     print("Hi ============")
#     path = Path("./")
#     ifos = ["L1", "H1", "V1"]
#     run_names = ["O4-HL", "O4-HLV", "O5a-HL", "O5a-HLV", "O5b-HLV", "O5c-HLV"]

#     calculator = BNSInspiralRangeCalculator(
#         run_names=run_names,
#         ifos=ifos,
#         path=path,
#         data_type="ASD",
#     )
#     calculator.calculate_bns_range()
#     calculator.plot_sensitivity_curves(
#         run_names=["O5a-HLV", "O5b-HLV", "O5c-HLV"],
#         outdir=outdir,
#     )


# if __name__ == "__main__":
#     main()

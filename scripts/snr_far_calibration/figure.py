"""Paper figure: the SNR threshold equivalent to FAR < 1/yr.

    python measure.py
    python sensitivity.py
    python figure.py          # writes outputs/snr_threshold.pdf

(a) How rho_eq is set, for BNS in O4.
(b) Fraction of injections found against simulated SNR, O3 and O4.
(c) rho_eq in every chirp-mass bin and run.
(d) Ratio of rho_eq between O4 and O3 for all searches, each search and each
    network, against the ratio of the network BNS ranges.
"""

import matplotlib.pyplot as plt
import numpy as np
from config import CURRENT_THRESHOLD, OUT_DIR, REFERENCE_BINS, TABLE_BINS, read_csv
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

ERAS = {"O3": ("O3a", "O3b"), "O4": ("O4a", "O4b")}
COLOR = {"BNS": "#d95f02", "NSBH": "#7570b3", "BBH": "#1b9e77"}
MARKER = {"BNS": "^", "NSBH": "s", "BBH": "o"}
GREY, DARK = "#9a9a9a", "#222222"
SNR_TICKS = (5, 6, 8, 10, 15, 20, 25)
SHIFT_ROWS = (
    ("All searches", "all", None),
    ("GstLAL", "search", "GstLAL"),
    ("MBTA", "search", "MBTA"),
    ("PyCBC", "search", "PyCBC"),
    ("HL network", "network", "H1L1"),
    ("HLV network", "network", "H1L1V1"),
)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
        "savefig.dpi": 300,
    }
)


def read(name):
    rows = read_csv(OUT_DIR / name)
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except ValueError:
                pass
    return rows


def pick(rows, **match):
    return [
        r
        for r in rows
        if all(
            np.isclose(r[k], v) if isinstance(v, float) else r[k] == v
            for k, v in match.items()
        )
    ]


def rho(results, run, cls, lo):
    found = pick(results, run=run, source_class=cls, chirp_mass_min=lo)
    return found[0]["rho_eq"] if found else np.nan


def snr_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(SNR_TICKS[0], SNR_TICKS[-1])
    ax.xaxis.set_major_locator(FixedLocator(SNR_TICKS))
    ax.xaxis.set_major_formatter(plt.FixedFormatter([str(t) for t in SNR_TICKS]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r"Simulated network SNR, $\rho_\mathrm{net}$")


def pooled_histogram(histograms, cls, lo, runs):
    rows = [
        r
        for r in histograms
        if r["run"] in runs
        and r["source_class"] == cls
        and np.isclose(r["chirp_mass_min"], lo)
    ]
    edges = np.unique([r["snr_min"] for r in rows] + [r["snr_max"] for r in rows])
    found, missed = np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)
    for r in rows:
        k = np.searchsorted(edges, r["snr_min"])
        found[k] += r["found"]
        missed[k] += r["missed"]
    return edges, found, missed


def panel_principle(ax, results, histograms):
    cls, lo, hi = REFERENCE_BINS[0]
    edges, found, missed = pooled_histogram(histograms, cls, lo, ERAS["O4"])
    threshold = np.nanmean([rho(results, run, cls, lo) for run in ERAS["O4"]])
    share = np.mean(
        [
            r["found_below"]
            for run in ERAS["O4"]
            for r in pick(results, run=run, source_class=cls, chirp_mass_min=lo)
        ]
    )
    left, width = edges[:-1], np.diff(edges)
    scale = 1.0 / found.sum()
    below = np.clip((threshold - left) / width, 0, 1)

    ax.bar(left, found * scale, width, align="edge", color=COLOR[cls], lw=0)
    ax.bar(
        left,
        missed * scale,
        width,
        bottom=found * scale,
        align="edge",
        color="#d4d4d4",
        lw=0,
    )
    ax.bar(
        left,
        found * scale,
        width * below,
        align="edge",
        fill=False,
        hatch="////",
        edgecolor="white",
        lw=0,
    )
    ax.bar(
        left + width * below,
        missed * scale,
        width * (1 - below),
        bottom=found * scale,
        align="edge",
        fill=False,
        hatch="\\\\\\\\",
        edgecolor=DARK,
        lw=0,
    )
    ax.axvline(threshold, color=DARK, lw=1.0)
    ax.axvline(CURRENT_THRESHOLD, color=DARK, lw=0.8, ls=":")

    snr_axis(ax)
    top = ax.secondary_xaxis("top")
    top.xaxis.set_major_locator(FixedLocator([CURRENT_THRESHOLD, threshold]))
    top.xaxis.set_major_formatter(
        plt.FixedFormatter(
            [f"SNR = {CURRENT_THRESHOLD:g}", rf"$\rho_\mathrm{{eq}} = {threshold:.1f}$"]
        )
    )
    top.xaxis.set_minor_locator(NullLocator())
    top.tick_params(direction="in", length=3, pad=2)
    ax.tick_params(axis="x", which="both", top=False)
    ax.set_ylim(0, 1.75 * ((found + missed) * scale).max())
    ax.set_ylabel("Injections per bin / number found")
    ax.legend(
        handles=[
            Patch(color=COLOR[cls], label="Found, FAR < 1 yr$^{-1}$"),
            Patch(color="#d4d4d4", label="Missed"),
            Patch(
                facecolor=COLOR[cls],
                hatch="////",
                edgecolor="white",
                label=rf"Found below $\rho_\mathrm{{eq}}$ ({share:.0%})",
            ),
            Patch(
                facecolor="#d4d4d4",
                hatch="\\\\\\\\",
                edgecolor=DARK,
                label=rf"Missed above $\rho_\mathrm{{eq}}$ ({share:.0%})",
            ),
        ],
        loc="upper right",
        handlelength=1.4,
    )


def panel_efficiency(ax, results, histograms):
    for cls, lo, _ in REFERENCE_BINS:
        for era, runs in ERAS.items():
            edges, found, missed = pooled_histogram(histograms, cls, lo, runs)
            total = found + missed
            ok = total > 5
            centre = np.sqrt(edges[:-1] * edges[1:])[ok]
            fraction = found[ok] / total[ok]
            filled = era == "O4"
            ax.plot(
                centre,
                fraction,
                color=COLOR[cls],
                lw=1.2 if filled else 1.0,
                ls="-" if filled else "--",
            )
            threshold = np.mean([rho(results, run, cls, lo) for run in runs])
            ax.plot(
                threshold,
                np.interp(threshold, centre, fraction),
                MARKER[cls],
                ms=5,
                color=COLOR[cls],
                mfc=COLOR[cls] if filled else "white",
                mew=1.0,
                zorder=3,
            )
    ax.axvline(CURRENT_THRESHOLD, color=DARK, lw=0.8, ls=":")
    snr_axis(ax)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Fraction found at FAR < 1 yr$^{-1}$")


def panel_thresholds(ax, results):
    run_offset = {"O3a": -0.027, "O3b": -0.009, "O4a": 0.009, "O4b": 0.027}
    class_offset = {"BNS": -0.055, "NSBH": 0.055, "BBH": 0.0}
    values = []
    for cls, lo, hi in TABLE_BINS:
        centre = np.sqrt(lo * hi) * 10 ** class_offset[cls]
        for run, dx in run_offset.items():
            row = pick(results, run=run, source_class=cls, chirp_mass_min=lo)
            if not row:
                continue
            filled = run.startswith("O4")
            ax.errorbar(
                centre * 10**dx,
                row[0]["rho_eq"],
                row[0]["error"],
                fmt=MARKER[cls],
                ms=3.6,
                color=COLOR[cls],
                mfc=COLOR[cls] if filled else "white",
                mew=0.8,
                elinewidth=0.7,
                capsize=1.3,
                capthick=0.7,
                zorder=3,
            )
            values.append(row[0]["rho_eq"])
    ax.axhline(CURRENT_THRESHOLD, color=DARK, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(0.75, 340)
    ax.xaxis.set_major_locator(FixedLocator((1, 2, 5, 10, 20, 50, 100, 200)))
    ax.xaxis.set_major_formatter(
        plt.FixedFormatter(["1", "2", "5", "10", "20", "50", "100", "200"])
    )
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(min(values + [CURRENT_THRESHOLD]) - 0.5, max(values) + 0.6)
    ax.set_xlabel(
        r"Detector-frame chirp mass, $\mathcal{M}_\mathrm{c}^\mathrm{det}$ [$M_\odot$]"
    )
    ax.set_ylabel(r"$\rho_\mathrm{eq}$ for FAR < 1 yr$^{-1}$")


def panel_shift(ax, results, variants, ranges):
    def ratio(cls, lo, group, name):
        def era_mean(runs):
            if group == "all":
                vals = [rho(results, run, cls, lo) for run in runs]
            else:
                vals = [
                    r["rho_eq"]
                    for run in runs
                    for r in pick(
                        variants,
                        run=run,
                        source_class=cls,
                        chirp_mass_min=lo,
                        group=group,
                        variant=name,
                    )
                ]
            return np.nanmean(vals) if vals else np.nan

        return era_mean(ERAS["O4"]) / era_mean(ERAS["O3"])

    def network(run):
        return np.hypot(
            *[pick(ranges, run=run, ifo=ifo)[0]["range_mean"] for ifo in ("H1", "L1")]
        )

    span = [network("O4a") / network("O3a"), network("O4b") / network("O3a")]
    ax.axvspan(min(span), max(span), color=GREY, alpha=0.25, lw=0)
    ax.axvline(1.0, color=DARK, lw=0.6)

    y = np.arange(len(SHIFT_ROWS))[::-1]
    for (cls, lo, _), dy in zip(REFERENCE_BINS, (0.22, 0.0, -0.22)):
        x = [ratio(cls, lo, group, name) for _, group, name in SHIFT_ROWS]
        ax.plot(x, y + dy, MARKER[cls], ms=4.5, color=COLOR[cls], ls="none")
    for boundary in (y[0] - 0.5, y[3] - 0.5):
        ax.axhline(boundary, color=GREY, lw=0.5)
    ax.set_yticks(y, [label for label, _, _ in SHIFT_ROWS])
    ax.tick_params(axis="y", which="both", length=0)
    ax.set_ylim(y[-1] - 0.6, y[0] + 0.6)
    ax.set_xlim(0.95, max(span) + 0.12)
    ax.set_xlabel(r"$\rho_\mathrm{eq}(\mathrm{O4})\,/\,\rho_\mathrm{eq}(\mathrm{O3})$")


def legend(fig):
    handles = [
        Line2D([], [], color=COLOR[c], marker=MARKER[c], ls="none", ms=5, label=c)
        for c in ("BNS", "NSBH", "BBH")
    ]
    handles += [
        Line2D(
            [],
            [],
            color=DARK,
            ls="--",
            marker="o",
            mfc="white",
            ms=4,
            lw=1.0,
            label="O3a, O3b",
        ),
        Line2D([], [], color=DARK, ls="-", marker="o", ms=4, lw=1.2, label="O4a, O4b"),
        Line2D([], [], color=DARK, lw=0.8, ls=":", label="SNR = 8"),
        Patch(color=GREY, alpha=0.25, lw=0, label="BNS range, O4 / O3"),
    ]
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=7,
        handlelength=1.6,
        columnspacing=1.4,
        handletextpad=0.4,
    )


def main():
    results = read("rho_eq.csv")
    variants = read("variants.csv")
    histograms = read("histograms.csv")
    ranges = read("release_ranges.csv")

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.6), layout="constrained")
    (a, b), (c, d) = axes
    panel_principle(a, results, histograms)
    panel_efficiency(b, results, histograms)
    panel_thresholds(c, results)
    panel_shift(d, results, variants, ranges)
    for ax, label in zip(axes.flat, "abcd"):
        ax.set_title(f"({label})", loc="left", fontsize=9, fontweight="bold", pad=3)
    legend(fig)

    for suffix in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"snr_threshold.{suffix}")
    print(f"wrote {OUT_DIR / 'snr_threshold.pdf'}")


if __name__ == "__main__":
    main()

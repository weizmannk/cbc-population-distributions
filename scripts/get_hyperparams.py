import numpy as np
import pandas as pd
from bilby.core.result import read_in_result

# # in ".rst" format
# def to_rst(df, title="Hyperparameters of the BP2P model"):
#     lines = []
#     lines.append(f".. table:: {title}")
#     lines.append("   :widths: 20 60 20")
#     lines.append("   :header-rows: 1\n")
#     lines.append("   * - Parameter")
#     lines.append("     - Description")
#     lines.append("     - Value")
#     for _, row in df.iterrows():
#         lines.append(f"   * - :math:`{row['Parameter']}`")
#         lines.append(f"     - {row['Description']}")
#         lines.append(f"     - {row['Value']}")
#     return "\n".join(lines)


# def to_rst(df, title="Hyperparameters of the BP2P model"):
#     """
#     Render dataframe as a reStructuredText grid table with separators.
#     """
#     widths = [max(len(str(x)) for x in df[col]) for col in df.columns]
#     widths = [max(w, len(col)) for w, col in zip(widths, df.columns)]

#     def hline(sep="-"):
#         return "+" + "+".join(sep * (w + 2) for w in widths) + "+"

#     def row(cells):
#         return "|" + "|".join(f" {str(c).ljust(w)} " for c, w in zip(cells, widths)) + "|"


#     lines = [f".. table:: {title}", ""]
#     lines.append(hline("="))
#     lines.append(row(df.columns))
#     lines.append(hline("="))
#     for _, r in df.iterrows():
#         lines.append(row(r))
#         lines.append(hline())
#     return "\n".join(lines)
def to_rst(df, title="Hyperparameters of the BP2P model"):
    widths = [max(len(str(x)) for x in df[col]) for col in df.columns]
    widths = [max(w, len(col)) for w, col in zip(widths, df.columns)]

    def hline(sep="-"):
        return "+" + "+".join(sep * (w + 2) for w in widths) + "+"

    def row(cells):
        return (
            "|" + "|".join(f" {str(c).ljust(w)} " for c, w in zip(cells, widths)) + "|"
        )

    # build the grid table body (no directive yet)
    body = [hline("-"), row(df.columns), hline("=")]
    for _, r in df.iterrows():
        body.append(row(r))
        body.append(hline())

    # indent body so it belongs to the table directive
    indented = ["   " + line for line in body]  # 3 spaces is conventional

    lines = [f".. table:: {title}", ""]
    lines.extend(indented)
    return "\n".join(lines)


# --- in  LaTeX ---
def to_latex(df, caption="Hyperparameters of the BP2P model"):
    out = []
    out.append(r"\begin{table}[ht]")
    out.append(r"\centering")
    out.append(rf"\caption{{{caption}}}")
    out.append(r"\begin{tabular}{lll}")
    out.append(r"\hline")
    out.append(r"Parameter & Description & Value\\")
    out.append(r"\hline")
    for _, row in df.iterrows():
        out.append(rf"{row['Parameter']} & {row['Description']} & {row['Value']} \\")
    out.append(r"\hline")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    return "\n".join(out)


def _get_map_sample(hyperparams) -> pd.Series:
    """
    Return MAP sample if prior is informative, else ML sample.
    """
    post = hyperparams.posterior.copy()
    if "log_prior" in post and post["log_prior"].nunique() > 1:
        score = post.log_likelihood + post.log_prior
        return post.iloc[np.argmax(score)]
    else:
        return post.iloc[np.argmax(post.log_likelihood)]


PARAMS_INFO = {
    "alpha_1": (r"\alpha_1", "Power-law exponent below $\\gamma_{low1}$"),
    "alpha_2": (r"\alpha_2", "Power-law exponent above $\\gamma_{high1}$"),
    "alpha_dip": (r"\alpha_d", "Power-law exponent inside the NS–BH mass gap"),
    "NSmin": (r"m_{\\min}", "Minimum compact object mass"),
    "NSmax": (r"\\gamma_{low1}", "Start of the lower mass gap"),
    "BHmin": (r"\\gamma_{high1}", "End of the lower mass gap"),
    "BHmax": (r"m_{\\max}", "Maximum BH mass in the power-law component"),
    "A": (r"A", "Depth of the dip between $\\gamma_{low1}$ and $\\gamma_{high1}$"),
    "UPPERmin": (r"\\gamma_{low2}", "Start of the upper (PI) mass gap"),
    "UPPERmax": (r"\\gamma_{high2}", "End of the upper (PI) mass gap"),
    "mu1": (r"\\mu_{peak1}", "Mean of the upper Gaussian peak"),
    "sig1": (r"\\sigma_{peak1}", "Width of the upper Gaussian peak"),
    "mix1": (r"c_1", "Mixing fraction of the upper Gaussian peak"),
    "mu2": (
        r"\\mu_{peak2}",
        "Mean of the lower Gaussian peak where an overdensity of merging compact objects is observed",
    ),
    "sig2": (
        r"\\sigma_{peak2}",
        "Width of the lower Gaussian peak where an overdensity of merging compact objects is observe",
    ),
    "mix2": (
        r"c_2",
        "Mixing fraction of the lower Gaussian peak with the powerlaw + notches",
    ),
    "absolute_mmin": (r"m_{abs,min}", "Absolute minimum truncation"),
    "absolute_mmax": (r"m_{abs,max}", "Absolute maximum truncation"),
    "n0": (r"\\eta_0", "Exponents to set the sharpness of the low mass cutoff"),
    "n5": (r"\\eta_5", "Exponents to set the sharpness of the high mass cutoff"),
    "n1": (
        r"\\eta_1",
        "Exponents to set the sharpness of the lower edge of the lower mass gap( $\\gamma_{low1})$",
    ),
    "n2": (
        r"\\eta_2",
        "Exponents to set the sharpness of upper edge of the lower mass gap ($\\gamma_{high1}$)",
    ),
    "n3": (
        r"\\eta_3",
        "Exponents to set the sharpness of the lower edge of the upper mass gap ($\\gamma_{low2}$)",
    ),
    "n4": (
        r"\\eta_4",
        " Exponents to set the sharpness of the upper edge of the upper mass gap, $\\gamma_{high2}$",
    ),
}


hyperparams_file = "../data/baseline5_widesigmachi2_mass_NotchFilterBinnedPairingMassDistribution_redshift_powerlaw_mag_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_result.hdf5"

# Load "Broken Power Law + 2 Peaks model" and extract MAP sample
hyperparams = read_in_result(hyperparams_file)
maxp_samp = _get_map_sample(hyperparams)


rows = []
for key, (param, desc) in PARAMS_INFO.items():
    if key in maxp_samp:
        val = maxp_samp[key]
        rows.append((f"${param}$", desc, f"{val:.3g}"))

df = pd.DataFrame(rows, columns=["Parameter", "Description", "Value"])

# Sauvegarde
with open("hyperparams_table.rst", "w") as f:
    f.write(to_rst(df))

with open("hyperparams_table.tex", "w") as f:
    f.write(to_latex(df))

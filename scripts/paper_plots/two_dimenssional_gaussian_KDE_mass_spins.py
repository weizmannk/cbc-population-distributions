import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.patheffects import withStroke
from scipy.stats import gaussian_kde

# ============================================================================
# Configuration
# ============================================================================
N_SAMPLES = 10_000
FIGURE_SIZE = (14, 6)
FONT_SIZE_LABEL = 16
FONT_SIZE_TICK = 14
FONT_SIZE_CBAR = 14
SCATTER_SIZE = 3
ALPHA = 0.7
COLORMAP = "plasma"  #'magma'

# ============================================================================
# Load and process data
# ============================================================================
fullpop4 = Table.read("fullpop4.h5")
ns_max_mass = 3.0


fullpop4.sort("mass1")

# ============================================================================
# Create combined visualization
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)

# ============================================================================
# LEFT PLOT: Mass Distribution
# ============================================================================
ax1 = axes[0]

# Mass distribution (log scale)
mass1 = fullpop4["mass1"]
mass2 = fullpop4["mass2"]

# Compute KDE
xy_mass = np.vstack([mass1, mass2])
kde_mass = gaussian_kde(xy_mass, bw_method="scott")
z_mass = kde_mass(xy_mass)

# Sort by density
idx_mass = z_mass.argsort()
mass1_sorted = mass1[idx_mass]
mass2_sorted = mass2[idx_mass]
z_mass_sorted = z_mass[idx_mass]

# Create scatter plot
scatter1 = ax1.scatter(
    mass1_sorted,
    mass2_sorted,
    c=z_mass_sorted,
    s=SCATTER_SIZE,
    cmap=COLORMAP,
    alpha=ALPHA,
    edgecolors="none",
    rasterized=True,
)

# Add separation lines for BNS/NSBH/BBH
log_ns_max = ns_max_mass  # np.log10(ns_max_mass)

# Horizontal line at ns_max_mass (separates NS from BH on mass2 axis)
ax1.axhline(
    y=log_ns_max,
    color="gray",
    linestyle="-",
    linewidth=1.2,
    label=f"NS-BH boundary ({ns_max_mass:.0f} $M_\\odot$)",
    alpha=0.7,
)

# Vertical line at ns_max_mass (separates NS from BH on mass1 axis)
ax1.axvline(x=log_ns_max, color="gray", linestyle="-", linewidth=1.2, alpha=0.7)

# Add text labels for regions with white text and black outline for better visibility
text_style = [withStroke(linewidth=3, foreground="black")]

ax1.text(
    0.08,
    0.32,
    "BNS",
    transform=ax1.transAxes,
    fontsize=16,
    fontweight="bold",
    color="white",
    path_effects=text_style,
)

ax1.text(
    0.5,
    0.015,
    "NSBH",
    transform=ax1.transAxes,
    fontsize=16,
    fontweight="bold",
    color="white",
    path_effects=text_style,
)

ax1.text(
    0.5,
    0.7,
    "BBH",
    transform=ax1.transAxes,
    fontsize=16,
    fontweight="bold",
    color="white",
    path_effects=text_style,
)

# Customize axes
ax1.set_xlabel(r"$ m_1 \ [M_\odot]$", fontsize=FONT_SIZE_LABEL, fontweight="bold")
ax1.set_ylabel(r"$ m_2 \ [M_\odot]$", fontsize=FONT_SIZE_LABEL, fontweight="bold")
ax1.tick_params(axis="both", which="major", labelsize=FONT_SIZE_TICK)
ax1.set_title(
    r"Mass Distribution", color="navy", fontsize=FONT_SIZE_LABEL, fontweight="bold"
)

# Add grid
ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

# Add legend with better formatting
ax1.legend(loc="upper left", fontsize=11, framealpha=0.9, edgecolor="gray")

# Colorbar
cbar1 = fig.colorbar(scatter1, ax=ax1, pad=0.02)
cbar1.set_label("Probability Density", fontsize=FONT_SIZE_CBAR, fontweight="bold")
cbar1.ax.tick_params(labelsize=FONT_SIZE_TICK)

ax1.set_xscale("log")
ax1.set_yscale("log")
# ============================================================================
# RIGHT PLOT: Spin Distribution
# ============================================================================
ax2 = axes[1]

# Extract spin data
spin1z = fullpop4["spin1z"]
spin2z = fullpop4["spin2z"]

# Compute KDE
xy_spin = np.vstack([spin1z, spin2z])
kde_spin = gaussian_kde(xy_spin, bw_method="scott")
z_spin = kde_spin(xy_spin)

# Sort by density
idx_spin = z_spin.argsort()
spin1z_sorted = spin1z[idx_spin]
spin2z_sorted = spin2z[idx_spin]
z_spin_sorted = z_spin[idx_spin]

# Scatter plot
scatter2 = ax2.scatter(
    spin1z_sorted,
    spin2z_sorted,
    c=z_spin_sorted,
    s=SCATTER_SIZE,
    cmap=COLORMAP,
    alpha=ALPHA,
    edgecolors="none",
    rasterized=True,
)

ax2.set_xlabel(r"$\chi_{z_1}$", fontsize=FONT_SIZE_LABEL + 2, fontweight="bold")
ax2.set_ylabel(r"$\chi_{z_2}$", fontsize=FONT_SIZE_LABEL + 2, fontweight="bold")
ax2.tick_params(axis="both", which="major", labelsize=FONT_SIZE_TICK)
ax2.set_title(
    r"Spin Distribution", color="navy", fontsize=FONT_SIZE_LABEL, fontweight="bold"
)

# Add grid
ax2.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

# Add reference lines at zero (slightly darker for print visibility)
ax2.axhline(y=0, color="gray", linestyle="-", linewidth=1.2, alpha=0.6)
ax2.axvline(x=0, color="gray", linestyle="-", linewidth=1.2, alpha=0.6)

# Colorbar
cbar2 = fig.colorbar(scatter2, ax=ax2, pad=0.02)
cbar2.set_label("Probability Density", fontsize=FONT_SIZE_CBAR, fontweight="bold")
cbar1.ax.tick_params(labelsize=FONT_SIZE_TICK)

# ============================================================================
# Save
# ============================================================================
plt.tight_layout()

plt.savefig(
    "gaussian_kde_distribution_of_masses_spins_logscale.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)
plt.savefig(
    "gaussian_kde_distribution_of_masses_spins_logscale.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)
print("Figures saved: gaussian_kde_distribution_of_masses_spins.pdf and .png")

# ============================================================================
# Print population statistics
# ============================================================================
print(
    f"Number of BNS: {len(fullpop4[(fullpop4['mass1'] < ns_max_mass)])}, "
    f"Number of NSBH: {len(fullpop4[(fullpop4['mass1'] >= ns_max_mass) & (fullpop4['mass2'] < ns_max_mass)])}, "
    f"Number of BBH: {len(fullpop4[(fullpop4['mass2'] >= ns_max_mass)])}"
)


plt.close()

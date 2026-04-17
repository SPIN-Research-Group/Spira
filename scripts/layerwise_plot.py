import os
import re
import matplotlib.pyplot as plt
from matplotlib import colormaps
from collections import defaultdict
import numpy as np
from matplotlib.ticker import MultipleLocator


# =========================================
# Paths
# =========================================


results_folder = os.path.join("results", "layerwise")
output_file = os.path.join("figures", "layerwise_performance.pdf")
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# =========================================
# Configuration
# =========================================
libraries = ["minuet", "torchsparse", "spira"]
baseline = "minuet"

custom_legend_names = {
    "minuet": "Minuet",
    "torchsparse": "TorchSparse++",
    "spira": "Spira"
}

colormap = plt.colormaps.get_cmap('bone')
colors = {
    "spira": colormap(80),
    "torchsparse": "#f39c12",
    "minuet": "#6B7C0E",
}


# =========================================
# --- Parse Results ---
# =========================================

results = defaultdict(lambda: defaultdict(list))
# results[(k, cin, cout)][library] = [times from all datasets]

for fname in os.listdir(results_folder):
    if not fname.endswith(".out"):
        continue

    match = re.match(r'([^_]+)_cin(\d+)_cout(\d+)_k(\d+)_([^.]+)\.out', fname)
    if not match:
        continue

    dataset = match.group(1).lower()
    cin = match.group(2)
    cout = match.group(3)
    k = match.group(4)
    library = match.group(5).lower()

    config_key = (cin, cout, k)  # remove dataset from key

    with open(os.path.join(results_folder, fname)) as f:
        content = f.read()

    avg_match = re.search(r'overall_avg:\s*([\d.]+)', content)
    if avg_match:
        avg_time = float(avg_match.group(1))
        results[config_key][library].append(avg_time)

# --- Geometric mean across datasets for each config ---
agg_times = {}
for config, libs in results.items():
    agg_times[config] = {}
    for lib, times in libs.items():
        times = [t for t in times if t > 0]
        if times:
            agg_times[config][lib] = np.exp(np.mean(np.log(times)))


# =========================================
# --- Compute Speedups ---
# =========================================
speedups = {}
for config, libs in agg_times.items():
    if "minuet" not in libs:
        continue
    base_time = libs["minuet"]
    speedups[config] = {lib: base_time / t for lib, t in libs.items() if t > 0}


ordered_keys = sorted(speedups.keys(), key=lambda x: (int(x[2]), int(x[0]), int(x[1])))

# Overall geomean across all configs
geo_vals = {}
for lib in libraries:
    vals = [speedups[k].get(lib) for k in ordered_keys if speedups[k].get(lib) is not None]
    if vals:
        geo_vals[lib] = np.exp(np.mean(np.log(vals)))
speedups[("geomean", "all", "all")] = geo_vals
ordered_keys.append(("geomean", "all", "all"))

# =========================================
# --- Plotting ---
# =========================================
labels = []

plt.rcParams.update({
    'axes.labelsize': 21,
    'xtick.labelsize': 14,
    'ytick.labelsize': 20,
    'legend.fontsize': 21,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold'
})

plt.figure(figsize=(15, 5))

ymax = 2
plt.ylim(0, ymax)

bar_width = 0.04
intra_space = 0
inter_space = 0.02

group_width = len(libraries) * bar_width + (len(libraries) - 1) * intra_space
x_centers = np.arange(len(ordered_keys)) * (group_width + inter_space)


# --- Plot bars ---
for i, lib in enumerate(libraries):
    offset = -group_width/2 + i * (bar_width + intra_space) + bar_width/2

    vals = [speedups[key].get(lib, 0) for key in ordered_keys]
    clipped_vals = [min(v, ymax) if v else 0 for v in vals]

    bars = plt.bar(
        x_centers + offset,
        clipped_vals,
        bar_width,
        label=custom_legend_names.get(lib, lib),
        color=colors.get(lib, "gray"),
        edgecolor="black"
    )

    for bar, val, clipped in zip(bars, vals, clipped_vals):
        if val == 0:
            continue

        plt.text(
            bar.get_x() + bar.get_width()/2,
            clipped + 0.01,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            weight='bold'
        )


# --- X-axis labels ---
for key in ordered_keys:
    if key[0] == "geomean":
        labels.append("GeoMean")
    else:
        cin, cout, k = key
        labels.append(f"({cin},{cout},{k})")

plt.xticks(x_centers, labels)
plt.ylabel("Speedup")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.5))
plt.gca().set_axisbelow(True)

plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.25),
    ncol=len(libraries)
)

plt.margins(x=0.014)

first_geo_index = next(
    (i for i, key in enumerate(ordered_keys) if key[0] == "geomean"),
    None
)

if first_geo_index is not None:
    geo_x = (x_centers[first_geo_index - 1] + x_centers[first_geo_index]) / 2
    plt.axvline(x=geo_x, color='black', linestyle='--', linewidth=1.5)


plt.tight_layout()
plt.subplots_adjust(left=0.055, right=0.99, top=0.83, bottom=0.12)
plt.savefig(output_file, bbox_inches='tight')

print(f"Plot saved to: {output_file}")

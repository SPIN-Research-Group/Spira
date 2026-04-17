import os
import re
import matplotlib.pyplot as plt
from matplotlib import colormaps
from collections import defaultdict
import numpy as np
import math
from matplotlib.ticker import MultipleLocator

results_folder = os.path.join("results", "end_to_end")
output_file = os.path.join("figures", "end_to_end_performance.pdf")
os.makedirs(os.path.dirname(output_file), exist_ok=True)

libraries = ["minuet", "torchsparse", "spira", "spira_64"]
baseline = "minuet"

# Display names
custom_legend_names = {
    "minuet": "Minuet",
    "torchsparse": "TorchSparse++",
    "spira": "Spira (32p)",
    "spira_64": "Spira (64p)"
}

# Colors
colormap = plt.colormaps.get_cmap('bone')
colors = {
    "spira": colormap(80),
    "spira_64": colormap(40),
    "torchsparse": "#f39c12",
    "minuet": "#6B7C0E",
}

dataset_labels = {
    "kitti": "Kitti",
    "waymo": "Waymo",
    "scannet": "ScanNet"
}

model_labels = {
    "ResNet": "ResN",
    "UNet": "UNet",
    "ResNetLarge": "ResNL"
}

# --- Parse results --- #
results = defaultdict(dict)

for fname in os.listdir(results_folder):
    if not fname.endswith(".out"):
        continue

    # Expect filename: <dataset>_<model>_<library>.out
    match = re.match(r'([^_]+)_([^_]+)_([^.]+)\.out', fname)
    if not match:
        continue

    dataset = match.group(1).lower()
    model = match.group(2)
    library = match.group(3).lower()

    with open(os.path.join(results_folder, fname)) as f:
        content = f.read()

    # Extract overall average
    avg_match = re.search(r'overall_avg:\s*([\d.]+)', content)
    if avg_match:
        avg_time = float(avg_match.group(1))
        results[(model, dataset)][library] = avg_time


# --- Compute speedups relative to baseline --- #
speedups = defaultdict(dict)

for (model, dataset), libs in results.items():
    if baseline not in libs:
        continue
    base_time = libs[baseline]
    for lib, t in libs.items():
        speedups[(model, dataset)][lib] = base_time / t


# --- Order groups (model first, then dataset) --- #
ordered_keys = sorted(speedups.keys(), key=lambda k: (k[0], k[1]))


# --- Compute ONE global geometric mean --- #
def compute_geomean(keys_subset):
    geo_vals = {}
    for lib in libraries:
        vals = [speedups[key].get(lib, None)
                for key in keys_subset
                if speedups[key].get(lib, None) is not None]
        if len(vals) > 0:
            geo_vals[lib] = np.exp(np.mean(np.log(vals)))
        else:
            geo_vals[lib] = None
    return geo_vals


geo_vals = compute_geomean(ordered_keys)
speedups[("geomean", "all")] = geo_vals
ordered_keys.append(("geomean", "all"))

# --- Plotting --- #
labels = []
bar_width = 0.04
intra_space = 0
inter_space = 0.02


plt.rcParams.update({
    'axes.labelsize': 21,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 21,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold'
})

plt.figure(figsize=(20, 6))
ymax = 2
plt.ylim(0, ymax)


group_width = len(libraries) * bar_width + (len(libraries) - 1) * intra_space
x_centers = np.arange(len(ordered_keys)) * (group_width + inter_space)

# --- Plot bars --- #
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
            fontsize=14,
            weight='bold'
        )


# --- X-axis labels --- #
for (model, dataset) in ordered_keys:
    if model == "geomean":
        labels.append("GeoMean")
    else:
        labels.append(f"{model_labels.get(model, model)}\n{dataset_labels.get(dataset, dataset)}")

plt.xticks(x_centers, labels)
plt.ylabel("Speedup")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.5))
plt.gca().set_axisbelow(True)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.26), ncol=len(libraries))
plt.margins(x=0.014)

# --- Vertical separator before geomean blocks --- #
first_geo_index = next((i for i, (m, _) in enumerate(ordered_keys) if m == "geomean"), None)
if first_geo_index is not None:
    geo_x = (x_centers[first_geo_index - 1] + x_centers[first_geo_index]) / 2
    plt.axvline(x=geo_x, color='black', linestyle='--', linewidth=1.5)

plt.tight_layout()
plt.subplots_adjust(left=0.050, right=0.99, top=0.84, bottom=0.2)
plt.savefig(output_file, bbox_inches='tight')

print(f"Plot saved to: {output_file}")

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.ticker import MultipleLocator

# --- Configuration --- #
results_folder = os.path.join("results", "mapping")
output_file = os.path.join("figures", "mapping_performance.pdf")
os.makedirs(os.path.dirname(output_file), exist_ok=True)

libraries = ["minuet", "torchsparse", "simple_bs", "spira"]

# Mapping library display names
custom_legend_names = {
    "minuet": "Minuet",
    "torchsparse": "TorchSparse++",
    "simple_bs": "Simple BSearch",
    "spira": "Spira"
}

# Color map
colormap = plt.colormaps.get_cmap('bone')
colors = {
    "spira": colormap(80),
    "torchsparse": "#f39c12",
    "minuet": "#6B7C0E",
    "simple_bs": colormap(140),
}

# --- Parse results --- #
results = defaultdict(dict)
voxel_counts = {}

for fname in os.listdir(results_folder):
    if not fname.endswith(".out"):
        continue

    # Expecting filename: <scene_name>_k<kernel>_<library>.out
    match = re.match(r'(.+)_k(\d+)_([a-z_+]+)\.out', fname, re.IGNORECASE)
    if not match:
        continue

    scene_name = match.group(1)
    kernel = int(match.group(2))
    library = match.group(3).lower()

    # Load file content
    with open(os.path.join(results_folder, fname)) as f:
        content = f.read()

    # Extract voxel count
    voxel_match = re.search(r'Loaded[:\s]+(\d+)\s*Voxels', content, re.IGNORECASE)
    if voxel_match:
        voxel_counts[scene_name] = int(voxel_match.group(1))

    # Extract average time
    avg_match = re.search(r'Average[:\s]+([\d.]+)\s*ms', content)
    if avg_match:
        avg_time = float(avg_match.group(1))
        results[(scene_name, kernel)][library] = avg_time

# --- Compute speedups relative to Minuet --- #
speedups = defaultdict(dict)
for (scene_name, kernel), libs in results.items():
    if "minuet" not in libs:
        continue
    minuet_time = libs["minuet"]
    for lib, t in libs.items():
        speedups[(scene_name, kernel)][lib] = minuet_time / t

# --- Order groups --- #
ordered_keys = sorted(speedups.keys(), key=lambda k: (voxel_counts.get(k[0], float('inf')), k[1]))

# --- Compute geometric means per kernel size --- #
def compute_geomean(keys_subset):
    geo_vals = {}
    for lib in libraries:
        vals = [speedups[key].get(lib, None) for key in keys_subset 
                if speedups[key].get(lib, None) is not None]
        if len(vals) > 0:
            geo_vals[lib] = np.exp(np.mean(np.log(vals)))  # geometric mean
        else:
            geo_vals[lib] = None
    return geo_vals

kernel_sizes = sorted(set(k[1] for k in ordered_keys))
for k in kernel_sizes:
    geo_vals = compute_geomean([key for key in ordered_keys if key[1] == k])
    speedups[("geomean", k)] = geo_vals
    ordered_keys.append(("geomean", k))

# --- Plotting --- #
labels = []
bar_width = 0.25
plt.rcParams.update({
    'axes.labelsize': 21,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 18,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold'
})

plt.figure(figsize=(15, 5))
ymax = 2.5
plt.ylim(0, ymax)

group_width = len(libraries) * bar_width
x_centers = np.arange(len(ordered_keys)) * (group_width + 0.3)

# Plot bars
for i, lib in enumerate(libraries):
    offset = -group_width/2 + i * bar_width + bar_width/2
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

    # Value labels
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

# X-axis labels
for (scene_name, kernel) in ordered_keys:
    if scene_name == "geomean":
        labels.append(f"GeoMean\nK={kernel}")
    else:
        voxel_count = voxel_counts.get(scene_name, 0)
        if voxel_count >= 1000:
            voxel_label = f"{int(np.ceil(voxel_count / 1000))}k"
        else:
            voxel_label = str(voxel_count)
        labels.append(f"{voxel_label}\nK={kernel}")

plt.xticks(x_centers, labels, rotation=0)
plt.ylabel("Speedup")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.5))
plt.gca().set_axisbelow(True)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=len(libraries))
plt.margins(x=0.014)

# Vertical line before geomean
first_geo_index = next((i for i, (fname, _) in enumerate(ordered_keys) if fname == "geomean"), None)
if first_geo_index is not None:
    geo_x = x_centers[first_geo_index] - (group_width + 0.3) / 2
    plt.axvline(x=geo_x, color='black', linestyle='--', linewidth=1.5)

plt.tight_layout()
plt.subplots_adjust(left=0.049, right=0.99, top=0.86, bottom=0.2)
plt.savefig(output_file, bbox_inches='tight')
print(f"Plot saved to: {output_file}")
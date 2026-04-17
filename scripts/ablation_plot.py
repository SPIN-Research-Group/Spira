import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from matplotlib.ticker import MultipleLocator


scene_sizes = [
    (10_000,  "1e4"),
    (50_000,  "5e4"),
    (100_000, "1e5"),
    (500_000, "5e5"),
    (1_000_000, "1e6"),
]

def format_power_label(token):
    base, exp = token.split("e")
    if base == "1":
        return rf"$10^{exp}$"
    else:
        return rf"${base} \times 10^{exp}$"

scene_label_map = {
    token: format_power_label(token)
    for _, token in scene_sizes
}

# =========================================
# Paths
# =========================================
results_folder = os.path.join("results", "ablation")
output_file = os.path.join("figures", "ablation_performance.pdf")
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

# results[scene][library] = [avg_times_across_models]
results = defaultdict(lambda: defaultdict(list))

for fname in os.listdir(results_folder):
    if not fname.endswith(".out"):
        continue

    with open(os.path.join(results_folder, fname)) as f:
        content = f.read()

    scene_match = re.search(r'scene:\s*(.+)', content)
    model_match = re.search(r'model:\s*(.+)', content)
    lib_match = re.search(r'library:\s*(.+)', content)
    avg_match = re.search(r'overall_avg_ms:\s*([\d.]+)', content)

    if not (scene_match and lib_match and avg_match):
        continue

    scene = scene_match.group(1).strip()
    library = lib_match.group(1).strip().lower()
    avg_time = float(avg_match.group(1))

    results[scene][library].append(avg_time)


# =========================================
# --- Geometric Mean Across Networks ---
# =========================================

# agg_times[scene][library] = geomean_across_models
agg_times = {}

for scene, libs in results.items():
    agg_times[scene] = {}
    for lib, times in libs.items():
        times = [t for t in times if t > 0]
        if times:
            agg_times[scene][lib] = np.exp(np.mean(np.log(times)))


# =========================================
# --- Compute Speedups ---
# =========================================

speedups = {}
for scene, libs in agg_times.items():
    if baseline not in libs:
        continue

    base_time = libs[baseline]
    speedups[scene] = {
        lib: base_time / t for lib, t in libs.items() if t > 0
    }

scene_order_map = {token: idx for idx, (_, token) in enumerate(scene_sizes)}

def scene_sort_key(scene):
    if scene == "geomean":
        return float('inf')  # put geomean last
    size_match = re.search(r'(\de\d+)', scene)
    if size_match:
        token = size_match.group(1)
        return scene_order_map.get(token, float('inf'))
    return float('inf')

ordered_scenes = sorted(speedups.keys(), key=scene_sort_key)

# --- Overall geomean across scenes ---
geo_vals = {}
for lib in libraries:
    vals = [
        speedups[scene].get(lib)
        for scene in ordered_scenes
        if speedups[scene].get(lib) is not None
    ]
    if vals:
        geo_vals[lib] = np.exp(np.mean(np.log(vals)))

speedups["geomean"] = geo_vals
ordered_scenes.append("geomean")


# =========================================
# --- Plotting ---
# =========================================

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

ymax = 3.2
plt.ylim(0, ymax)

bar_width = 0.04
intra_space = 0
inter_space = 0.02

group_width = len(libraries) * bar_width + (len(libraries) - 1) * intra_space
x_centers = np.arange(len(ordered_scenes)) * (group_width + inter_space)


# --- Plot bars ---
for i, lib in enumerate(libraries):
    offset = -group_width/2 + i * (bar_width + intra_space) + bar_width/2

    vals = [speedups[scene].get(lib, 0) for scene in ordered_scenes]
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
            clipped + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            weight='bold'
        )


# --- X-axis labels ---
TOTAL_VOXELS = 8_000_000

labels = []
for scene in ordered_scenes:
    if scene == "geomean":
        labels.append("GeoMean")
    else:
        size_match = re.search(r'(\de\d+)', scene)
        if size_match:
            token = size_match.group(1)

            # Convert 1e6 -> float
            num_voxels = float(token.replace("e", "e"))
            percentage = (num_voxels / TOTAL_VOXELS) * 100

            labels.append(f"{percentage:.2f}%")
        else:
            labels.append(scene)


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

# Vertical separator before GeoMean
if "geomean" in ordered_scenes:
    geo_index = ordered_scenes.index("geomean")
    if geo_index > 0:
        geo_x = (x_centers[geo_index - 1] + x_centers[geo_index]) / 2
        plt.axvline(x=geo_x, color='black', linestyle='--', linewidth=1.5)


plt.tight_layout()
plt.subplots_adjust(left=0.055, right=0.99, top=0.83, bottom=0.12)
plt.savefig(output_file, bbox_inches='tight')

print(f"Plot saved to: {output_file}")

import numpy as np
import os
import argparse
import random

def main(seed: int = 2026):

    random.seed(seed)
    np.random.seed(seed)

    # Output directory
    out_dir = "random_voxels"
    os.makedirs(out_dir, exist_ok=True)

    # Scene sizes (voxels)
    scene_sizes = [
        (10_000,  "1e4"),    # 0.1% density
        (50_000,  "5e4"),    # 0.6% density
        (100_000, "1e5"),    # 1.25% density
        (500_000, "5e5"),    # 6.25% density
        (1_000_000, "1e6"),  # 12.5% density
    ]

    # Max values for each axis
    x_max, y_max, z_max = 200, 200, 200

    for N, tag in scene_sizes:
        print(f"Generating scene with {N} voxels...")

        points = np.column_stack((
            np.random.randint(0, x_max + 1, size=N * 2, dtype=np.int32),
            np.random.randint(0, y_max + 1, size=N * 2, dtype=np.int32),
            np.random.randint(0, z_max + 1, size=N * 2, dtype=np.int32)
        ))

        points = np.unique(points, axis=0)

        while points.shape[0] < N:
            extra = np.column_stack((
                np.random.randint(0, x_max + 1, size=N, dtype=np.int32),
                np.random.randint(0, y_max + 1, size=N, dtype=np.int32),
                np.random.randint(0, z_max + 1, size=N, dtype=np.int32)
            ))
            points = np.unique(np.vstack((points, extra)), axis=0)

        np.random.shuffle(points)

        points = points[:N]

        out_path = os.path.join(out_dir, f"cloud_{tag}.npy")
        np.save(out_path, points)

        print(f"Saved {out_path} with shape {points.shape}")

    print("All scenes generated successfully.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducibility (default: 2026)"
    )
    args = parser.parse_args()

    main(seed=args.seed)

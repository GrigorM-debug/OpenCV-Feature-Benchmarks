"""
speed_benchmark.py
-------------------
Benchmarks the detection + description speed of multiple OpenCV
feature extraction algorithms across one or more images.

Usage:
    python benchmarks/speed_benchmark.py --images images/image_left.jpg
    python benchmarks/speed_benchmark.py --images images/image_left.jpg images/image_rigth.jpg
"""

import time
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def build_algorithms():
    algos = {}

    algos["ORB"]   = cv2.ORB.create()
    algos["SIFT"] = cv2.SIFT.create()
    algos["AKAZE"] = cv2.AKAZE.create()

    return algos

def benchmark_algorithm(name, detector, image, runs):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
        if len(image.shape) == 3 else image

    times   = []
    n_kps   = 0

    for i in range(runs):
        t0 = time.perf_counter()
        kps, des = detector.detectAndCompute(gray, None)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)   
        if i == 0:
            n_kps = len(kps) if kps else 0

    times = np.array(times)
    return {
        "name":      name,
        "mean_ms":   float(np.mean(times)),
        "std_ms":    float(np.std(times)),
        "min_ms":    float(np.min(times)),
        "max_ms":    float(np.max(times)),
        "keypoints": n_kps,
    }


def run_benchmarks(images, runs):
    algorithms = build_algorithms()
    aggregated = []

    for name, detector in algorithms.items():
        per_image = []
        kps_total = 0

        for img in images:
            result     = benchmark_algorithm(name, detector, img, runs)
            per_image.append(result["mean_ms"])
            kps_total += result["keypoints"]

        aggregated.append({
            "name":      name,
            "mean_ms":   float(np.mean(per_image)),
            "std_ms":    float(np.std(per_image)),
            "keypoints": kps_total // len(images),   
        })

    aggregated.sort(key=lambda r: r["mean_ms"])
    return aggregated


def print_table(results):
    header = f"{'Algorithm':<20} {'Mean (ms)':>10} {'Std (ms)':>10} {'Keypoints':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        print(
            f"{r['name']:<20} "
            f"{r['mean_ms']:>10.2f} "
            f"{r['std_ms']:>10.2f} "
            f"{r['keypoints']:>10}"
        )
    print("=" * len(header) + "\n")


def plot_results(results, output_path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    names    = [r["name"]      for r in results]
    means    = [r["mean_ms"]   for r in results]
    stds     = [r["std_ms"]    for r in results]
    keypts   = [r["keypoints"] for r in results]

    x      = np.arange(len(names))
    width  = 0.4
    colors = plt.cm.tab10.colors

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Feature Extraction – Speed Comparison", fontsize=14, fontweight="bold")

    bars = ax1.bar(x, means, width, yerr=stds, capsize=4,
                   color=colors[:len(names)], edgecolor="black", linewidth=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.set_ylabel("Mean detection time (ms)")
    ax1.set_title("Detection Speed")
    ax1.bar_label(bars, fmt="%.1f ms", padding=4, fontsize=8)
    ax1.set_ylim(0, max(means) * 1.3)

    bars2 = ax2.bar(x, keypts, width,
                    color=colors[:len(names)], edgecolor="black", linewidth=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha="right")
    ax2.set_ylabel("Keypoints detected (avg)")
    ax2.set_title("Keypoints per Image")
    ax2.bar_label(bars2, padding=4, fontsize=8)
    ax2.set_ylim(0, max(keypts) * 1.3 if keypts else 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"[INFO] Chart saved → {output_path}")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark OpenCV feature extraction algorithms."
    )
    parser.add_argument(
        "--images", nargs="+", required=True,
        help="Path(s) to input image(s).",
    )
    parser.add_argument(
        "--runs", type=int, default=10,
        help="Number of repeated runs per algorithm (default: 10).",
    )
    return parser.parse_args()


def load_images(paths):
    images = []
    for p in paths:
        print(p)
        img = cv2.imread(p)
        if img is None:
            print(f"[WARN] Could not load image: {p} – skipping.")
            continue
        images.append(img)
    if not images:
        raise FileNotFoundError("No valid images were loaded. Check your --images paths.")
    return images


def main():
    args   = parse_args()
    print(args)
    images = load_images(args.images)

    print(f"[INFO] Loaded {len(images)} image(s).")
    print(f"[INFO] Running each algorithm {args.runs} time(s) per image …\n")

    results = run_benchmarks(images, runs=args.runs)
    print_table(results)

    output_path = "benchmarks/results/speed_comparison.png"

    plot_results(results, output_path)


if __name__ == "__main__":
    main()
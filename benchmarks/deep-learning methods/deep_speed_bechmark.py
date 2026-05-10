"""
deep_speed_benchmark.py
-----------------------
Benchmarks deep feature extraction speed for:
- SuperPoint
- ALIKED
- DISK

Usage:
    python deep_speed_bechmark.py --images "../../images/image7.jpg" "../../images/image8.jpg"
"""

import argparse
import time
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


def to_torch_image(image, torch, device):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
    if len(rgb.shape) == 2:
        img_t = torch.from_numpy(rgb).float().unsqueeze(0).unsqueeze(0)
    else:
        img_t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)
    return (img_t / 255.0).to(device)


def load_deep_extractors(max_keypoints):
    repo_root = Path(__file__).resolve().parents[2]
    lightglue_root = repo_root / "deep-leaning-methods" / "LightGlue"
    if str(lightglue_root) not in sys.path:
        sys.path.append(str(lightglue_root))

    import torch
    from lightglue import SuperPoint, ALIKED, DISK  # type: ignore[reportMissingImports]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractors = {
        "SuperPoint": SuperPoint(max_num_keypoints=max_keypoints).eval().to(device),
        "ALIKED": ALIKED(max_num_keypoints=max_keypoints).eval().to(device),
        "DISK": DISK(max_num_keypoints=max_keypoints).eval().to(device),
    }
    return extractors, torch, device


def benchmark_algorithm(name, extractor, image, runs, torch, device):
    times = []
    n_kps = 0

    for i in range(runs):
        img_t = to_torch_image(image, torch, device)

        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            feats = extractor.extract(img_t)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        keypoints = feats["keypoints"]
        n_cur = int(keypoints.shape[1] if keypoints.ndim == 3 else keypoints.shape[0])

        times.append((t1 - t0) * 1000.0)
        if i == 0:
            n_kps = n_cur

    times = np.array(times, dtype=np.float64)
    return {
        "name": name,
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "keypoints": n_kps,
    }


def run_benchmarks(images, runs, max_keypoints):
    extractors, torch, device = load_deep_extractors(max_keypoints=max_keypoints)
    print(f"[INFO] Device: {device}")

    aggregated = []
    for name, extractor in extractors.items():
        per_image = []
        kps_total = 0

        for img in images:
            result = benchmark_algorithm(name, extractor, img, runs, torch, device)
            per_image.append(result["mean_ms"])
            kps_total += result["keypoints"]

        aggregated.append({
            "name": name,
            "mean_ms": float(np.mean(per_image)),
            "std_ms": float(np.std(per_image)),
            "keypoints": int(kps_total / len(images)),
        })

    aggregated.sort(key=lambda r: r["mean_ms"])
    return aggregated


def print_table(results):
    header = f"{'Algorithm':<14} {'Mean (ms)':>10} {'Std (ms)':>10} {'Keypoints':>10}"
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for r in results:
        print(
            f"{r['name']:<14} "
            f"{r['mean_ms']:>10.2f} "
            f"{r['std_ms']:>10.2f} "
            f"{r['keypoints']:>10}"
        )
    print(f"{sep}\n")


def plot_results(results, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    names = [r["name"] for r in results]
    means = [r["mean_ms"] for r in results]
    keypts = [r["keypoints"] for r in results]

    x = np.arange(len(names))
    width = 0.55
    colors = plt.cm.Set2.colors

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("Deep Feature Extraction - Speed Comparison", fontsize=14, fontweight="bold")

    bars = ax1.bar(x, means, width, color=colors[:len(names)], edgecolor="black", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15)
    ax1.set_ylabel("Mean extraction time (ms)")
    ax1.set_title("Detection Speed")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.bar_label(bars, fmt="%.1f ms", padding=4, fontsize=8)
    ax1.set_ylim(0, max(means) * 1.2 if means else 1.0)

    bars2 = ax2.bar(x, keypts, width, color=colors[:len(names)], edgecolor="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15)
    ax2.set_ylabel("Keypoints detected (avg)")
    ax2.set_title("Keypoints per Image")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.bar_label(bars2, padding=4, fontsize=8)
    ax2.set_ylim(0, max(keypts) * 1.35 if keypts else 1.0)

    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"[INFO] Chart saved -> {output_path}")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark deep feature extraction speed.")
    parser.add_argument("--images", nargs="+", required=True, help="Input image paths.")
    parser.add_argument("--runs", type=int, default=10, help="Repeated runs per image (default: 10).")
    parser.add_argument("--max-keypoints", type=int, default=2048, help="Max keypoints (default: 2048).")
    return parser.parse_args()


def load_images(paths):
    images = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"[WARN] Could not load image: {p} - skipping.")
            continue
        images.append(img)
    if not images:
        raise FileNotFoundError("No valid images loaded.")
    return images


def main():
    args = parse_args()
    images = load_images(args.images)

    print(f"[INFO] Loaded {len(images)} image(s).")
    print(f"[INFO] Running each algorithm {args.runs} time(s) per image.\n")

    results = run_benchmarks(images, runs=args.runs, max_keypoints=args.max_keypoints)
    print_table(results)

    output_path = "benchmarks/deep-learning_algorithms/results/deep_speed_comparison.png"
    plot_results(results, output_path)


if __name__ == "__main__":
    main()
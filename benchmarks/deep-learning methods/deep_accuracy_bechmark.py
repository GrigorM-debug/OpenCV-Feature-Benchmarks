"""
deep_accuracy_benchmark.py
--------------------------
Benchmarks deep feature matching accuracy for:
- SuperPoint + LightGlue
- ALIKED + LightGlue
- DISK + LightGlue

Usage:
python deep_accuracy_bechmark.py --left "../../images/image7.jpg" --right "../../images/image8.jpg"
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def to_torch_image(image, torch, device):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
    if len(rgb.shape) == 2:
        img_t = torch.from_numpy(rgb).float().unsqueeze(0).unsqueeze(0)
    else:
        img_t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)
    return (img_t / 255.0).to(device)


def load_deep_models(max_keypoints=2048):
    repo_root = Path(__file__).resolve().parents[2]
    lightglue_root = repo_root / "deep-leaning-methods" / "LightGlue"
    if str(lightglue_root) not in sys.path:
        sys.path.append(str(lightglue_root))

    import torch
    from lightglue import LightGlue, SuperPoint, ALIKED, DISK  # type: ignore[reportMissingImports]
    from lightglue.utils import rbd  # type: ignore[reportMissingImports]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {
        "SuperPoint": {
            "extractor": SuperPoint(max_num_keypoints=max_keypoints).eval().to(device),
            "matcher": LightGlue(features="superpoint").eval().to(device),
        },
        "ALIKED": {
            "extractor": ALIKED(max_num_keypoints=max_keypoints).eval().to(device),
            "matcher": LightGlue(features="aliked").eval().to(device),
        },
        "DISK": {
            "extractor": DISK(max_num_keypoints=max_keypoints).eval().to(device),
            "matcher": LightGlue(features="disk").eval().to(device),
        },
    }
    return models, torch, device, rbd


def compute_homography_inliers(kps1, kps2, good_matches):
    if len(good_matches) < 4:
        return 0, None

    src = np.float32([kps1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst = np.float32([kps2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if mask is None:
        return 0, None

    return int(mask.sum()), H


def evaluate_deep_algorithm(name, extractor, matcher, img1, img2, score_threshold, torch, device, rbd):
    image0 = to_torch_image(img1, torch, device)
    image1 = to_torch_image(img2, torch, device)

    with torch.no_grad():
        feats0 = extractor.extract(image0)
        feats1 = extractor.extract(image1)
        matches01 = matcher({"image0": feats0, "image1": feats1})

    feats0, feats1, matches01 = [rbd(x) for x in (feats0, feats1, matches01)]

    matches = matches01["matches"]
    scores = matches01["scores"]

    total_matches = int(matches.shape[0])

    keep = scores > score_threshold
    good_idx = matches[keep]
    good_scores = scores[keep]
    good_count = int(good_idx.shape[0])

    kps0_np = feats0["keypoints"].detach().cpu().numpy()
    kps1_np = feats1["keypoints"].detach().cpu().numpy()

    kp0 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in kps0_np]
    kp1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in kps1_np]

    good_matches = []
    if good_count > 0:
        idx_np = good_idx.detach().cpu().numpy()
        s_np = good_scores.detach().cpu().numpy()
        good_matches = [
            cv2.DMatch(int(idx_np[i, 0]), int(idx_np[i, 1]), float(1.0 - s_np[i]))
            for i in range(good_count)
        ]

    inliers, _ = compute_homography_inliers(kp0, kp1, good_matches)

    return {
        "name": name,
        "kps_left": len(kp0),
        "kps_right": len(kp1),
        "total_matches": total_matches,
        "good_matches": good_count,
        "inliers": inliers,
        "match_ratio": (good_count / total_matches) if total_matches > 0 else 0.0,
        "inlier_ratio": (inliers / good_count) if good_count > 0 else 0.0,
        "_kps1": kp0,
        "_kps2": kp1,
        "_good": good_matches,
        "_img1": img1,
        "_img2": img2,
    }


def print_table(results):
    cols = (
        f"{'Algorithm':<14} {'KPs L':>7} {'KPs R':>7} "
        f"{'Matches':>9} {'Good':>7} {'Inliers':>9} "
        f"{'Match%':>8} {'Inlier%':>9}"
    )
    sep = "=" * len(cols)
    print(f"\n{sep}\n{cols}\n{sep}")
    for r in results:
        print(
            f"{r['name']:<14} "
            f"{r['kps_left']:>7} "
            f"{r['kps_right']:>7} "
            f"{r['total_matches']:>9} "
            f"{r['good_matches']:>7} "
            f"{r['inliers']:>9} "
            f"{r['match_ratio'] * 100:>7.1f}% "
            f"{r['inlier_ratio'] * 100:>8.1f}%"
        )
    print(f"{sep}\n")


def plot_results(results, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    names = [r["name"] for r in results]
    match_ratios = [r["match_ratio"] * 100 for r in results]
    inlier_ratios = [r["inlier_ratio"] * 100 for r in results]
    good_counts = [r["good_matches"] for r in results]
    inlier_counts = [r["inliers"] for r in results]

    x = np.arange(len(names))
    width = 0.35
    colors = plt.cm.tab10.colors

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    fig.suptitle("Deep Feature Matching - Accuracy Comparison", fontsize=14, fontweight="bold")

    b1 = ax1.bar(x - width / 2, good_counts, width, label="Good matches",
                 color=colors[0], edgecolor="black", linewidth=0.6)
    b2 = ax1.bar(x + width / 2, inlier_counts, width, label="Inliers (RANSAC)",
                 color=colors[1], edgecolor="black", linewidth=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.set_ylabel("Count")
    ax1.set_title("Match & Inlier Counts")
    ax1.legend(fontsize=8)
    ax1.bar_label(b1, padding=3, fontsize=7)
    ax1.bar_label(b2, padding=3, fontsize=7)
    ax1.set_ylim(0, max(good_counts + [1]) * 1.3)

    b3 = ax2.bar(x - width / 2, match_ratios, width, label="Match ratio %",
                 color=colors[2], edgecolor="black", linewidth=0.6)
    b4 = ax2.bar(x + width / 2, inlier_ratios, width, label="Inlier ratio %",
                 color=colors[3], edgecolor="black", linewidth=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha="right")
    ax2.set_ylabel("Ratio (%)")
    ax2.set_title("Match & Inlier Ratios")
    ax2.set_ylim(0, 110)
    ax2.legend(fontsize=8)
    ax2.bar_label(b3, fmt="%.1f%%", padding=3, fontsize=7)
    ax2.bar_label(b4, fmt="%.1f%%", padding=3, fontsize=7)

    best = max(results, key=lambda r: r["inliers"])
    vis = cv2.drawMatches(
        best["_img1"], best["_kps1"],
        best["_img2"], best["_kps2"],
        best["_good"][:50],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    ax3.imshow(vis_rgb)
    ax3.set_title(
        f"Best algorithm: {best['name']} - "
        f"{best['good_matches']} good matches, {best['inliers']} inliers\n"
        f"(showing top 50) - Matcher used: LightGlue",
        fontsize=12
    )
    ax3.axis("off")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Chart saved -> {output_path}")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark deep feature matching accuracy.")
    parser.add_argument("--left", required=True, help="Left/query image.")
    parser.add_argument("--right", required=True, help="Right/train image.")
    parser.add_argument("--deep-score-threshold", type=float, default=0.5,
                        help="LightGlue score threshold (default: 0.5).")
    parser.add_argument("--max-keypoints", type=int, default=2048,
                        help="Max keypoints for extractors (default: 2048).")
    return parser.parse_args()


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def main():
    args = parse_args()
    img1 = load_image(args.left)
    img2 = load_image(args.right)

    print(f"[INFO] Left image : {args.left}  {img1.shape}")
    print(f"[INFO] Right image: {args.right}  {img2.shape}")

    models, torch, device, rbd = load_deep_models(max_keypoints=args.max_keypoints)
    print(f"[INFO] Device: {device}")

    results = []
    for name, pack in models.items():
        print(f"[INFO] Evaluating {name} + LightGlue ...")
        result = evaluate_deep_algorithm(
            name=name,
            extractor=pack["extractor"],
            matcher=pack["matcher"],
            img1=img1,
            img2=img2,
            score_threshold=args.deep_score_threshold,
            torch=torch,
            device=device,
            rbd=rbd,
        )
        results.append(result)

    results.sort(key=lambda r: r["inliers"], reverse=True)

    print_table(results)

    output_path = "benchmarks/deep-learning_algorithms/results/deep_accuracy_comparison.png"
    plot_results(results, output_path)


if __name__ == "__main__":
    main()
"""
accuracy_comparison.py
----------------------
Benchmarks the matching accuracy of multiple OpenCV feature extraction
algorithms using a pair of images (e.g. stereo pair, same scene different angle).

Metrics reported:
  - Total keypoints detected
  - Total matches found
  - Good matches (after Lowe's ratio test)
  - Match ratio (good / total)
  - Homography inliers (geometric verification via RANSAC)
  - Inlier ratio (inliers / good matches)

Usage:
    python accuracy_benchmark.py --left "../images/image11.jpeg" --right "../images/image12.jpeg" --matcher bf_matcher
    python accuracy_benchmark.py --left "../images/image11.jpeg" --right "../images/image12.jpeg" --matcher flann
    python accuracy_benchmark.py --left "../images/image11.jpeg" --right "../images/image12.jpeg" --matcher bf_matcher --ratio 0.8 
"""

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


def build_algorithms():
    algos = {}

    algos["ORB"]   = (cv2.ORB_create(), cv2.NORM_HAMMING)
    algos["AKAZE"] = (cv2.AKAZE_create(), cv2.NORM_HAMMING)

    algos["SIFT"]  = (cv2.SIFT_create(), cv2.NORM_L2)

    return algos

def match_features(des1, des2, norm_type, ratio, matcher_name):
    if des1 is None or des2 is None:
        return [], []

    matcher= None

    if(matcher_name.lower() == "bf_matcher"): 
        matcher = cv2.BFMatcher(norm_type)
    elif(matcher_name.lower() == "flann"): 
        if norm_type == cv2.NORM_L2 or norm_type == cv2.NORM_L1:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50) 
            
        elif norm_type in [cv2.NORM_HAMMING, cv2.NORM_HAMMING2]:
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,      
                                key_size=12,          
                                multi_probe_level=1)  
            search_params = dict(checks=50)
        
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

    all_matches = matcher.knnMatch(des1, des2, k=2)
    good        = []

    for pair in all_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)

    return all_matches, good


def compute_homography_inliers(kps1, kps2, good_matches):
    if len(good_matches) < 4:
        return 0, None
    src = np.float32([kps1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst = np.float32([kps2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if mask is None:
        return 0, None

    return int(mask.sum()), H

def evaluate_algorithm(name, detector, norm_type, img1, img2, ratio, matcher_name):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

    kps1, des1 = detector.detectAndCompute(gray1, None)
    kps2, des2 = detector.detectAndCompute(gray2, None)

    all_matches, good_matches = match_features(des1, des2, norm_type, ratio, matcher_name)
    inliers, H                = compute_homography_inliers(kps1, kps2, good_matches)

    total_matches = len(all_matches)
    good_count    = len(good_matches)
    kp1_count     = len(kps1) if kps1 else 0
    kp2_count     = len(kps2) if kps2 else 0

    return {
        "name":          name,
        "kps_left":      kp1_count,
        "kps_right":     kp2_count,
        "total_matches": total_matches,
        "good_matches":  good_count,
        "inliers":       inliers,
        "match_ratio":   good_count / total_matches  if total_matches > 0 else 0.0,
        "inlier_ratio":  inliers    / good_count     if good_count    > 0 else 0.0,
        "_kps1":         kps1,
        "_kps2":         kps2,
        "_good":         good_matches,
        "_img1":         img1,
        "_img2":         img2,
    }

def print_table(results):
    cols = (
        f"{'Algorithm':<20} {'KPs L':>7} {'KPs R':>7} "
        f"{'Matches':>9} {'Good':>7} {'Inliers':>9} "
        f"{'Match%':>8} {'Inlier%':>9}"
    )
    sep = "=" * len(cols)
    print(f"\n{sep}\n{cols}\n{sep}")
    for r in results:
        print(
            f"{r['name']:<20} "
            f"{r['kps_left']:>7} "
            f"{r['kps_right']:>7} "
            f"{r['total_matches']:>9} "
            f"{r['good_matches']:>7} "
            f"{r['inliers']:>9} "
            f"{r['match_ratio']*100:>7.1f}% "
            f"{r['inlier_ratio']*100:>8.1f}%"
        )
    print(f"{sep}\n")


def plot_results(results, output_path, matcher_name):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    names         = [r["name"]          for r in results]
    match_ratios  = [r["match_ratio"]   * 100 for r in results]
    inlier_ratios = [r["inlier_ratio"]  * 100 for r in results]
    good_counts   = [r["good_matches"]  for r in results]
    inlier_counts = [r["inliers"]       for r in results]

    x      = np.arange(len(names))
    width  = 0.35
    colors = plt.cm.tab10.colors

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])   

    fig.suptitle("Feature Matching – Accuracy Comparison", fontsize=14, fontweight="bold")

    b1 = ax1.bar(x - width / 2, good_counts,   width, label="Good matches",
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

    b3 = ax2.bar(x - width / 2, match_ratios,  width, label="Match ratio %",
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
    vis  = cv2.drawMatches(
        best["_img1"], best["_kps1"],
        best["_img2"], best["_kps2"],
        best["_good"][:50],            
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    ax3.imshow(vis_rgb)
    ax3.set_title(
        f"Best algorithm: {best['name']}  —  "
        f"{best['good_matches']} good matches, {best['inliers']} inliers\n" # Added \n here
        f"(showing top 50)  —  Matcher used: {matcher_name}",
        fontsize=12
    )
    ax3.axis("off")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Chart saved → {output_path}")
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark OpenCV feature matching accuracy."
    )
    parser.add_argument("--left",    required=True, help="Path to the left/query image.")
    parser.add_argument("--right",   required=True, help="Path to the right/train image.")
    parser.add_argument(
        "--ratio", type=float, default=0.75,
        help="Lowe's ratio test threshold (default: 0.75).",
    )
    parser.add_argument("--matcher_name", required=True, help="Which feature matching algoritym to use")
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

    print(f"[INFO] Left image  : {args.left}  {img1.shape}")
    print(f"[INFO] Right image : {args.right}  {img2.shape}")
    print(f"[INFO] Ratio test  : {args.ratio}\n")

    algorithms = build_algorithms()
    results    = []

    for name, (detector, norm_type) in algorithms.items():
        print(f"[INFO] Evaluating {name} …")
        result = evaluate_algorithm(name, detector, norm_type, img1, img2, args.ratio, args.matcher_name)
        results.append(result)

    results.sort(key=lambda r: r["inliers"], reverse=True)

    print_table(results)

    output_path = "benchmarks/results/accuracy_comparison.png"
    plot_results(results, output_path, args.matcher_name)


if __name__ == "__main__":
    main()

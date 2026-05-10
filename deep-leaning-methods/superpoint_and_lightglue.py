from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
import argparse
import cv2 as cv
from datetime import datetime
import numpy as np
import pathlib
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="SuperPoint + LightGlue feature matching")
    parser.add_argument("image0", type=str, help="Path to the first image")
    parser.add_argument("image1", type=str, help="Path to the second image")
    return parser.parse_args()

def main():
    args = parse_args()

    extractor = SuperPoint(max_num_keypoints=None).eval().cuda()
    # matcher = LightGlue(features='superpoint').eval().cuda()
    matcher = LightGlue(features="superpoint", depth_confidence=-1, width_confidence=-1).eval().cuda()

    image0 = load_image(args.image0).cuda()
    image1 = load_image(args.image1).cuda()

    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)

    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    matches = matches01['matches']         
    scores  = matches01['scores'] 

    score_threshold = 0.5
    mask = scores > score_threshold
    matches = matches[mask]
    scores  = scores[mask]

    num_matches = len(matches)

    kpts0 = feats0['keypoints'][matches[:, 0]].cpu().numpy()  # (K, 2)
    kpts1 = feats1['keypoints'][matches[:, 1]].cpu().numpy()  # (K, 2)

    inliers = 0
    inlier_mask = None

    if num_matches >= 4:
        _, inlier_mask = cv.findHomography(
            kpts0.reshape(-1, 1, 2),
            kpts1.reshape(-1, 1, 2),
            cv.RANSAC,
            ransacReprojThreshold=3.0,
            confidence=0.999
        )
        if inlier_mask is not None:
            inlier_mask = inlier_mask.ravel().astype(bool)
            inliers = int(inlier_mask.sum())

    precision = (inliers / num_matches * 100.0) if num_matches > 0 else 0.0

    kp0 = [cv.KeyPoint(float(k[0]), float(k[1]), 1) for k in feats0['keypoints'].cpu().numpy()]
    kp1 = [cv.KeyPoint(float(k[0]), float(k[1]), 1) for k in feats1['keypoints'].cpu().numpy()]

    matched_indices = matches.cpu().numpy()
    if inlier_mask is not None:
        draw_matches = [
            cv.DMatch(int(matched_indices[i, 0]), int(matched_indices[i, 1]), float(1 - scores[i].cpu()))
            for i in range(num_matches) if inlier_mask[i]
        ]
    else:
        draw_matches = [
            cv.DMatch(int(matched_indices[i, 0]), int(matched_indices[i, 1]), float(1 - scores[i].cpu()))
            for i in range(num_matches)
        ]

    img0_bgr = cv.imread(args.image0)
    img1_bgr = cv.imread(args.image1)

    img3 = cv.drawMatches(
        img0_bgr, kp0,
        img1_bgr, kp1,
        draw_matches, None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv.putText(img3, f"Features L/R: {len(kp0)}/{len(kp1)}",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)

    cv.putText(img3, f"Matches (after score filter): {num_matches}",
               (10, 65), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)

    cv.putText(img3, f"Inliers (RANSAC): {inliers}",
               (10, 100), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)

    cv.putText(img3, f"Precision: {precision:.2f}%",
               (10, 135), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)

    results_dir = pathlib.Path(__file__).resolve().parent.parent / "results/deep-learning_algorithms"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = results_dir / f"SuperPoint_LightGlue_{timestamp}.png"
    cv.imwrite(str(output_path), img3)

    cv.namedWindow("SuperPoint and LightGlue", cv.WINDOW_FULLSCREEN)
    cv.imshow("SuperPoint and LightGlue", img3)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
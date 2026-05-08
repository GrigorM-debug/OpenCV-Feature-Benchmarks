import argparse
from pathlib import Path

import cv2 as cv
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="ORB + BF feature matching")
    parser.add_argument("left_image", type=Path, help="Path to the left image")
    parser.add_argument("right_image", type=Path, help="Path to the right image")
    return parser.parse_args()


def main():
    args = parse_args()

    img1 = cv.imread(str(args.left_image))
    img2 = cv.imread(str(args.right_image))

    if img1 is None or img2 is None:
        raise FileNotFoundError(
            "Could not read one or both images. "
            f"left='{args.left_image}', right='{args.right_image}'"
        )

    orb = cv.ORB.create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        raise ValueError("No descriptors found in one or both images.")

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    number_of_matches = len(matches)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])

    inliers = 0
    if len(good_matches) >= 4:
        src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        _, inlier_mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        if inlier_mask is not None:
            inliers = int(inlier_mask.sum())

    precision = (inliers / len(good_matches) * 100.0) if good_matches else 0.0

    img3 = cv.drawMatchesKnn(
        img1,
        kp1,
        img2,
        kp2,
        good_matches,
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    cv.putText(
        img3,
        f"Features L/R: {len(kp1)}/{len(kp2)}",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv.LINE_AA,
    )
    
    cv.putText(
        img3,
        f"Matches (after score filter): {number_of_matches}",
        (10, 65),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv.LINE_AA,
    )

    cv.putText(
        img3,
        f"Inliers: {inliers}",
        (10, 100),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv.LINE_AA,
    )
    
    cv.putText(
        img3,
        f"Precision: {precision:.2f}%",
        (10, 135),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv.LINE_AA,
    )
    results_dir = Path(__file__).resolve().parent.parent / "results/traditional_algorithms"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "ORB_BF_Matcher.png"
    cv.imwrite(str(output_path), img3)

    cv.namedWindow("ORB BF Matches", cv.WINDOW_FULLSCREEN)
    cv.imshow("ORB BF Matches", img3)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()

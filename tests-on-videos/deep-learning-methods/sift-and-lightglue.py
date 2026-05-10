from lightglue import LightGlue, SIFT
from lightglue.utils import rbd
import argparse
import cv2 as cv
import pygame
import torch
from pathlib import Path

W = 1920//2
H = 1080//2

def parse_args():
    parser = argparse.ArgumentParser(description="SIFT + LightGlue feature matching")
    parser.add_argument(
        "video",
        type=str,
        nargs="?",
        default=None,
        help="Path to the video (optional)",
    )
    return parser.parse_args()


def resolve_video_path(video_arg):
    script_dir = Path(__file__).resolve().parent
    default_video = script_dir.parent / "videos" / "video.mp4"

    if video_arg is None:
        return default_video

    p = Path(video_arg)
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()

class Display2D(object):
    def __init__(self, W, H):
        self.W = W
        self.H = H
        pygame.init()
        self.window = pygame.display.set_mode((W, H))
        pygame.display.set_caption("SIFT + LightGlue on video")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 30, bold=True)

    def paint(self, img):
        self.clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)
        img = cv.resize(img, (self.W, self.H))

        if len(img.shape) == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        else:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        pygame.surfarray.blit_array(self.window, img.swapaxes(0, 1))
        fps = self.clock.get_fps()
        text_surface = self.font.render(f"FPS: {fps:.1f}", True, (255, 255, 0))
        self.window.blit(text_surface, (10, 10))
        pygame.display.flip()

class FeatureExtractor(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = SIFT(max_num_keypoints=2048).eval().to(self.device)

    def extract(self, gray):
        # Convert grayscale uint8 frame to tensor [1, 1, H, W] in [0, 1].
        img_t = torch.from_numpy(gray).float().to(self.device) / 255.0
        img_t = img_t.unsqueeze(0).unsqueeze(0)
        feats = self.extractor.extract(img_t)
        return feats


class FeatureMatcher(object):
    def __init__(self, device):
        self.matcher = LightGlue(features="sift").eval().to(device)
        self.prev_feats = None
        self.score_threshold = 0.2
        self.keypoint_radius = 5
        self.keypoint_thickness = 3
        self.match_line_thickness = 4

    def match_and_draw(self, gray, feats):
        vis = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        kpts = feats["keypoints"][0]

        if self.prev_feats is None:
            kp_np = kpts.cpu().numpy()
            for p in kp_np:
                cv.circle(
                    vis,
                    (int(p[0]), int(p[1])),
                    self.keypoint_radius,
                    (0, 255, 0),
                    self.keypoint_thickness,
                )
            self.prev_feats = feats
            return vis, 0, int(kpts.shape[0])

        matches01 = self.matcher({"image0": self.prev_feats, "image1": feats})
        matches01 = rbd(matches01)

        matches = matches01["matches"]
        scores = matches01["scores"]

        if scores.numel() > 0:
            keep = scores > self.score_threshold
            matches = matches[keep]

        prev_kpts = self.prev_feats["keypoints"][0].cpu().numpy()
        curr_kpts = kpts.cpu().numpy()

        matches_np = matches.cpu().numpy() if matches.numel() > 0 else []
        n_good = 0
        for m in matches_np:
            p_prev = prev_kpts[int(m[0])]
            p_curr = curr_kpts[int(m[1])]
            x1, y1 = int(p_prev[0]), int(p_prev[1])
            x2, y2 = int(p_curr[0]), int(p_curr[1])
            cv.line(
                vis,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                self.match_line_thickness,
            )
            cv.circle(
                vis,
                (x2, y2),
                self.keypoint_radius,
                (0, 255, 0),
                self.keypoint_thickness,
            )
            n_good += 1

        self.prev_feats = feats
        return vis, int(matches.shape[0]), int(kpts.shape[0])


def process_frame(img, display, extractor, matcher):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    feats = extractor.extract(gray)

    vis, n_good, n_kpts = matcher.match_and_draw(gray, feats)

    label = f"Features: {n_kpts}  Matches: {n_good}"

    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    thickness = 3

    (text_w, text_h), baseline = cv.getTextSize(label, font, font_scale, thickness)
    x = (vis.shape[1] - text_w) // 2
    y = 20 + text_h

    cv.putText(
        vis,
        label,
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv.LINE_AA,
    )

    display.paint(vis)

def main():
    args = parse_args()
    video_path = resolve_video_path(args.video)
    print(f"[INFO] Video path: {video_path}")

    if not video_path.exists():
        print("[ERROR] Video file does not exist.")
        print("        Pass an absolute path or run with the correct relative path.")
        return

    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("[ERROR] OpenCV failed to open the video.")
        return

    display = Display2D(W, H)
    extractor = FeatureExtractor()
    matcher = FeatureMatcher(extractor.device)

    ret, frame = cap.read()
    if not ret or frame is None:
        print("[ERROR] Video opened but no readable frames were found.")
        cap.release()
        return

    while True:
        process_frame(frame, display, extractor, matcher)

        if cv.waitKey(1) == ord('q'):
            break
        ret, frame = cap.read()
        if not ret or frame is None:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
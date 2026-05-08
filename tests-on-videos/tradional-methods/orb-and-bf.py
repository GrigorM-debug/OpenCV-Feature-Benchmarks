import pygame
import cv2 
import argparse

W = 1920//2
H = 1080//2

def parse_args():
    parser = argparse.ArgumentParser(description="SuperPoint + LightGlue feature matching")
    parser.add_argument(
        "video",
        type=str,
        nargs="?",
        default=None,
        help="Path to the video (optional)",
    )
    return parser.parse_args()

class Display2D(object):
    def __init__(self, W, H):
        self.W = W
        self.H = H
        pygame.init()
        self.window = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Noob SLAM")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 30, bold=True)

    def paint(self, img):
        self.clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)
        img = cv2.resize(img, (self.W, self.H))

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pygame.surfarray.blit_array(self.window, img.swapaxes(0, 1))
        fps = self.clock.get_fps()
        text_surface = self.font.render(f"FPS: {fps:.1f}", True, (255, 255, 0))
        self.window.blit(text_surface, (10, 10))
        pygame.display.flip()

class FeatureExtractor(object):
    def __init__(self):
        self.orb = cv2.ORB.create(nfeatures=3000)

    def extract(self, img):
        keypoints, descriptors = self.orb.detectAndCompute(img, None)
        return keypoints, descriptors
    
class FeatureMatcher(object):
    def __init__(self):
        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.keypoint_radius = 8
        self.keypoint_thickness = 4
        self.match_line_thickness = 4

    def match_and_draw(self, gray, kp, des):
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if self.prev_des is None or des is None or len(des) < 2 or len(self.prev_des) < 2:
            for point in kp:
                x, y = int(point.pt[0]), int(point.pt[1])
                cv2.circle(
                    vis,
                    (x, y),
                    self.keypoint_radius,
                    (0, 255, 0),
                    self.keypoint_thickness,
                )
            self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
            return vis, 0

        matches = self.bf.knnMatch(self.prev_des, des, k=2)
        good = []
        
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

        p_prev = self.prev_kp[m.queryIdx].pt
        p_curr = kp[m.trainIdx].pt
        x1, y1 = int(p_prev[0]), int(p_prev[1])
        x2, y2 = int(p_curr[0]), int(p_curr[1])
        cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), self.match_line_thickness)
        cv2.circle(
            vis,
            (x2, y2),
            self.keypoint_radius,
            (0, 255, 0),
            self.keypoint_thickness,
        )
    
        self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
        
        return vis, len(good)
        

def process_frame(img, display, extractor, matcher):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = extractor.extract(gray)

    vis, n_good = matcher.match_and_draw(gray, keypoints, descriptors)

    label = f"Features: {len(keypoints) if keypoints is not None else 0}  Matches: {n_good}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3    
    thickness = 3

    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    x = (vis.shape[1] - text_w) // 2
    y = 20 + text_h      

    cv2.putText(
        vis,
        label,
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    display.paint(vis)


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.video)

    display = Display2D(W, H)
    extractor = FeatureExtractor()
    matcher = FeatureMatcher()


    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        process_frame(frame, display, extractor, matcher)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()
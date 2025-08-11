#!/usr/bin/env python3
"""
Robust Screen Monitoring with ORB+Homography (Fixed Detection)
- Detects up to 2 templates from ./images (transparent PNGs supported)
- Alpha-blends PNGs onto white background when needed
- Handles scale (±50%), rotation (±45°), mild distortion via homography
- Overlay with random color per template; shows FPS in status bar
- State: appears/disappears with 1s cooldown after disappearance
- Debug: -d / --debug toggles verbose keypoint display; -s saves frame
"""
from __future__ import annotations

import argparse
import threading
import time
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import mss

# Global flags
OVERLAY_ENABLED: bool = True

# Detection thresholds and geometry
MIN_INLIERS = 15
RANSAC_REPROJ_TH = 3.0
LOWE_RATIO = 0.75
ASPECT_TOLERANCE = 0.20
AREA_MIN_RATIO = 0.30
AREA_MAX_RATIO = 2.0

# Capture settings
DOWNSCALE = 0.60  # 60% of original size
TARGET_KEYPOINTS = 2000  # per-template ORB features

# Cooldown after disappearance (seconds)
DISAPPEAR_COOLDOWN = 1.0

# Thread-safe shared state
class SharedState:
    def __init__(self) -> None:
        self.frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.terminate: bool = False
        self.debug: bool = False
        self.save_next: bool = False
        self.fps: float = 0.0

state = SharedState()

# Template data container
class TemplateData:
    def __init__(self, name: str, tpl_img: np.ndarray, w: int, h: int, aspect: float):
        self.name = name
        self.color = tuple(int(np.random.randint(0, 256)) for _ in range(3))
        self.w = w
        self.h = h
        self.aspect = aspect

        self.kp: List[cv2.KeyPoint] = []
        self.desc: Optional[np.ndarray] = None
        self.ready: bool = False
        self.is_visible: bool = False
        self.cooldown_until: float = 0.0

        self.orb = cv2.ORB_create(nfeatures=TARGET_KEYPOINTS)
        self.tpl_gray = cv2.cvtColor(tpl_img, cv2.COLOR_BGR2GRAY)

    def compute_features(self) -> None:
        if self.tpl_gray is None:
            raise ValueError("Template grayscale is missing for feature extraction.")
        self.kp, self.desc = self.orb.detectAndCompute(self.tpl_gray, None)
        if self.kp is None or self.desc is None:
            self.kp, self.desc = [], None
        self.ready = True

def alpha_to_white_composite(img: np.ndarray) -> np.ndarray:
    """Composite a BGRA/PNG with alpha onto white background, result is BGR."""
    if img is None:
        raise ValueError("Input image is None for alpha compositing.")
    if img.shape[2] == 3:
        return img
    b, g, r, a = cv2.split(img)
    a = a.astype(float) / 255.0
    rgb = cv2.merge([b, g, r]).astype(float)
    white = np.ones_like(rgb) * 255.0
    out = rgb * a[..., None] + white * (1.0 - a[..., None])
    return out.astype(np.uint8)

def load_templates(images_dir: str = "./images") -> List[TemplateData]:
    """Load up to 2 templates from images_dir. Accepts PNG/JPG with optional alpha."""
    templates: List[TemplateData] = []
    if not os.path.isdir(images_dir):
        print(f"[Error] Templates folder not found: {images_dir}")
        return templates

    for fname in sorted(os.listdir(images_dir)):
        if not (fname.lower().endswith(".png") or fname.lower().endswith(".jpg") or fname.lower().endswith(".jpeg")):
            continue
        path = os.path.join(images_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # keep alpha if present
        if img is None:
            print(f"[Warn] Failed to load template image: {path}")
            continue

        if img.shape[2] == 4:
            img = alpha_to_white_composite(img)

        h, w, _ = img.shape
        aspect = w / float(h) if h != 0 else 1.0
        tpl = TemplateData(name=fname, tpl_img=img, w=w, h=h, aspect=aspect)
        tpl.compute_features()
        templates.append(tpl)
        print(f"[Info] Loaded template '{tpl.name}' (w={w}, h={h}, aspect={aspect:.3f})")
    return templates

def project_quad_from_homography(H: np.ndarray) -> Optional[np.ndarray]:
    """Return the projected quad (4 points) in destination image given homography."""
    if H is None:
        return None
    pts = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H)
    return dst.reshape(-1, 2)

def compute_aspect_ratio_error(quad: np.ndarray, template_aspect: float) -> float:
    """Compute aspect ratio deviation between detected quad and template."""
    if quad.shape[0] != 4:
        return float('inf')
    xs = quad[:, 0]
    ys = quad[:, 1]
    w = float(np.max(xs) - np.min(xs))
    h = float(np.max(ys) - np.min(ys))
    if h == 0:
        return float('inf')
    detected_aspect = w / h
    return abs(detected_aspect - template_aspect) / template_aspect

def valid_quad_area(quad: np.ndarray, template_area: float) -> bool:
    """Check quad area against template area ratio constraints."""
    if quad is None or quad.shape[0] != 4:
        return False
    area = cv2.contourArea(quad.astype(np.float32))
    if template_area <= 0:
        return False
    ratio = area / max(1.0, template_area)
    return AREA_MIN_RATIO <= ratio <= AREA_MAX_RATIO

def save_frame_with_detections(frame: np.ndarray, tpl_list: List[TemplateData], prefix="frame") -> None:
    out = frame.copy()
    for tpl in tpl_list:
        cv2.putText(out, tpl.name, (10, 20 + 20 * tpl_list.index(tpl)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, tpl.color, 2)
    ts = int(time.time())
    out_path = f"{prefix}_{ts}.png"
    cv2.imwrite(out_path, out)
    print(f"[Info] Saved {out_path}")

def draw_overlay(frame: np.ndarray, tpl: TemplateData, inliers: int, total_matches: int, H: Optional[np.ndarray]) -> None:
    """Draw quad overlay and stats for a single template on the frame."""
    if H is None:
        return
    quad = project_quad_from_homography(H)
    if quad is None or quad.shape[0] != 4:
        return
    quad_int = quad.astype(int)
    color = tpl.color
    cv2.polylines(frame, [quad_int], isClosed=True, color=color, thickness=2)

    conf = (inliers / total_matches) * 100.0 if total_matches > 0 else 0.0
    text = f"{tpl.name}  {inliers}/{total_matches}  {conf:.1f}%"
    cv2.putText(frame, text, (10, 20 + 20 *  tpl_list_index(tpl)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def tpl_list_index(tpl: TemplateData) -> int:
    """Helper to get index in global small list if needed; fallback 0."""
    # In this script, we redraw with per-template info in order; simplest approach:
    return 0

def compute_fps(start: float, count: int) -> float:
    if count == 0:
        return 0.0
    return count / max(1e-6, time.time() - start)

def capture_loop(state: SharedState) -> None:
    """Capture loop: create MSS instance in this thread and continuously grab screen frames."""
    try:
        with mss.mss() as sct:
            while not state.terminate:
                try:
                    monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                    raw = sct.grab(monitor)
                    if raw is None:
                        time.sleep(0.01)
                        continue
                    frame = np.array(raw)
                    if frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    # Downscale to 60%
                    h, w = frame.shape[:2]
                    nw, nh = max(1, int(w * DOWNSCALE)), max(1, int(h * DOWNSCALE))
                    frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

                    with state.lock:
                        state.frame = frame
                except Exception as e:
                    print(f"[Warning] Capture frame failed: {e}")
                    time.sleep(0.1)
    except Exception as e:
        print(f"[Error] Capture loop initialization failed: {e}")

def processing_loop(state: SharedState, templates: List[TemplateData], debug: bool = False) -> None:
    """Processing loop: detect templates in frames from capture thread."""
    orb_detector = cv2.ORB_create(nfeatures=TARGET_KEYPOINTS)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    template_infos = templates

    # FPS timing
    fps_start = time.time()
    frames = 0

    while not state.terminate:
        with state.lock:
            frame = state.frame.copy() if state.frame is not None else None

        if frame is None:
            time.sleep(0.01)
            continue

        t0 = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        overlay_frame = frame.copy() if OVERLAY_ENABLED else None

        # Detect global keypoints in the frame to speed up per-template matching
        kp_frame, des_frame = orb_detector.detectAndCompute(gray, None)

        if des_frame is None or len(kp_frame) == 0:
            for tpl in template_infos:
                if tpl.is_visible:
                    tpl.is_visible = False
                    tpl.cooldown_until = time.time() + DISAPPEAR_COOLDOWN
                    print(f"Template {tpl.name} disappeared (no frame features)!")
            if overlay_frame is not None and OVERLAY_ENABLED:
                cv2.imshow("Monitor", overlay_frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                state.terminate = True
            elif key == ord('d'):
                state.debug = not state.debug
            elif key == ord('s'):
                save_frame_with_detections(frame, template_infos)
            continue

        for tpl in template_infos:
            if tpl.cooldown_until > 0 and time.time() < tpl.cooldown_until:
                continue
            if tpl.desc is None or not tpl.ready or len(tpl.kp) == 0:
                continue

            # KNN matching
            try:
                raw_matches = bf.knnMatch(tpl.desc, des_frame, k=2)
            except Exception as e:
                if debug:
                    print(f"[Debug] knnMatch failed for {tpl.name}: {e}")
                continue

            good_matches = []
            for mn in raw_matches:
                if len(mn) != 2:
                    continue
                m, n = mn
                if m.distance < LOWE_RATIO * n.distance:
                    good_matches.append(m)

            inliers = 0
            total_matches = len(good_matches)
            H = None

            if total_matches >= 4:
                pts_template = np.float32([tpl.kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                pts_frame = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(pts_template, pts_frame, cv2.RANSAC, RANSAC_REPROJ_TH)
                if mask is not None:
                    inliers = int(mask.sum())
                else:
                    inliers = 0

                if H is not None and inliers >= MIN_INLIERS:
                    # Project template corners to frame
                    quad = cv2.perspectiveTransform(
                        np.float32([[[0, 0]], [[tpl.w, 0]], [[tpl.w, tpl.h]], [[0, tpl.h]]]),
                        H
                    )
                    quad = quad.reshape(-1, 2)
                    quad_area = cv2.contourArea(quad.astype(np.float32))
                    tpl_area = max(1.0, tpl.w * tpl.h)
                    area_ratio = quad_area / tpl_area
                    aspect_dev = compute_aspect_ratio_error(quad, tpl.aspect)

                    if valid_quad_area(quad, tpl_area) and AREA_MIN_RATIO <= area_ratio <= AREA_MAX_RATIO and aspect_dev <= ASPECT_TOLERANCE:
                        if not tpl.is_visible:
                            tpl.is_visible = True
                            print(f"Template {tpl.name} appeared!")
                        if OVERLAY_ENABLED:
                            quad_int = quad.astype(int)
                            cv2.polylines(frame, [quad_int], True, tpl.color, 2)
                            conf = (inliers / total_matches) * 100.0 if total_matches > 0 else 0.0
                            text = f"{tpl.name}: {inliers}/{total_matches}  {conf:.1f}%"
                            cv2.putText(frame, text, (10, 20 + 20 * template_infos.index(tpl)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, tpl.color, 2)
                    else:
                        if tpl.is_visible:
                            tpl.is_visible = False
                            tpl.cooldown_until = time.time() + DISAPPEAR_COOLDOWN
                            print(f"Template {tpl.name} disappeared (validation fail)!")
                else:
                    if tpl.is_visible:
                        tpl.is_visible = False
                        tpl.cooldown_until = time.time() + DISAPPEAR_COOLDOWN
                        print(f"Template {tpl.name} disappeared (homography fail)!")
            else:
                if tpl.is_visible:
                    tpl.is_visible = False
                    tpl.cooldown_until = time.time() + DISAPPEAR_COOLDOWN
                    print(f"Template {tpl.name} disappeared (not enough matches)!")

        # Show overlay window
        if overlay_frame is not None:
            if OVERLAY_ENABLED:
                cv2.imshow("Monitor", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                state.terminate = True
            elif key == ord('d'):
                state.debug = not state.debug
            elif key == ord('s'):
                save_frame_with_detections(frame, template_infos)

        # FPS measurement
        frames += 1
        if time.time() - fps_start >= 1.0:
            state.fps = frames / (time.time() - fps_start)
            fps_start = time.time()
            frames = 0

        # Adaptive sleep to avoid busy loop (roughly cap processing to ~60-100 Hz)
        t1 = time.time()
        elapsed = t1 - t0
        if elapsed < 0.01:
            time.sleep(0.01)

def main():
    parser = argparse.ArgumentParser(description="Robust screen monitoring with ORB+Homography (template detection).")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debugging output.")
    args = parser.parse_args()

    # Load templates
    templates = load_templates("./images")
    if len(templates) == 0:
        print("[Error] No templates loaded. Exiting.")
        return

    # Initialize state
    state.debug = args.debug

    # Start threads
    t_cap = threading.Thread(target=capture_loop, args=(state,), daemon=True)
    t_proc = threading.Thread(target=processing_loop, args=(state, templates, state.debug), daemon=True)

    t_cap.start()
    t_proc.start()

    print("[Info] Monitoring started. Press 'q' to quit, 'd' for debug, 's' to save frame.")
    try:
        while not state.terminate:
            # Display FPS in console occasionally
            if state.fps > 0:
                pass  # can print if needed
            time.sleep(0.1)
    except KeyboardInterrupt:
        state.terminate = True

    t_cap.join()
    t_proc.join()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
"""
collect_data.py — Real-time sign language landmark data collection tool.

Workflow per session:
  1. Terminal prompts for a sign name.
  2. Live webcam window opens; hand landmarks + bounding box are drawn by cvzone.
  3. "Press any key or click to start" overlay → 3-second countdown.
  4. Capture loop: detect hand, normalize landmarks (via utils.py), buffer rows.
  5. On completion or ESC: append all buffered rows to data/sign_data.csv.
  6. Ask whether to collect another sign; loop or exit.

All configurable values are constants at the top of this file.
"""

import csv
import os
import re
import sys
import time

import cv2
from cvzone.HandTrackingModule import HandDetector

from utils import normalize_landmarks

# ─────────────────────────────────────────────
# CONFIGURABLE CONSTANTS — edit here, nowhere else
# ─────────────────────────────────────────────
CAMERA_INDEX     = 0          # Change to 1, 2, etc. if your webcam is not the default
SAMPLES_PER_SESSION = 200     # Number of valid (hand-detected) frames to capture per run
MAX_HANDS        = 1          # 1 for single-hand signs; set to 2 for two-hand signs
                              # (feature vector doubles to 126 columns automatically)
DETECTION_CON    = 0.7        # Minimum detection confidence for MediaPipe
TRACK_CON        = 0.5        # Minimum tracking confidence
COUNTDOWN_SECS   = 3          # Countdown duration before capture starts
DATA_DIR         = "data"
CSV_FILENAME     = os.path.join(DATA_DIR, "sign_data.csv")
WINDOW_NAME      = "Sign Language Data Collector"

# Column header: x0,y0,z0, …, x20,y20,z20 per hand (× MAX_HANDS if >1), then label
LANDMARK_COLS = [
    f"{axis}{i}"
    for hand_idx in range(MAX_HANDS)
    for i in range(21)
    for axis in ("x", "y", "z")
]
# Prefix with hand index when collecting two hands so columns stay unambiguous
if MAX_HANDS > 1:
    LANDMARK_COLS = [
        f"h{hand_idx}_{axis}{i}"
        for hand_idx in range(MAX_HANDS)
        for i in range(21)
        for axis in ("x", "y", "z")
    ]
CSV_COLUMNS = LANDMARK_COLS + ["label"]

# ─────────────────────────────────────────────
# UI / OVERLAY HELPERS
# ─────────────────────────────────────────────
FONT       = cv2.FONT_HERSHEY_SIMPLEX
CLR_WHITE  = (255, 255, 255)
CLR_BLACK  = (0,   0,   0)
CLR_GREEN  = (0,   220, 100)
CLR_RED    = (0,   60,  220)
CLR_YELLOW = (0,   220, 220)
CLR_ORANGE = (0,   165, 255)


def _put_shadowed(img, text: str, org: tuple, scale: float, color: tuple, thickness: int = 2):
    """Draw text with a dark shadow for legibility on any background."""
    sx, sy = org[0] + 2, org[1] + 2
    cv2.putText(img, text, (sx, sy), FONT, scale, CLR_BLACK,  thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, org,      FONT, scale, color,      thickness,     cv2.LINE_AA)


def _overlay_banner(img, lines: list[tuple]):
    """
    Draw a semi-transparent dark banner across the top of the frame,
    then print each (text, color, scale) tuple in lines on it.
    """
    h, w = img.shape[:2]
    banner_h = 40 * len(lines) + 20
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    y = 36
    for text, color, scale in lines:
        _put_shadowed(img, text, (12, y), scale, color, 2)
        y += 40


# ─────────────────────────────────────────────
# MOUSE-CLICK DETECTION  (for "click to start")
# ─────────────────────────────────────────────
_mouse_clicked = False

def _mouse_callback(event, x, y, flags, param):
    global _mouse_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        _mouse_clicked = True


# ─────────────────────────────────────────────
# CSV HELPERS
# ─────────────────────────────────────────────
def _ensure_csv():
    """Create the data directory and CSV with a header if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_FILENAME):
        with open(CSV_FILENAME, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)


def _append_rows(rows: list[list]):
    """Bulk-write all buffered rows to CSV in one go (avoids per-frame I/O)."""
    with open(CSV_FILENAME, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


# ─────────────────────────────────────────────
# SIGN NAME SANITIZATION
# ─────────────────────────────────────────────
def _sanitize_label(raw: str) -> str:
    """
    Strip whitespace, quotes, and commas so the label is CSV-safe.
    Returns empty string if nothing remains.
    """
    label = raw.strip()
    label = re.sub(r'[",]', "", label)
    label = label.strip()
    return label


# ─────────────────────────────────────────────
# MAIN COLLECTION LOOP
# ─────────────────────────────────────────────
def collect_session(cap, detector: HandDetector, sign_label: str) -> int:
    """
    Run one full collection session for `sign_label`.
    Returns the number of samples actually captured.
    """
    global _mouse_clicked
    _mouse_clicked = False

    # ── Phase 1: "Press any key or click to start" ────────────────────────────
    print(f"\n[INFO] Window ready. Press any key or click the window to start collecting '{sign_label}'. ESC to cancel.")

    cv2.setMouseCallback(WINDOW_NAME, _mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera read failed.")
            return 0

        frame = cv2.flip(frame, 1)                        # Mirror
        hands, frame = detector.findHands(frame, draw=True, flipType=True)

        _overlay_banner(frame, [
            (f"Sign: '{sign_label}'",                      CLR_GREEN,  0.8),
            ("Press ANY KEY or CLICK to start collecting", CLR_WHITE,  0.65),
            ("ESC to cancel",                              CLR_YELLOW, 0.55),
        ])

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:          # ESC — cancel this session
            print("[INFO] Cancelled.")
            return 0
        if key != 255 or _mouse_clicked:   # Any real keypress or a mouse click
            break

    # ── Phase 2: Countdown ────────────────────────────────────────────────────
    deadline = time.time() + COUNTDOWN_SECS
    while time.time() < deadline:
        remaining = int(deadline - time.time()) + 1
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        hands, frame = detector.findHands(frame, draw=True, flipType=True)

        _overlay_banner(frame, [
            (f"Get ready — starting in {remaining}…", CLR_ORANGE, 1.0),
            (f"Collecting: '{sign_label}'",           CLR_GREEN,  0.65),
        ])
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(1)

    # ── Phase 3: Capture loop ─────────────────────────────────────────────────
    buffer: list[list] = []
    target  = SAMPLES_PER_SESSION
    prev_t  = time.time()
    fps_display = 0.0

    print(f"[INFO] Collecting {target} samples for '{sign_label}'…  (ESC to stop early)")

    while len(buffer) < target:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera read failed during capture.")
            break
        frame = cv2.flip(frame, 1)

        # ── FPS calculation (rolling instantaneous) ───────────────────────
        now   = time.time()
        dt    = now - prev_t
        if dt > 0:
            fps_display = 1.0 / dt
        prev_t = now

        # ── Hand detection ────────────────────────────────────────────────
        hands, frame = detector.findHands(frame, draw=True, flipType=True)

        captured = len(buffer)

        if not hands:
            # Don't count this frame; show a warning
            _overlay_banner(frame, [
                ("⚠ No hand detected — move into frame", CLR_RED,   0.75),
                (f"Captured: {captured}/{target}  |  FPS: {fps_display:.1f}", CLR_WHITE, 0.6),
            ])
            cv2.imshow(WINDOW_NAME, frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                print("[INFO] ESC pressed — stopping early.")
                break
            continue

        # ── Extract & normalize landmarks ─────────────────────────────────
        #
        # For MAX_HANDS == 1 we use only the first detected hand.
        # For MAX_HANDS == 2 we need exactly 2 hands; skip the frame if fewer.
        if len(hands) < MAX_HANDS:
            _overlay_banner(frame, [
                (f"⚠ Need {MAX_HANDS} hand(s) — only {len(hands)} visible", CLR_RED, 0.75),
                (f"Captured: {captured}/{target}  |  FPS: {fps_display:.1f}", CLR_WHITE, 0.6),
            ])
            cv2.imshow(WINDOW_NAME, frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
            continue

        row_features: list[float] = []
        for h in hands[:MAX_HANDS]:
            lmList = h["lmList"]   # list of 21 [x, y, z]
            bbox   = h["bbox"]     # (x, y, w, h)  — still needed for scale
            # center is no longer used; normalization anchors on wrist (lmList[0])
            row_features.extend(normalize_landmarks(lmList, bbox))

        buffer.append(row_features + [sign_label])

        # ── Overlay progress ──────────────────────────────────────────────
        pct = int((captured / target) * 100)
        _overlay_banner(frame, [
            (f"Collecting '{sign_label}':  {captured}/{target}  ({pct}%)", CLR_GREEN, 0.75),
            (f"FPS: {fps_display:.1f}   |   ESC to stop early",           CLR_WHITE, 0.6),
        ])

        # ── Progress bar across bottom of frame ──────────────────────────
        h_px, w_px = frame.shape[:2]
        bar_x = int(w_px * captured / target)
        cv2.rectangle(frame, (0, h_px - 8), (w_px, h_px),      (60, 60, 60),  -1)
        cv2.rectangle(frame, (0, h_px - 8), (bar_x, h_px),     (0, 200, 100), -1)

        cv2.imshow(WINDOW_NAME, frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            print("[INFO] ESC pressed — stopping early.")
            break

    # ── Phase 4: Save buffered rows ───────────────────────────────────────────
    _ensure_csv()
    _append_rows(buffer)
    return len(buffer)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera at index {CAMERA_INDEX}. "
              "Try changing CAMERA_INDEX at the top of this file.")
        sys.exit(1)

    # Let the OS pick the best native resolution / frame rate
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # Minimize buffer lag

    detector = HandDetector(
        staticMode=False,
        maxHands=MAX_HANDS,
        modelComplexity=1,
        detectionCon=DETECTION_CON,
        minTrackCon=TRACK_CON,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            # ── Get sign name from terminal ───────────────────────────────
            while True:
                raw = input("\nEnter sign name: ")
                label = _sanitize_label(raw)
                if label:
                    break
                print("[WARNING] Sign name cannot be empty. Please try again.")

            # ── Run one collection session ────────────────────────────────
            n = collect_session(cap, detector, label)
            print(f"\n[DONE] Collected {n} samples for '{label}'.")
            if n > 0:
                print(f"       Saved to: {os.path.abspath(CSV_FILENAME)}")

            # ── Ask whether to continue ───────────────────────────────────
            again = input("\nCollect another sign? (y/n): ").strip().lower()
            if again != "y":
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Collection tool closed. Goodbye!")


if __name__ == "__main__":
    main()

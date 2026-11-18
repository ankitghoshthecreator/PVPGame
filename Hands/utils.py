"""
utils.py — Shared hand-landmark normalization utility.

Imported by collect_data.py, train_model.py, and any future
real-time inference script. Do NOT redefine the normalization
logic anywhere else — this is the single source of truth.

Normalization strategy
----------------------
- Origin  : Wrist (landmark 0) — anatomically fixed, pose-invariant.
- Scale   : max(bbox_w, bbox_h) in pixels — removes camera distance.
- Z depth : Already wrist-relative in MediaPipe; divided by same scale
            so depth carries weight comparable to x / y.
"""


def normalize_landmarks(lmList: list, bbox: tuple) -> list[float]:
    """
    Normalize 21 MediaPipe hand landmarks so the output is invariant to:
      - Hand position in the frame  (translation to wrist position)
      - Hand distance from camera   (scale by max bounding-box dimension)
      - Absolute wrist depth        (z already wrist-relative in MediaPipe)

    Parameters
    ----------
    lmList : list of [x, y, z]
        Raw pixel-space landmarks from cvzone's hand dict["lmList"].
        Must contain exactly 21 entries, each with at minimum [x, y, z].
    bbox : tuple (x, y, w, h)
        Bounding box from cvzone's hand dict["bbox"].
        x, y = top-left corner of the box; w, h = width and height in pixels.

    Returns
    -------
    list[float]
        63 flat values in the fixed order:
          x0_norm, y0_norm, z0_norm,   # wrist  (always 0.0, 0.0, 0.0)
          x1_norm, y1_norm, z1_norm,   # thumb CMC
          ...
          x20_norm, y20_norm, z20_norm # pinky tip
        Suitable for direct use as a single feature row in the CSV / model.

    Normalization steps
    -------------------
    1. Translate: subtract wrist (landmark 0) x, y from every landmark so
       the wrist is the origin. The wrist is anatomically fixed — it does NOT
       shift when fingers extend or curl, unlike the bounding-box center.
       (The wrist row itself always becomes [0, 0, 0] after this step.)
    2. Scale: divide by scale = max(bbox_w, bbox_h) in pixels so the result
       is invariant to how close your hand is to the camera.
    3. Z (depth): MediaPipe already returns z relative to the wrist, so we
       only need to divide by the same scale factor to match the x/y range.
    """
    _, _, bbox_w, bbox_h = bbox

    # Guard against a degenerate bounding box (should never happen in practice,
    # but prevents a ZeroDivisionError if MediaPipe returns a tiny detection).
    scale = max(bbox_w, bbox_h)
    if scale == 0:
        scale = 1.0

    # Wrist pixel position — our translation anchor (landmark index 0).
    wrist_x = lmList[0][0]
    wrist_y = lmList[0][1]
    # Note: z is already wrist-relative from MediaPipe; no subtraction needed.

    normalized: list[float] = []
    for lm in lmList:
        x_raw, y_raw, z_raw = lm[0], lm[1], lm[2]

        x_norm = (x_raw - wrist_x) / scale
        y_norm = (y_raw - wrist_y) / scale
        z_norm =  z_raw            / scale   # already relative to wrist

        normalized.extend([x_norm, y_norm, z_norm])

    return normalized

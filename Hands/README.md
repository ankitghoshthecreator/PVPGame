# Sign Language Recognition — Data Collection & Training Pipeline

A lightweight, real-time sign language detector built on MediaPipe hand landmarks
(21 × (x, y, z) = 63 features) rather than raw images, so inference is fast enough
to run on CPU or a modest GPU (RTX 3050 / 4 GB VRAM tested).

---

## File Overview

| File | Purpose |
|---|---|
| `utils.py` | Shared landmark normalization function — imported by everything |
| `collect_data.py` | Real-time webcam data collection tool |
| `train_model.py` | Trains & evaluates classifiers; saves best model |
| `requirements.txt` | Python dependencies |

---

## Prerequisites

```powershell
# Activate the virtual environment (from d:\Hands)
.\venv\Scripts\Activate.ps1

# Install dependencies (already done if you followed the setup steps)
pip install -r requirements.txt
```

---

## Step 1 — Collect Training Data

```powershell
python collect_data.py
```

**What happens:**
1. Terminal asks `Enter sign name:` — type the sign label (e.g. `hello`, `yes`, `no`).
2. A live webcam window opens showing your hand with MediaPipe landmarks drawn.
3. Press **any key** or **click the window** to begin the countdown (3 s).
4. The tool automatically captures **200 frames** where a hand is detected.  
   Frames without a visible hand are skipped — only clean detections are saved.
5. A progress bar and live FPS counter are shown on-screen.
6. Press **ESC** at any point to stop early — whatever was captured is still saved.
7. After the session, the terminal asks `Collect another sign? (y/n)`.

**Data quality tips (important for accuracy):**
- Collect **at least 300–500 samples per sign**, ideally across multiple sessions.
- Vary your **hand angle** — tilt it left, right, rotate slightly.
- Vary **distance from camera** — the normalization handles scale, but more variety helps.
- Vary **lighting** — bright, dim, backlit.
- Collect with **both hands** if your signs are symmetric (change `MAX_HANDS = 2` in `collect_data.py`).

**Output:** `data/sign_data.csv`  
New sessions **append** to the file — you will never lose previously collected data.

**CSV format:**
```
x0,y0,z0,x1,y1,z1,...,x20,y20,z20,label
0.12,-0.34,0.01,...,hello
```
63 normalized landmark values + 1 label column.

---

## Step 2 — Train the Model

> **Only run this after collecting enough data.**

```powershell
python train_model.py
```

**What happens:**
1. Loads `data/sign_data.csv`.
2. Encodes class labels with a `LabelEncoder`.
3. Stratified 80/20 train/test split.
4. Trains two models:
   - `RandomForestClassifier` (300 trees) — robust, interpretable, fast
   - `MLPClassifier` (256→128→64 hidden units) — often higher accuracy on clean features
5. Prints accuracy, weighted F1, and a full confusion matrix for each.
6. Saves the **better** model and the **label encoder** to `models/`.

**Output files:**
```
models/
  sign_model.joblib       ← trained classifier
  label_encoder.joblib    ← fitted LabelEncoder (required at inference time)
```

---

## Step 3 — Inference (coming soon)

Your inference script will:
1. Open the webcam.
2. Detect a hand with cvzone.
3. Call `normalize_landmarks(lmList, bbox, center)` from `utils.py`.
4. Pass the 63-value vector to the loaded `sign_model.joblib`.
5. Decode the prediction with `label_encoder.joblib`.

Because both collection and inference use the **exact same** `normalize_landmarks()`
function from `utils.py`, there is zero train/serve skew.

---

## Normalization Summary

Each hand's 21 landmarks are normalized to be invariant to position and scale:

| Step | Operation |
|---|---|
| Translate | Subtract bounding-box center (cx, cy) from every x, y |
| Scale | Divide by `max(bbox_width, bbox_height)` |
| Z (depth) | Subtract wrist's raw z, then divide by the same scale |

Result: 63 floats in the range roughly `[−0.5, 0.5]`, centred at `(0, 0, 0)`.

---

## Configurable Constants

Both `collect_data.py` and `train_model.py` have a clearly marked
**CONFIGURABLE CONSTANTS** block at the top. The most common ones:

| Constant | File | Default | Meaning |
|---|---|---|---|
| `CAMERA_INDEX` | collect_data.py | `0` | Webcam device index |
| `SAMPLES_PER_SESSION` | collect_data.py | `200` | Frames captured per session |
| `MAX_HANDS` | collect_data.py | `1` | 1 or 2 hands (doubles feature vector to 126) |
| `TEST_SIZE` | train_model.py | `0.20` | Held-out evaluation fraction |
| `RF_N_ESTIMATORS` | train_model.py | `300` | RandomForest trees |
| `MLP_HIDDEN_LAYERS` | train_model.py | `(256, 128, 64)` | MLP layer sizes |

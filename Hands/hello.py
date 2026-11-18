
import asyncio
import json
import os
import sys

import cv2
import joblib
import numpy as np
import websockets
from cvzone.HandTrackingModule import HandDetector

from utils import normalize_landmarks


# ============================================================
# CONFIGURATION
# ============================================================

# False = Run webcam and gesture detection without showing a window.
# True  = Show webcam preview for debugging.
SHOW_CAMERA = False

SERVER_HOST = "localhost"
SERVER_PORT = 8765


# ============================================================
# PATHS
# ============================================================

def resource_path(relative_path):
    """
    Resolve paths for development and PyInstaller execution.
    """

    if getattr(sys, "frozen", False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)


MODEL_PATH = resource_path("models/sign_model.joblib")
ENCODER_PATH = resource_path("models/label_encoder.joblib")


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

print("[INFO] Classifier loaded")
print("[INFO] Classes:", list(label_encoder.classes_))


# ============================================================
# WEBSOCKET CLIENTS
# ============================================================

connected_clients = set()


async def websocket_handler(websocket):

    connected_clients.add(websocket)

    print("[INFO] Unity connected")

    try:
        await websocket.wait_closed()

    finally:
        connected_clients.discard(websocket)
        print("[INFO] Unity disconnected")


async def send_prediction(prediction):

    if not connected_clients:
        return

    message = json.dumps({
        "sign": prediction
    })

    await asyncio.gather(
        *[
            client.send(message)
            for client in list(connected_clients)
        ],
        return_exceptions=True
    )


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("[INFO] Webcam opened")


# ============================================================
# MEDIAPIPE HAND DETECTOR
# ============================================================

detector = HandDetector(
    staticMode=False,
    maxHands=1,
    modelComplexity=1,
    detectionCon=0.7,
    minTrackCon=0.5,
)


# ============================================================
# PREDICTION LOOP
# ============================================================

async def prediction_loop():

    print("[INFO] Gesture prediction loop started")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("[WARNING] Could not read webcam frame")
            await asyncio.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)

        hands, frame = detector.findHands(
            frame,
            draw=SHOW_CAMERA,
            flipType=True
        )

        prediction_text = "No hand detected"

        if hands:

            hand = hands[0]

            lmList = hand["lmList"]
            bbox = hand["bbox"]

            # Feature extraction
            features = normalize_landmarks(
                lmList,
                bbox
            )

            X = np.array(
                features,
                dtype=np.float32
            ).reshape(1, -1)

            # Model prediction
            prediction = model.predict(X)[0]

            predicted_label = label_encoder.inverse_transform(
                [prediction]
            )[0]

            predicted_label = str(predicted_label).upper()

            prediction_text = f"Sign: {predicted_label}"

            print("[PREDICTION]", predicted_label)

            # Send prediction to Unity
            await send_prediction(predicted_label)

        # ====================================================
        # OPTIONAL CAMERA PREVIEW
        # ====================================================

        if SHOW_CAMERA:

            cv2.putText(
                frame,
                prediction_text,
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3
            )

            cv2.imshow(
                "Sign Language Prediction",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                print("[INFO] Exit requested")
                break

        # Allow asyncio to process WebSocket events
        await asyncio.sleep(0)


# ============================================================
# MAIN
# ============================================================

async def main():

    print("[INFO] Starting WebSocket server...")
    print(f"[INFO] Server: ws://{SERVER_HOST}:{SERVER_PORT}")

    async with websockets.serve(
        websocket_handler,
        SERVER_HOST,
        SERVER_PORT
    ):

        print("[INFO] WebSocket server running")
        print("[INFO] Waiting for Unity connection...")

        await prediction_loop()


# ============================================================
# CLEANUP
# ============================================================

try:

    asyncio.run(main())

except KeyboardInterrupt:

    print("[INFO] Shutdown requested")

finally:

    cap.release()
    cv2.destroyAllWindows()

    print("[INFO] Camera closed")

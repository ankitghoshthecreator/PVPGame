# SignStrike

SignStrike is a 2D side-scrolling action game built with Unity. It combines keyboard controls with a camera-based hand-sign recognition engine: detected signs can move the player, make the player jump, and fire at enemies. The project also includes Python tools for collecting and training sign-classification data.

## Features

- 2D platforming with player movement, jumping, and camera follow.
- Projectile combat against enemies, including enemy AI, health, and respawning.
- Optional gesture controls using a webcam and a local WebSocket connection.
- A Unity menu flow that starts the gesture engine and waits for it to connect before entering gameplay.
- Python scripts for collecting normalized hand-landmark samples and training a classifier.
- A Windows build and a packaged Windows gesture-engine executable are included in the repository.

## Project layout

| Path | Contents |
| --- | --- |
| `Assets/` | Unity scenes, scripts, settings, prefabs, audio, and art assets. |
| `Assets/Scenes/SampleScene.unity` | Scene enabled in the current Unity build settings. |
| `Assets/SunnyLand Artwork/Sprites/player/run/` | Gameplay scripts for player, enemies, UI, menu, and gesture integration. |
| `Hands/` | Python source for hand tracking, sample collection, and model training. |
| `Hands/data/sign_data.csv` | Collected labeled landmark samples used by the training script. |
| `Hands/models/` | Trained classifier and label encoder used by the Python tooling. |
| `GestureEngine/` | Windows gesture-engine executable and its runtime/model files. |
| `game/gamerelease/` | Packaged Windows game build. |
| `Packages/manifest.json` | Unity package dependencies. |
| `ProjectSettings/` | Unity project and editor configuration. |

Unity generated folders such as `Library/`, `Temp/`, and `Logs/` are ignored by Git. Unity recreates them when opening the project.

## Requirements

### To open and build the Unity project

- Unity **6000.5.8f1** (see `ProjectSettings/ProjectVersion.txt`).
- A Windows environment to run the bundled gesture engine or the packaged game.
- A webcam for gesture control.

The Unity package manifest includes Universal Render Pipeline, Input System, 2D Animation, 2D Tilemap, TextMesh Pro UI support, and NativeWebSocket.

### To run the Python collection and training scripts

- Python 3.9 or newer is recommended.
- A working webcam and the relevant camera permissions.
- Python packages listed in `Hands/requirements.txt` (`opencv-python`, `cvzone`, `mediapipe`, `pandas`, `scikit-learn`, `numpy`, and `joblib`).

## Getting started

### Open in Unity

1. Clone this repository and open its root folder in Unity Hub.
2. Use Unity 6000.5.8f1 and let Unity import the assets and resolve packages.
3. Open `Assets/Scenes/SampleScene.unity` and press **Play**.
4. Use the in-game menu to start gameplay. When the scene's gesture launcher is configured, the menu starts the local gesture engine and attempts to connect to its WebSocket server.

### Run the packaged Windows game

Run `game/gamerelease/My project.exe` on Windows. Gesture input requires a working camera and the matching gesture engine available at the configured path. The Unity launcher currently looks for `GestureEngine/GestureEngine.exe` relative to the game/project root and connects to `ws://localhost:8765`.

### Use keyboard controls

| Input | Action |
| --- | --- |
| **A** | Move left |
| **D** | Move right |
| **Space** | Shoot |

Jumping is exposed through the player's controller and gesture command handlers. The exact gesture labels are defined by the gesture engine/model and should match the labels it sends to the Unity WebSocket client.

## Gesture recognition pipeline

The Unity client launches `GestureEngine/GestureEngine.exe` as a child process and connects to `ws://localhost:8765`. The `SignWebSocket` component receives JSON messages containing a `sign` field and maps recognized labels to player commands. The gesture executable uses the trained model files packaged beside it. A webcam must be available to the process, and the default port must be free.

The `Hands/` directory contains the data preparation workflow used to build classifiers:

1. Install dependencies in a Python environment:

   ```powershell
   cd Hands
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

2. Collect labeled samples from the webcam:

   ```powershell
   python collect_data.py
   ```

   Enter a sign label when prompted. The collector captures up to 200 valid frames per session by default and appends rows to `Hands/data/sign_data.csv`. Change `CAMERA_INDEX`, `SAMPLES_PER_SESSION`, or `MAX_HANDS` near the top of `collect_data.py` to suit your setup. Collect examples for each label under varied camera positions and lighting.

3. Train and evaluate the classifier:

   ```powershell
   python train_model.py
   ```

   The script reads `data/sign_data.csv`, evaluates candidate classifiers, and writes `models/sign_model.joblib` and `models/label_encoder.joblib`. Training requires enough examples for every label to support the configured stratified split and cross-validation.

`Hands/hand_tracking.py` is a standalone webcam hand-landmark preview. Run it with `python hand_tracking.py`; press **Q** to close the preview. The Unity game uses the packaged `GestureEngine` executable rather than launching this preview or the training scripts.

## Build notes

The active build scene is `Assets/Scenes/SampleScene.unity`. In Unity, use **File → Build Profiles** (or **Build Settings**, depending on the editor UI) to choose a target platform and create a new build. For gesture support in a Windows build, include the `GestureEngine/` runtime files at the relative location expected by `GesturePythonLauncher` and confirm the server and Unity client use the same port and message format.

## Troubleshooting

- **Unity reports a missing package or a long import:** open the project with Unity 6000.5.8f1 and allow Package Manager to finish resolving dependencies.
- **Gesture engine does not start:** confirm `GestureEngine/GestureEngine.exe` exists relative to the project/game root and can run on the machine.
- **Unity cannot connect:** confirm the engine started successfully, port `8765` is available, and the client URL is `ws://localhost:8765`.
- **No hand is detected:** check camera permissions, close other apps using the camera, and verify the configured camera index.
- **Training fails due to missing data:** collect examples first and ensure every sign has sufficient samples for the stratified evaluation.

## Repository notes

This repository contains third-party Unity art and package assets alongside project code. Preserve any included asset attribution and license terms when redistributing or publishing builds. The packaged Windows executables are platform-specific; use the Unity project to produce builds for other platforms.

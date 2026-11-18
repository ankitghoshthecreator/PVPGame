import cv2
from cvzone.HandTrackingModule import HandDetector

# Initialize the webcam (0 is usually the default built-in camera)
cap = cv2.VideoCapture(0)

# Initialize the HandDetector class
# MediaPipe's hand tracking model tracks 21 3D landmarks per hand.
detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

print("Starting camera... Press 'q' to quit.")

while True:
    # Read each frame from the webcam
    success, img = cap.read()
    if not success:
        print("Failed to grab frame from camera.")
        break

    # Find hands in the current frame
    # This returns the detected hands and the image with drawn landmarks
    hands, img = detector.findHands(img, draw=True, flipType=True)

    # Check if any hands are detected
    if hands:
        for hand in hands:
            # List of 21 landmark points
            lmList = hand["lmList"]
            # Hand Type: "Left" or "Right"
            handType = hand["type"]

            print(f"--- {handType} Hand Detected ---")
            
            # Print all 21 points for the hand
            # Note: MediaPipe Hands tracks 21 points (0 to 20).
            for i, lm in enumerate(lmList):
                # Each landmark contains [x, y, z] coordinates
                # x, y are pixel coordinates, z is depth
                if len(lm) == 3:
                    x, y, z = lm
                    print(f"  Point {i:02}: (x: {x:4}, y: {y:4}, z: {z:4})")
                else:
                    x, y = lm
                    print(f"  Point {i:02}: (x: {x:4}, y: {y:4})")
            
            print("=" * 35)

    # Display the image in a window
    cv2.imshow("Hand Tracking", img)

    # Wait 1 ms and check if 'q' is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()

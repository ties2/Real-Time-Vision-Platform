import time

import cv2
import requests

# URL of your running FastAPI server
API_URL = "http://127.0.0.1:8000/predict"

# Open the webcam (0 is usually the default laptop camera)
# To use a video file instead, replace 0 with 'my_video.mp4'
cap = cv2.VideoCapture(0)

print("Starting real-time detection... Press 'q' to quit.")

while True:
    start_time = time.time()

    # 1. Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # 2. Encode the frame as a JPEG image in memory
    _, img_encoded = cv2.imencode(".jpg", frame)

    # 3. Send the image to the FastAPI server
    # We send the raw bytes of the image
    files = {"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}

    try:
        response = requests.post(API_URL, files=files)
        data = response.json()

        # 4. Draw the results on the frame
        if "detections" in data:
            for det in data["detections"]:
                box = det["bounding_box"]
                label = det["class"]
                conf = det["confidence"]

                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

                # Draw Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw Label
                cv2.putText(
                    frame,
                    f"{label} {conf}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

    except Exception as e:
        print("Error connecting to API:", e)

    # Calculate and display FPS (Frames Per Second)
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 5. Show the video stream with detections
    cv2.imshow("Real-Time YOLO API Client", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

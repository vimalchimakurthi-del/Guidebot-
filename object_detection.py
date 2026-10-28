
import cv2
import numpy as np

# ---------------- CONFIGURATION ----------------

MODEL_PATH = "yolov8n.onnx"
STREAM_URL = "http://192.168.29.172:81/stream"

CONFIDENCE_THRESHOLD = 0.50
NMS_THRESHOLD = 0.45
INPUT_SIZE = 640

WINDOW_NAME = "GuideBot AI - Object Detection"

# Standard YOLOv8 COCO class labels
CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

# ---------------- LOAD MODEL ----------------

print("Loading YOLOv8 AI model...")

try:
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
except Exception as error:
    print("Could not load the AI model.")
    print("Check that yolov8n.onnx is in the GuideBot folder.")
    print(error)
    raise SystemExit

print("AI model loaded successfully.")

# ---------------- CONNECT CAMERA ----------------

print("Connecting to ESP32-CAM...")

camera = cv2.VideoCapture(STREAM_URL)

if not camera.isOpened():
    print("Cannot connect to ESP32-CAM.")
    print("Check the camera IP address and Wi-Fi.")
    camera.release()
    raise SystemExit

print("Camera connected!")
print("GuideBot object detection started.")
print("Press Q to quit.")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)

# ---------------- MAIN LOOP ----------------

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not receive camera frame.")
            break

        height, width = frame.shape[:2]

        # Preserve the original aspect ratio
        scale = min(
            INPUT_SIZE / width,
            INPUT_SIZE / height
        )

        new_width = int(round(width * scale))
        new_height = int(round(height * scale))

        resized = cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR
        )

        # Add padding instead of stretching the image
        pad_x = (INPUT_SIZE - new_width) // 2
        pad_y = (INPUT_SIZE - new_height) // 2

        image = np.full(
            (INPUT_SIZE, INPUT_SIZE, 3),
            114,
            dtype=np.uint8
        )

        image[
            pad_y:pad_y + new_height,
            pad_x:pad_x + new_width
        ] = resized

        # Prepare input for YOLO
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1 / 255.0,
            size=(INPUT_SIZE, INPUT_SIZE),
            swapRB=True,
            crop=False
        )

        net.setInput(blob)
        output = net.forward()

        # Expected YOLOv8 output:
        # (1, 84, 8400) or (84, 8400)
        if output.ndim == 3:
            output = output[0]

        # Convert to one detection per row
        if output.shape[0] < output.shape[1]:
            output = output.T

        boxes = []
        scores = []
        class_ids = []

        # ---------------- DETECT OBJECTS ----------------

        for row in output:
            if len(row) < 5:
                continue

            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])

            # Filter low-confidence detections
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            if class_id >= len(CLASSES):
                continue

            cx, cy, box_w, box_h = map(float, row[:4])

            # Convert from model coordinates
            # back to original camera coordinates
            x1 = int(round(
                (cx - box_w / 2 - pad_x) / scale
            ))
            y1 = int(round(
                (cy - box_h / 2 - pad_y) / scale
            ))
            x2 = int(round(
                (cx + box_w / 2 - pad_x) / scale
            ))
            y2 = int(round(
                (cy + box_h / 2 - pad_y) / scale
            ))

            # Keep boxes within the camera image
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))

            box_width = x2 - x1
            box_height = y2 - y1

            # Ignore tiny detections
            if box_width < 20 or box_height < 20:
                continue

            boxes.append([
                x1, y1, box_width, box_height
            ])
            scores.append(confidence)
            class_ids.append(class_id)

        # ---------------- REMOVE DUPLICATES ----------------

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            CONFIDENCE_THRESHOLD,
            NMS_THRESHOLD
        )

        detected_names = []

        if len(indices) > 0:
            for i in np.array(indices).flatten():
                x, y, box_w, box_h = boxes[i]

                name = CLASSES[class_ids[i]]
                confidence = scores[i]

                detected_names.append(name)

                label = f"{name}: {confidence:.0%}"

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + box_w, y + box_h),
                    (0, 255, 0),
                    2
                )

                # Draw label background
                text_y = max(y - 8, 22)

                cv2.putText(
                    frame,
                    label,
                    (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

        # ---------------- DISPLAY ----------------

        # Show the current unique object names
        unique_names = sorted(set(detected_names))

        if unique_names:
            status = "Objects: " + ", ".join(unique_names)
        else:
            status = "Objects: none detected"

        cv2.putText(
            frame,
            status[:100],
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Enlarge the display without stretching its ratio
        display_frame = cv2.resize(
            frame,
            None,
            fx=1.5,
            fy=1.5,
            interpolation=cv2.INTER_LINEAR
        )

        cv2.imshow(WINDOW_NAME, display_frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("GuideBot camera stopped.")

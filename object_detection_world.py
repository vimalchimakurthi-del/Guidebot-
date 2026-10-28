
import time
import cv2
from ultralytics import YOLOWorld

# ESP32-CAM stream
STREAM_URL = "http://192.168.29.172:81/stream"
MODEL_PATH = "yolov8s-worldv2.pt"

# Detect objects every 2 frames to reduce lag
DETECT_EVERY_N_FRAMES = 2
IMAGE_SIZE = 320


CLASSES = [
    # People and navigation
    "person",
    "chair",
    "table",
    "desk",
    "door",
    "stairs",
    "window",
    "ceiling fan",
    "lamp",
    "wall clock",

    # Everyday objects
    "bottle",
    "water jug",
    "cup",
    "mug",
    "plate",
    "bowl",
    "spoon",
    "fork",
    "book",
    "backpack",
    "handbag",
    "suitcase",
    "wallet",
    "keys",
    "glasses",
    "umbrella",
    "shoes",
    "helmet",
    "cardboard box",
    "trash bin",

    # Electronics
    "laptop",
    "mobile phone",
    "keyboard",
    "computer mouse",
    "computer monitor",
    "television",
    "remote control",
    "headphones",
    "charger",
    "printer",
    "speaker",

    # Furniture and appliances
    "sofa",
    "bed",
    "pillow",
    "blanket",
    "refrigerator",
    "microwave oven",
    "washing machine",
    "potted plant",

    # Safety and other objects
    "fire extinguisher",
    "car",
    "bicycle",
    "dog",
    "cat",
    "toy"
]


def connect_camera():
    camera = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


print("Loading YOLO-World model...")
model = YOLOWorld(MODEL_PATH)
model.set_classes(CLASSES)

print("Using NVIDIA GPU for detection.")

print("Connecting to ESP32-CAM...")
camera = connect_camera()

if not camera.isOpened():
    print("Camera connection failed.")
    print("Check the ESP32-CAM IP and Wi-Fi.")
    camera.release()
    raise SystemExit

window = "GuideBot AI"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)

print("GuideBot is running. Press Q to quit.")

frame_count = 0
cached_detections = []

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Camera frame lost. Reconnecting...")
            camera.release()
            time.sleep(1)

            camera = connect_camera()

            if not camera.isOpened():
                print("Reconnect failed. Retrying...")
                camera.release()
                time.sleep(2)

            frame_count = 0
            cached_detections = []
            continue

        frame_count += 1

        # Run AI detection on alternate frames
        if (
            frame_count == 1
            or frame_count % DETECT_EVERY_N_FRAMES == 0
        ):
            results = model.predict(
                source=frame,
                imgsz=IMAGE_SIZE,
                conf=0.30,
                iou=0.45,
                device=0,
                verbose=False
            )

            result = results[0]
            cached_detections = []

            # Save detected boxes for the next frame
            for box in result.boxes:
                x1, y1, x2, y2 = (
                    box.xyxy[0].cpu().tolist()
                )

                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                label = result.names[class_id]

                cached_detections.append(
                    (
                        int(x1), int(y1),
                        int(x2), int(y2),
                        label, confidence
                    )
                )

        # Draw the latest detections on the current frame
        output = frame.copy()

        for x1, y1, x2, y2, label, confidence in cached_detections:
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            text = f"{label} {confidence:.2f}"

            cv2.putText(
                output,
                text,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        # Only shrink oversized frames; never enlarge small ones
        height, width = output.shape[:2]
        scale = min(1.0, 1280 / width, 720 / height)

        if scale < 1.0:
            output = cv2.resize(
                output,
                (
                    int(width * scale),
                    int(height * scale)
                ),
                interpolation=cv2.INTER_AREA
            )

        # Display in one window only
        cv2.imshow(window, output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("GuideBot stopped.")


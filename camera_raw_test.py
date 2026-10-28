
import cv2

url = "http://192.168.29.172:81/stream"
camera = cv2.VideoCapture(url)

if not camera.isOpened():
    print("Could not connect to ESP32-CAM")
    exit()

print("Camera connected. Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Failed to read frame")
        break

    cv2.imshow("Raw Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

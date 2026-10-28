
import cv2

url = "http://192.168.29.172:81/stream"
camera = cv2.VideoCapture(url)

if not camera.isOpened():
    print("Cannot connect to ESP32-CAM.")
    print("Check the camera stream and Wi-Fi.")
    raise SystemExit

print("GuideBot camera connected!")

while True:
    success, frame = camera.read()

    if not success:
        print("Failed to receive video.")
        break

    cv2.imshow("GuideBot AI - Live Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

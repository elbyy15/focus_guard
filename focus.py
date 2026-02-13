import cv2
from ultralytics import YOLO
import pygame
import time

pygame.mixer.init()
pygame.mixer.music.load("warning.mp3")

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

last_alert_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    results = model(frame)
    annotated_frame = results[0].plot()

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "cell phone":
                current_time = time.time()

                if current_time - last_alert_time > 5:
                    print("PHONE DETECTED! STAY FOCUSED 😡")
                    pygame.mixer.music.play()
                    last_alert_time = current_time

    cv2.imshow("Focus Guard", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

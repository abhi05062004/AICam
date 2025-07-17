import cv2
from ultralytics import YOLO

model = YOLO('yolo11s.pt')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('webcam_output.mp4', fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, verbose=False)
    annotated_frame = results[0].plot()

    cv2.imshow('YOLOv8 Webcam', annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

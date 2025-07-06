import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model (you can change to 'yolov8n', 'yolov8s', 'yolov8m', etc.)
model = YOLO("yolov8n.pt")
model.to("cpu")      # Small and fast for real-time

# Initialize Deep SORT
tracker = DeepSort(max_age=30)

# Initialize video capture (0 for webcam or replace with 'video.mp4')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO object detection
    results = model(frame)[0]

    detections = []

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        class_name = model.names[int(class_id)]

        # Format: [x1, y1, x2, y2, confidence, class_id]
        detections.append(([x1, y1, x2 - x1, y2 - y1], score, class_name))

    # Deep SORT tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw tracking info
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        class_name = track.get_det_class()

        # Draw box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{class_name} ID:{track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Show frame
    cv2.imshow("Object Detection and Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

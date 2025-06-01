from ultralytics import YOLO
import cv2

# Load model YOLOv5 (pastikan yolov5s.pt sudah ada)
model = YOLO('yolov5s.pt')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Kamera tidak bisa dibuka")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Tidak dapat membaca frame dari kamera")
        break

    results = model(frame)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            if label == 'bottle' and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Deteksi Botol (Webcam)', frame)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):  # ESC atau q untuk keluar
        break

cap.release()
cv2.destroyAllWindows()

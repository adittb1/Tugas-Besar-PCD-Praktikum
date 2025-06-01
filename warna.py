import cv2
import numpy as np

# Fungsi untuk menggabungkan banyak mask
def combine_masks(hsv, color_dict):
    combined_result = np.zeros_like(hsv)
    all_masks = None

    for color_name, ranges in color_dict.items():
        mask = None
        for lower, upper in ranges:
            current_mask = cv2.inRange(hsv, lower, upper)
            if mask is None:
                mask = current_mask
            else:
                mask = cv2.bitwise_or(mask, current_mask)
        
        color_output = cv2.bitwise_and(frame, frame, mask=mask)
        combined_result = cv2.bitwise_or(combined_result, color_output)

        if all_masks is None:
            all_masks = mask
        else:
            all_masks = cv2.bitwise_or(all_masks, mask)

    return combined_result, all_masks

# Rentang HSV untuk berbagai warna
color_ranges = {
    "Merah": [
        (np.array([0, 120, 70]), np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([180, 255, 255]))
    ],
    "Hijau": [
        (np.array([35, 50, 50]), np.array([85, 255, 255]))
    ],
    "Biru": [
        (np.array([90, 100, 100]), np.array([130, 255, 255]))
    ],
    "Kuning": [
        (np.array([20, 100, 100]), np.array([30, 255, 255]))
    ],
    "Oranye": [
        (np.array([10, 100, 20]), np.array([20, 255, 255]))
    ],
    "Ungu": [
        (np.array([130, 50, 50]), np.array([160, 255, 255]))
    ],
    "Cokelat": [
        (np.array([10, 100, 20]), np.array([20, 200, 150]))
    ],
    "Pink": [
        (np.array([145, 100, 100]), np.array([170, 255, 255]))
    ],
    "Putih": [
        (np.array([0, 0, 200]), np.array([180, 30, 255]))
    ],
    "Hitam": [
        (np.array([0, 0, 0]), np.array([180, 255, 50]))
    ]
}

# Mulai kamera
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    result, all_masks = combine_masks(hsv, color_ranges)

    # Tampilkan hasil
    cv2.imshow("Asli", frame)
    cv2.imshow("Mask Semua Warna", all_masks)
    cv2.imshow("Hasil Deteksi Warna", result)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cam.release()
cv2.destroyAllWindows()

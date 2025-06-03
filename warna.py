import cv2
import numpy as np

# Rentang HSV untuk warna kuning (kulit pisang)
yellow_range = [
    (np.array([20, 100, 100]), np.array([30, 255, 255]))
]

# Baca gambar dari file
image_path = "pisang2.jpg"  # Ganti path dengan gambar Anda
frame = cv2.imread(image_path)
if frame is None:
    print("Gambar tidak ditemukan.")
    exit()

# Konversi ke HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Buat mask untuk warna kuning
mask = None
for lower, upper in yellow_range:
    current_mask = cv2.inRange(hsv, lower, upper)
    if mask is None:
        mask = current_mask
    else:
        mask = cv2.bitwise_or(mask, current_mask)

# Hitung jumlah piksel kuning
yellow_pixels = cv2.countNonZero(mask)

# Threshold untuk menentukan apakah ada pisang
banana_detected = yellow_pixels > 500  # Ubah threshold sesuai kebutuhan

# Tampilkan hasil
result = cv2.bitwise_and(frame, frame, mask=mask)

# Tampilkan info pada gambar
display_frame = frame.copy()
message = "Banana Detected!" if banana_detected else "No Banana Detected"
color = (0, 255, 0) if banana_detected else (0, 0, 255)
cv2.putText(display_frame, message, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

# Resize untuk tampilan
def resize_for_display(img, scale=0.5):
    return cv2.resize(img, (0, 0), fx=scale, fy=scale)

# Tampilkan jendela
cv2.imshow("Gambar Asli + Info", resize_for_display(display_frame))
cv2.imshow("Mask Kuning", resize_for_display(mask))
cv2.imshow("Deteksi Kuning", resize_for_display(result))
cv2.waitKey(0)
cv2.destroyAllWindows()

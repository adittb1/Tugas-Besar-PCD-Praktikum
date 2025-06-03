import cv2
import numpy as np
import os

# Folder yang berisi dataset gambar
folder_path = "dataset_banana_skin"  # Ganti dengan path folder Anda
output_file = "banana_detection_results.txt"

# Rentang HSV untuk warna kuning (kulit pisang)
yellow_range = [
    (np.array([20, 100, 100]), np.array([30, 255, 255]))
]

# Ambil semua nama file gambar dalam folder
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

# Simpan hasil deteksi
results = []

# Proses semua gambar
for filename in image_files:
    image_path = os.path.join(folder_path, filename)
    frame = cv2.imread(image_path)
    if frame is None:
        results.append((filename, "Gagal Dibaca"))
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Gabungkan semua mask
    mask = sum([cv2.inRange(hsv, lower, upper) for lower, upper in yellow_range])
    yellow_pixels = cv2.countNonZero(mask)

    # Threshold deteksi pisang
    banana_detected = yellow_pixels > 500

    result_text = "Banana Detected" if banana_detected else "No Banana"
    results.append((filename, result_text))

# Tampilkan dan simpan hasil
with open(output_file, "w") as f:
    for filename, result_text in results:
        print(f"{filename}: {result_text}")
        f.write(f"{filename}: {result_text}\n")

print(f"\nHasil deteksi disimpan di: {output_file}")

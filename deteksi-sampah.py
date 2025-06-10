import cv2
import numpy as np
import os
from skimage.feature import graycomatrix, graycoprops

def extract_features(image):
    features = {}
    viz_images = {}

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_mean, s_mean, v_mean = np.mean(hsv, axis=(0,1))
    features['hue'] = h_mean
    features['saturation'] = s_mean
    features['value'] = v_mean
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    viz_images['texture'] = gray
    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    viz_images['threshold'] = thresh 
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    features['area'] = area
    features['circularity'] = 4 * np.pi * area / (perimeter**2) if perimeter != 0 else 0
    
    color_mask = cv2.bitwise_and(image, image, mask=thresh)
    viz_images['color'] = color_mask

    glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    features['contrast'] = graycoprops(glcm, 'contrast')[0, 0]
    
    return features, viz_images

def classify_waste(features):
    circularity = features['circularity']
    contrast = features['contrast']
    saturation = features['saturation']
    value = features['value']

    if circularity >= 0.75:
        return 'Botol'
    elif saturation < 40 and value > 180:
        return 'Kertas'
    elif contrast > 400:
        return 'Plastik'
    return 'Tidak dikenal'

def create_labeled_image(image, label):
    labeled_img = image.copy()
    cv2.rectangle(labeled_img, (0,0), (labeled_img.shape[1], 40), (0,0,0), -1)
    cv2.putText(labeled_img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return labeled_img

if __name__ == "__main__":
    input_folder = 'data'
    output_folder = 'extracted'
    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files = sorted(image_files)[:60] 
    
    print(f"\nProcessing {len(image_files)} images...\n")

    print("===================================")
    print("Filename\tJenis Sampah")
    print("===================================")

    for idx, filename in enumerate(image_files):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"{filename}\tGagal membaca gambar")
            continue

        features, viz_images = extract_features(image)
        
        if features is None:
            print(f"{filename}\tGagal ekstraksi fitur")
            continue

        prediction_label = classify_waste(features)

        print(f"{filename}\t{prediction_label}")
  
        h, w, _ = image.shape
        original_labeled = create_labeled_image(image, '1. Original')
        color_mask_labeled = create_labeled_image(viz_images['color'], '2. Deteksi Warna')

        threshold_img = viz_images['threshold']
        threshold_bgr = cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2BGR)
        threshold_labeled = create_labeled_image(threshold_bgr, '3. Bentuk (Threshold)')

        texture_bgr = cv2.cvtColor(viz_images['texture'], cv2.COLOR_GRAY2BGR)
        texture_labeled = create_labeled_image(texture_bgr, '4. Tekstur')

        final_result_img = image.copy()
        cv2.putText(final_result_img, f'{prediction_label}', (10, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        final_result_labeled = create_labeled_image(final_result_img, '5. Hasil')

        all_images = [original_labeled, color_mask_labeled, threshold_labeled, texture_labeled, final_result_labeled]
        combined_view = np.hstack(all_images)

        max_dim = 1600
        (h_c, w_c) = combined_view.shape[:2]
        if h_c > max_dim or w_c > max_dim:
            scale = max_dim / max(h_c, w_c)
            new_w = int(w_c * scale)
            new_h = int(h_c * scale)
            combined_view = cv2.resize(combined_view, (new_w, new_h), interpolation=cv2.INTER_AREA)

        output_image_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_extracted.jpg")
        cv2.imwrite(output_image_path, combined_view)

    print("\nSelesai!")

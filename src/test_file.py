import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import time
import os
import crown_detector
import preprocessing
import tile_detector
cmax = 270 #Canny max value, old value was 320
blurvalue = 11 #Median blur value, old value was 11
denoisevalue = 8 #Denoise value, old value was 18

base_dir = Path(__file__).resolve().parent
image_dir = base_dir.parent / "Cropped and perspective corrected boards"
full_object_detect = 0
#objects in all 85
start_time = time.time()
image_dir_template = base_dir.parent / "Images"
template = image_dir_template / f"crown.png"
template = cv2.imread(str(template), cv2.IMREAD_GRAYSCALE)
num_images = 74
results = [] 
total_objects = 0
j = 0
target_intensity = 100
for i in range(1, num_images + 1):
    img_path = image_dir / f"{i}.jpg"
    
    
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Failed to load image at {img_path}")
        continue

    gamma_corrected = preprocessing.Preprocessing_tools.gamma_correction(img, preprocessing.Preprocessing_tools.find_gamma_value(img, target_intensity))

    num_labels, crown_position, image_with_crowns = crown_detector.output.output_for_score(gamma_corrected,template)

    total_objects += num_labels
    results.append([num_labels, i])
    start_time = time.time()
print(f"Total crowns detected in all images: {total_objects}")
df = pd.DataFrame(results, columns=["objects", "image_number"])

csv_path = "/home/dksoren/KingD_Porj/src/crown_results.csv"
df.to_csv(csv_path, index=False, sep=";")

print(f"✅ Saved results to {csv_path}")
cv2.waitKey(0)






#

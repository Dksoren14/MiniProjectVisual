from zipfile import Path
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import math
import crown_detector
from pathlib import Path
import os

import preprocessing

def all_squares_checked(found_squares, total_squares):
    return found_squares >= total_squares

#def is_square_hill_or_forest(rezised_img):
#    lowest_green_value = 

def convert_to_HSI(image):
    # Convert BGR to float32 for precision
    bgr = image.astype(np.float32) / 255.0
    B, G, R = cv2.split(bgr)

    # Calculate Intensity
    I = (R + G + B) / 3.0

    # Calculate Saturation
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1 - (3 / (R + G + B + 1e-6)) * min_rgb
    S[I == 0] = 0  # If intensity is zero, saturation is zero

    # Calculate Hue
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G)**2 + (R - B) * (G - B)) + 1e-6 # Avoid division by zero by adding 0.000001
    theta = np.arccos(num / den)
    H = np.zeros_like(I)

    H[B <= G] = theta[B <= G]
    H[B > G] = (2 * np.pi) - theta[B > G]
    H = H / (2 * np.pi)  # Normalize to [0, 1]

    HSI = cv2.merge((H, S, I))
    return HSI

def convert_H_to_degrees(H):
    return H * 360.0

class color_checkers:
    def check_for_green_squares(resized_img):
        global found_squares, total_squares_checked, checker_array, new_height, new_width
        global light_green_square, dark_green_square
        for i in range(new_height):
            for j in range(new_width):
                if i == start_pos[0] and j == start_pos[1] or checker_array[i, j] == True:
                    continue
                else:
                    total_squares_checked += 1
                    if convert_to_HSI(resized_img)[i, j, 0] > 0.20 and convert_to_HSI(resized_img)[i, j, 0] < 0.4 and convert_to_HSI(resized_img)[i, j, 1] > 0.09:
                        found_squares += 1
                        checker_array[i, j] = 1
                        B, G, R = cv2.split(resized_img)
                        if G[i, j] > 100:
                            light_green_square += 1
                            color_array[i, j] = "LG"
                        else:
                            dark_green_square += 1
                            color_array[i, j] = "DG"
                        #print(f"Found green pixel at ({j}, {i}) with H: {convert_H_to_degrees(convert_to_HSI(resized_img)[i, j, 0])}, S: {convert_to_HSI(resized_img)[i, j, 1]}, I: {convert_to_HSI(resized_img)[i, j, 2]}")

        
    def check_for_yellow_squares(resized_img):
        global found_squares, total_squares_checked, checker_array, new_height, new_width
        global yellow_square
        for i in range(new_height):
            for j in range(new_width):
                if i == start_pos[0] and j == start_pos[1] or checker_array[i, j] == True:
                    continue
                else:
                    total_squares_checked += 1
                    if convert_to_HSI(resized_img)[i, j, 0] > 0.10 and convert_to_HSI(resized_img)[i, j, 0] < 0.20 and convert_to_HSI(resized_img)[i, j, 2] > 0.45: #bright yellow
                        found_squares += 1
                        checker_array[i, j] = 1
                        yellow_square += 1
                        color_array[i, j] = "Y"
                       
       
    def check_for_blue_squares(resized_img):
        global found_squares, total_squares_checked, checker_array, new_height, new_width
        global blue_square
        for i in range(new_height):
            for j in range(new_width):
                if i == 2 and j == 2 or checker_array[i, j] == True:
                    continue
                else:
                    total_squares_checked += 1
                    if convert_to_HSI(resized_img)[i, j, 0] > 0.55 and convert_to_HSI(resized_img)[i, j, 0] < 0.75:
                        found_squares += 1
                        checker_array[i, j] = 1
                        blue_square += 1
                        color_array[i, j] = "B"
                        

    

def main():
    
    base_dir = Path(__file__).resolve().parent
    image_dir = base_dir.parent / "Images"
    img = image_dir / f"2.jpg"
    template = image_dir / f"crown.png"
    target_intensity = 100
    img = cv2.imread(str(img))
    template = cv2.imread(str(template), cv2.IMREAD_GRAYSCALE)
    gamma_corrected = preprocessing.Preprocessing_tools.gamma_correction(img, preprocessing.Preprocessing_tools.find_gamma_value(img, target_intensity))
    blur = cv2.GaussianBlur(gamma_corrected, (5,5), 0)
    sharpened = cv2.addWeighted(gamma_corrected, 1.5, blur, -0.5, 0)
  
    cv2.imshow("Gamma Corrected", gamma_corrected)
    cv2.imshow("Sharpened", sharpened)
    print(f"crown found {crown_detector.output.output_for_score(gamma_corrected,template)[0]} crowns at positions {crown_detector.output.output_for_score(gamma_corrected,template)[1]}")
    result_img = crown_detector.output.output_for_score(gamma_corrected,template)[2]
    cv2.imshow("Original", img)
    cv2.waitKey(0)
   
if __name__ == '__main__':
    main()

#from 5x5 to 100x100 there is 20x20 pixel in each square
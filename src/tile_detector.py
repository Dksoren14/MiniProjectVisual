import cv2 
import numpy as np
import matplotlib.pyplot as plt
import math
import crown_detector
from pathlib import Path
import os

found_squares = 0
total_squares = 24
total_squares_checked = 0
base_dir = Path(__file__).resolve().parent
image_dir = base_dir.parent / "Images"
img = image_dir / f"5.jpg"
img = cv2.imread(str(img))
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#template = cv2.imread("/home/jesper-kwame-jensen/MiniProjectVisual/Images/crown.png", cv2.IMREAD_GRAYSCALE)
light_green_square = 0
dark_green_square = 0
brown_square = 0
yellow_square = 0
blue_square = 0
black_square = 0

new_width = 5
new_height = 5

newer_width = 10
newer_height = 10
start_pos = (2,2)  # Center position to skip
start_pos_10x10 = (4,4)  # Center position to skip 
resized_img_10x10 = cv2.resize(img, (newer_width, newer_height))

resized_img = cv2.resize(img, (new_width, new_height))
#print(f"Resized image shape: {resized_img[1,1]}")  


new_array_10x10 = np.zeros((newer_height, newer_width), dtype=bool)

checker_array = np.zeros((new_height, new_width), dtype=bool)

color_array = np.empty((new_height, new_width), dtype=object)

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
    def check_for_light_green_squares(newer_width, newer_height):
        global found_squares, total_squares_checked, checker_array, new_height, new_width, new_array_10x10,start_pos, start_pos_10x10
        global light_green_square
        # Liniing the 10x10 dark green array to the 5x5 checker array
        scale_i = newer_height // new_height  # 10 // 5 = 2
        scale_j = newer_width // new_width    # 10 // 5 = 2

        # resize the image for dark green squares
        resized_for_light_green = cv2.resize(img, (newer_width, newer_height))

        # Compute HSI once
        hsi_img = convert_to_HSI(resized_for_light_green)
        # create a new checker array for 10x10
        for i in range(newer_height):
            for j in range(newer_width):
                if (i == start_pos_10x10[0] and j == start_pos_10x10[1]):
                    continue
                
                h, s, intensity = hsi_img[i, j]
                total_squares_checked += 1
        
                if 0.20 < h < 0.40 and intensity > 0.29:
                    new_array_10x10[i, j] = 1
                    
                    
        for i in range(new_height):
            for j in range(new_width):
                if i == start_pos[0] and j == start_pos[1] or checker_array[i, j] == True:
                    continue 
                else:
                # Map 5x5 square to corresponding 10x10 block
                    i_start = i * scale_i
                    j_start = j * scale_j
                    i_end = i_start + scale_i
                    j_end = j_start + scale_j

                # If any pixel in this block is True in dark green_array, it marks the checker_array
                if np.any(new_array_10x10[i_start:i_end, j_start:j_end]):
                    #print(f"Dark green found at ({j}, {i})")
                    checker_array[i, j] = True
                    found_squares += 1
                    light_green_square += 1    
                    color_array[i, j] = "LG"         

    def check_for_dark_green_squares(newer_width, newer_height):
        global found_squares, total_squares_checked, checker_array, new_height, new_width, new_array_10x10,start_pos, start_pos_10x10
        global dark_green_square

        # Liniing the 10x10 dark green array to the 5x5 checker array
        scale_i = newer_height // new_height  # 10 // 5 = 2
        scale_j = newer_width // new_width    # 10 // 5 = 2
        

        # resize the image for dark green squares
        resized_for_dark_green = cv2.resize(img, (newer_width, newer_height))
        
        # create a new checker array for 10x10
        for i in range(newer_height):
            for j in range(newer_width):
                 if i == start_pos_10x10 and j == start_pos_10x10:
                     continue   
                 else:
                    total_squares_checked += 1
                    if (0.2 <= convert_to_HSI(resized_for_dark_green)[i, j, 0] <= 0.5 and 0.15 <= convert_to_HSI(resized_for_dark_green)[i, j, 1] <= 8.0 and 0 <= convert_to_HSI(resized_for_dark_green)[i, j, 2] < 0.2):
                     
                        new_array_10x10[i, j] = 1

                        #print(f"dark green pixel found at ({i}, {j}) - HSI={convert_to_HSI(resized_for_dark_green)[i, j]}")
        for i in range(new_height):
            for j in range(new_width):
                if i == start_pos[0] and j == start_pos[1] or checker_array[i, j] == True:
                    continue 
                else:
                # Map 5x5 square to corresponding 10x10 block
                    i_start = i * scale_i
                    j_start = j * scale_j
                    i_end = i_start + scale_i
                    j_end = j_start + scale_j

                # If any pixel in this block is True in dark green_array, it marks the checker_array
                if np.any(new_array_10x10[i_start:i_end, j_start:j_end]):
                    #print(f"Dark green found at ({j}, {i})")
                    checker_array[i, j] = True
                    found_squares += 1
                    dark_green_square += 1    
                    color_array[i, j] = "DG"    
        
    def check_for_yellow_squares(newer_width, newer_height):
        global found_squares, total_squares_checked, checker_array, new_height, new_width, new_array_10x10,start_pos, start_pos_10x10
        global yellow_square
    
        # Liniing the 10x10 dark green array to the 5x5 checker array
        scale_i = newer_height // new_height  # 10 // 5 = 2
        scale_j = newer_width // new_width    # 10 // 5 = 2
        

        # resize the image for dark green squares
        resized_for_yellow= cv2.resize(img, (newer_width, newer_height))
        
        # create a new checker array for 10x10
        for i in range(newer_height):
            for j in range(newer_width):
                if i == start_pos_10x10[0] and j == start_pos_10x10[1]: 
                    continue
                else:
                    total_squares_checked += 1
                    if (convert_to_HSI(resized_for_yellow)[i, j, 0] > 0.10 and convert_to_HSI(resized_for_yellow)[i, j, 0] < 0.20 and convert_to_HSI(resized_for_yellow)[i, j, 1] > 0.45):

                        new_array_10x10[i, j] = 1
                        
        for i in range(new_height):
            for j in range(new_width):
                if i == start_pos[0] and j == start_pos[1] or checker_array[i, j] == True:
                    continue 
                else:
                # Map 5x5 square to corresponding 10x10 block
                    i_start = i * scale_i
                    j_start = j * scale_j
                    i_end = i_start + scale_i
                    j_end = j_start + scale_j

                # If any pixel in this block is True in dark green_array, it marks the checker_array
                if np.any(new_array_10x10[i_start:i_end, j_start:j_end]):
                    #print(f"Dark green found at ({j}, {i})")
                    checker_array[i, j] = True
                    found_squares += 1
                    yellow_square += 1    
                    color_array[i, j] = "Y"    

                        #print(f"Found yellow pixel at ({j}, {i}) with H: {convert_H_to_degrees(convert_to_HSI(resized_img)[i, j, 0])}, S: {convert_to_HSI(resized_img)[i, j, 1]}, I: {convert_to_HSI(resized_img)[i, j, 2]}")
    def check_for_blue_squares(newer_width, newer_height):
        global found_squares, total_squares_checked, checker_array, new_height, new_width, new_array_10x10,start_pos, start_pos_10x10
        global blue_square

        # Liniing the 10x10 blue array to the 5x5 checker array
        scale_i = newer_height // new_height  # 10 // 5 = 2
        scale_j = newer_width // new_width    # 10 // 5 = 2
        
        # resize the image for blue squares
        resized_for_blue = cv2.resize(img, (newer_width, newer_height))
        
        # create a new checker array for 10x10

        for i in range(newer_height):
            for j in range(newer_width):
                 if i == start_pos_10x10[0] and j == start_pos_10x10[1]: 
                     continue   
                 else:
                    total_squares_checked += 1
                    if (0.58 <= convert_to_HSI(resized_for_blue)[i, j, 0] <= 0.61 and 0.78 <= convert_to_HSI(resized_for_blue)[i, j, 1] <= 1.0 and 0.23 <= convert_to_HSI(resized_for_blue)[i, j, 2] < 0.38):
                     
                        new_array_10x10[i, j] = 1

        for i in range(new_height):
            for j in range(new_width):
                if i == 2 and j == 2 or checker_array[i, j] == True:
                    continue 
                else:
                # Map 5x5 square to corresponding 10x10 block
                    i_start = i * scale_i
                    j_start = j * scale_j
                    i_end = i_start + scale_i
                    j_end = j_start + scale_j

                # If any pixel in this block is True in blue_array, it marks the checker_array
                if np.any(new_array_10x10[i_start:i_end, j_start:j_end]):
                    checker_array[i, j] = True
                    found_squares += 1
                    blue_square += 1    
                    color_array[i, j] = "B"
                        #print(f"Found blue pixel at ({j}, {i}) with H: {convert_H_to_degrees(convert_to_HSI(resized_img)[i, j, 0])}, S: {convert_to_HSI(resized_img)[i, j, 1]}, I: {convert_to_HSI(resized_img)[i, j, 2]}")                   
    def check_for_red_squares(newer_width, newer_height):
        global found_squares, total_squares_checked, checker_array, new_height, new_width, new_array_10x10,start_pos, start_pos_10x10
        global light_green_square

        # Liniing the 10x10 red array to the 5x5 checker array
        scale_i = newer_height // new_height  # 10 // 5 = 2
        scale_j = newer_width // new_width    # 10 // 5 = 2
        
        # resize the image for red squares
        resized_for_red = cv2.resize(img, (newer_width, newer_height))
    

        # Compute HSI once
        hsi_img = convert_to_HSI(resized_for_red)
        # create a new checker array for 10x10
        for i in range(newer_height):
            for j in range(newer_width):
                if (i == start_pos_10x10[0] and j == start_pos_10x10[1]) or checker_array[i, j]:
                    continue

                total_squares_checked += 1
                h, s, intensity = hsi_img[i, j]

                # Detect reddish tiles, but exclude dark/brown ones
                if ((0.00 < h < 0.10) or (0.90 < h <= 1.00)) and s > 0.3 and intensity > 0.2:
                    
                    new_array_10x10[i, j] = 1

        for i in range(new_height):
            for j in range(new_width):
                if i == start_pos[0] and j == start_pos[1] or checker_array[i, j] == True:
                    continue 
                else:
                # Map 5x5 square to corresponding 10x10 block
                    i_start = i * scale_i
                    j_start = j * scale_j
                    i_end = i_start + scale_i
                    j_end = j_start + scale_j

                # If any pixel in this block is True in brown_array, it marks the checker_array
                if np.any(new_array_10x10[i_start:i_end, j_start:j_end]):
                    
                    checker_array[i, j] = True
                    found_squares += 1
                    brown_square += 1    
                    color_array[i, j] = "LG"           

    def check_for_brown_squares(newer_width, newer_height):
        global found_squares, total_squares_checked, checker_array, new_height, new_width,start_pos, start_pos_10x10
        global brown_square

# Liniing the 10x10 brown array to the 5x5 checker array
        scale_i = newer_height // new_height  # 10 // 5 = 2
        scale_j = newer_width // new_width    # 10 // 5 = 2
        
        # resize the image for brown squares
        resized_for_brown = cv2.resize(img, (newer_width, newer_height))

        for i in range(newer_height):
            for j in range(newer_width):
                if i == start_pos_10x10[0] and j == start_pos_10x10[1]:
                    continue
                else:
                    total_squares_checked += 1
                    if (0.1 < convert_to_HSI(resized_for_brown)[i, j, 0] < 0.139 and 0.15 < convert_to_HSI(resized_for_brown)[i, j, 1] < 0.77 and 0.18 < convert_to_HSI(resized_for_brown)[i, j, 2] < 0.37):
                
                        new_array_10x10[i, j] = 1
                        
        for i in range(new_height):
            for j in range(new_width):
                if i == start_pos[0] and j == start_pos[1] or checker_array[i, j] == True:
                    continue 
                else:
                # Map 5x5 square to corresponding 10x10 block
                    i_start = i * scale_i
                    j_start = j * scale_j
                    i_end = i_start + scale_i
                    j_end = j_start + scale_j

                # If any pixel in this block is True in brown_array, it marks the checker_array
                if np.any(new_array_10x10[i_start:i_end, j_start:j_end]):
                    
                    checker_array[i, j] = True
                    found_squares += 1
                    brown_square += 1    
                    color_array[i, j] = "BR"     
                        #print(f"Found brown pixel at ({j}, {i}) with H: {convert_H_to_degrees(convert_to_HSI(resized_img)[i, j, 0])}, S: {convert_to_HSI(resized_img)[i, j, 1]}, I: {convert_to_HSI(resized_img)[i, j, 2]}")                    
    def check_for_black_squares(newer_width, newer_height):
        global found_squares, total_squares_checked, checker_array, new_height, new_width,start_pos, start_pos_10x10
        global black_square

# Liniing the 10x10 black array to the 5x5 checker array
        scale_i = newer_height // new_height  # 10 // 5 = 2
        scale_j = newer_width // new_width    # 10 // 5 = 2
        
        # resize the image for black squares
        resized_for_black = cv2.resize(img, (newer_width, newer_height))

        # create a new checker array for 10x10
        for i in range(newer_height):
            for j in range(newer_width):
                if i == start_pos_10x10[0] and j == start_pos_10x10[1]: 
                    continue
                else:
                    total_squares_checked += 1
                    if (0 < convert_to_HSI(resized_for_black)[i, j, 0] < 0.19 and 0 < convert_to_HSI(resized_for_black)[i, j, 1] < 0.4 and 0 < convert_to_HSI(resized_for_black)[i, j, 2] < 0.5):
                        
                        new_array_10x10[i, j] = 1
                        
        for i in range(new_height):
            for j in range(new_width):
                if i == start_pos[0] and j == start_pos[1] or checker_array[i, j] == True:
                    continue 
                else:
                # Map 5x5 square to corresponding 10x10 blockt
                    i_start = i * scale_i
                    j_start = j * scale_j
                    i_end = i_start + scale_i
                    j_end = j_start + scale_j

                # If any pixel in this block is True in brown_array, it marks the checker_array
                if np.any(new_array_10x10[i_start:i_end, j_start:j_end]):
                    
                    checker_array[i, j] = True
                    found_squares += 1
                    black_square += 1    
                    color_array[i, j] = "BL"                 

                        #print(f"Found black pixel at ({j}, {i}) with H: {convert_H_to_degrees(convert_to_HSI(resized_img)[i, j, 0])}, S: {convert_to_HSI(resized_img)[i, j, 1]}, I: {convert_to_HSI(resized_img)[i, j, 2]}")
        print(f"{checker_array} and found {found_squares} out of {total_squares} squares ")
class image_manipulator:
    def mask(img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([30, 0, 120])   # a bit below
        upper = np.array([35, 255, 255])   # yellow! note the order
        mask = cv2.inRange(img_hsv, lower, upper) #
        return mask
    def blur(img):
        median = cv2.medianBlur(img, 11) 
        return median
    def histogram_stretch(img):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(img)
        return cl1
    def normalize(img):
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        # Equalize only the Y channel (brightness)
        ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])

        # Convert back to BGR
        normalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return normalized
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated
def template_matching(img, template):
    found_crown = 0
    w, h = template.shape[:2]
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.3
    loc = np.where(res >= threshold)
    for i in range(0,5):
        rotated_template = rotate_image(template, i*90)
        w, h = rotated_template.shape[:2]
        res = cv2.matchTemplate(img, rotated_template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        found_crown += 1
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), 255, 2)
    #print(f"Found crown {found_crown} times")
    return img
class output:
    def output_for_score():
        return color_array
        
def main():
    global found_squares, total_squares, total_squares_checked, checker_array
    color_checkers.check_for_dark_green_squares(newer_width, newer_height)
    color_checkers.check_for_light_green_squares(newer_width, newer_height)
    color_checkers.check_for_brown_squares(newer_width, newer_height)
    color_checkers.check_for_blue_squares(newer_width, newer_height) 
    color_checkers.check_for_black_squares(newer_width, newer_height)
    color_checkers.check_for_yellow_squares(newer_width, newer_height)
    
    
    
    print(f"tile 0,1: {convert_to_HSI(resized_img)[0,1]}")
    print(f"tile 1,0: {convert_to_HSI(resized_img)[1,1]}")
    print(f"light green squares: {light_green_square}")
    print(f"dark green squares: {dark_green_square}")
    print(f"yellow squares: {yellow_square}")   
    print(f"Blue squares: {blue_square}")
    print(f"Black squares: {black_square}") 

    #cv2.imshow("OG Image", resized_img)
    cv2.imshow("OG Image", resized_img_10x10)
    cv2.imshow("normalized Image", image_manipulator.normalize(img))
    #edges = cv2.Canny(image_manipulator.normalize(img), 240,255) #max 
    
    #cv2.imshow("Manipulated",template_matching(edges, template))
    
    #print(convert_to_HSI(resized_img)[1,0]) #For printing HSI values of specific pixel
    print(convert_to_HSI(resized_img_10x10)[2,4]) #For printing HSI values of specific pixel
    #print(checker_array)
    print(new_array_10x10)
    print(color_array)
    cv2.waitKey(0)
    
if __name__ == '__main__':
    main()

#from 5x5 to 100x100 there is 20x20 pixel in each square

import cv2 
import numpy as np
import matplotlib.pyplot as plt
import math



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

        print(f"{checker_array} and found {found_squares} out of {total_squares} squares ")
        print(f"Light green squares: {light_green_square}, Dark green squares: {dark_green_square}")
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
                        #print(f"Found yellow pixel at ({j}, {i}) with H: {convert_H_to_degrees(convert_to_HSI(resized_img)[i, j, 0])}, S: {convert_to_HSI(resized_img)[i, j, 1]}, I: {convert_to_HSI(resized_img)[i, j, 2]}")

        print(f"{checker_array} and found {found_squares} out of {total_squares} squares ")
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
                        #print(f"Found blue pixel at ({j}, {i}) with H: {convert_H_to_degrees(convert_to_HSI(resized_img)[i, j, 0])}, S: {convert_to_HSI(resized_img)[i, j, 1]}, I: {convert_to_HSI(resized_img)[i, j, 2]}")

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
class crown_finder:
    def template_matching(img, template):
        global crown_array_buffer
        found_crown = 0
        w, h = template.shape[:2]
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
     
        threshold = 0.275

        loc = np.where(res >= threshold)
        buffer = 20
        for i in range(0,5):
            rotated_template = rotate_image(template, i*90)
            w, h = rotated_template.shape[:2]
            res = cv2.matchTemplate(img, rotated_template, cv2.TM_CCOEFF_NORMED) 
            loc = np.where(res >= threshold)
            points = list(zip(*loc[::-1]))
            for pt in zip(*loc[::-1]):
                    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), 255, 2)
           
       
        return img, w,h

    def map_value(value, in_min, in_max, out_min, out_max):
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def morphology(img):

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=15) 
        #erosion = cv2.erode(closed, kernel, iterations=10)
        return closed 
    def component_analysis(img, w, h):
        ''
        
        ''
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
        expected_area = int(w * h * 0.95)
        estimated_crowns = 0
        new_centroids = []
        for i in range(1, num_labels):  # skip background
            x, y, bw, bh, area = stats[i]
            cx, cy = centroids[i]
            n = round(area / expected_area)
            n = max(1, n)
            estimated_crowns += n

            if n == 1:
                new_centroids.append((cx, cy))
            else:
                # distribute synthetic centroids in the bounding box region
                for j in range(n):
                    # small random offset to spread them inside the blob box
                    ox = np.random.uniform(-bw/4, bw/4)
                    oy = np.random.uniform(-bh/4, bh/4)
                    new_centroids.append((cx + ox, cy + oy))
        
        return estimated_crowns, np.array(new_centroids), stats



def main():
    found_squares = 0
    total_squares = 24
    total_squares_checked = 0
    new_width = 5
    new_height = 5
    img = cv2.imread("/home/dksoren/KingD_Porj/Images/3.jpg")
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.imread("/home/dksoren/KingD_Porj/Images/crown.png", cv2.IMREAD_GRAYSCALE)
    checker_array = np.zeros((new_height, new_width), dtype=bool)

    start_pos = [2,2]
    checker_array[start_pos] = True
    light_green_square = 0
    dark_green_square = 0
    yellow_square = 0
    blue_square = 0
  
    found_crown = 0
    resized_img = cv2.resize(img, (new_width, new_height))
    color_array = np.zeros((new_height, new_width), dtype=object)
    crown_array = np.zeros((new_height, new_width), dtype=bool)
    crown_array_buffer = []

    edges = cv2.Canny(image_manipulator.normalize(img), 240,255)
    
    #color_checkers.check_for_green_squares(resized_img)
    #color_checkers.check_for_yellow_squares(resized_img)
    #color_checkers.check_for_blue_squares(resized_img)
    #print(f"Found light greeen: {light_green_square}, dark green: {dark_green_square} and yellow: {yellow_square} and blue: {blue_square}")
    #crowns = crown_finder.template_matching(grayscale_img, template)
    #print(f"Color array: \n{color_array}")
    #print(f"crowns found: {found_crown}")
    #cv2.imshow("Resized Image", resized_img )
    #cv2.imshow("Blurred Image", image_manipulator.normalize(img))
    #resized_img = cv2.resize(image_manipulator.mask(image_manipulator.blur(img)), (new_width, new_height))
    #test_map = crown_finder.map_value(2, 0,5,0,100)
     # Function to map a value from one range to another
    #crown_finder.template_matching(grayscale_img, template)
    saved_edges = cv2.Canny(image_manipulator.normalize(img), 240,255)
    image_with_crowns, width, height = crown_finder.template_matching(edges, template)
    only_squares = image_with_crowns - saved_edges
    separeted_crown = crown_finder.morphology(only_squares)
    #cv2.imshow("OG Image", img)
    #cv2.imshow("normalized Image", image_manipulator.normalize(img))
     #max 
    #cv2.imshow("Wtf am i doing", (separeted_crown))
    cv2.imshow("Template", image_with_crowns)
    numb_labels, centroids, stats = crown_finder.component_analysis(separeted_crown, width, height)
    height_img, width_img = img.shape[:2]
    print(f"Number of crowns: {numb_labels}, with width {width} and height {height}")
    wtf = height/5
    print(wtf)
    print(f"corwn at height {round(crown_finder.map_value(centroids[1,1], 0, height_img, 0, 7.4))}, and width {round(crown_finder.map_value(centroids[1,0], 0, width_img, 0, 7.2))}")
    print(f"Og position = ({centroids[1,1]}, {centroids[1,0]})")
    
    cv2.waitKey(0)
   
if __name__ == '__main__':
    main()

#from 5x5 to 100x100 there is 20x20 pixel in each square
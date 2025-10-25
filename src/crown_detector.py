
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import math
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
class crown_detector:
    def template_matching(img, template):

    

        threshold = 0.275

  
        for i in range(0,5):
            rotated_template = rotate_image(template, i*90)
            w, h = rotated_template.shape[:2]
            res = cv2.matchTemplate(img, rotated_template, cv2.TM_CCOEFF_NORMED) 
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):
                    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), 255, 2)
           
       
        return img, w,h

    def map_value(value, in_min, in_max, out_min, out_max):
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def morphology(img):

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=20) 
        
        return closed 
    
    def correspond_to_5x5(centroids, width, height, new_width=5, new_height=5):
        crown_positions = []
        cell_width = width / new_width
        cell_height = height / new_height
        for (cx, cy) in centroids:
            
            grid_x = int(cx // cell_width)
            grid_y = int(cy // cell_height)
            crown_positions.append((grid_x, grid_y))
            
        return crown_positions


    def component_analysis(img, w, h):
        '' 

        ''
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
        expected_area = int(w * h * 1.2)  # slightly larger than template area
       
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
                step_x = bw / (n + 1)
                for j in range(n):
                    new_x = x + step_x * (j + 1)
                    new_y = y + bh / 2  # centered vertically
                    new_centroids.append((new_x, new_y))
        
        return estimated_crowns, np.array(new_centroids), stats
    

class output:
    def output_for_score(img,template,):
        edges = cv2.Canny(image_manipulator.normalize(img), 240,255)
        
        non_manipulated_edges = cv2.Canny(image_manipulator.normalize(img), 240,255)
        image_with_crowns, width, height = crown_detector.template_matching(edges, template)
        only_squares = image_with_crowns - non_manipulated_edges
        separeted_crown = crown_detector.morphology(only_squares)

        numb_labels, centroids, stats = crown_detector.component_analysis(separeted_crown, width, height)
      
        crown_position = crown_detector.correspond_to_5x5(centroids, img.shape[1], img.shape[0])
        return numb_labels, crown_position, image_with_crowns
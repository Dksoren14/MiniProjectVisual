import cv2
import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt
import math

class Preprocessing_tools:
    def determine_intensity(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        average_intensity = np.mean(gray_image)
        return average_intensity

    def find_gamma_value(image, target_intensity):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current_intensity = np.mean(gray_image)
        if current_intensity == 0:
            return None
        gamma_value = math.log(target_intensity / 255) / math.log(current_intensity / 255)
        return gamma_value

    def gamma_correction(image, gamma):
        gamma_corrected = np.array(255*(image / 255) ** gamma, dtype = 'uint8')
        return gamma_corrected
    def sharpen_image(img):
        blur = cv2.GaussianBlur(img, (5,5), 0)
        sharpened = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
        return sharpened
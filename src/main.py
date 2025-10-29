import cv2 
import numpy as np
import matplotlib.pyplot as plt
import math
import crown_detector
from pathlib import Path
import os
import tile_detector
import crown_detector
import preprocessing
from collections import deque
import pandas as pd
from pathlib import Path



def get_start_positions10x(start_pos):
    x, y = start_pos
    start_pos_10x10 = (x + 2, y + 2)
    return start_pos_10x10


def calculate_score(grid, crown_positions):
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    total_score = 0
    # Convert crown positions to a set for quick lookup
    crown_set = set(crown_positions)

    def bfs(start_y, start_x):
        color = grid[start_y][start_x]
        queue = deque([(start_y, start_x)])
        visited[start_y][start_x] = True
        region_tiles = 0
        region_crowns = 0

        while queue:
            y, x = queue.popleft()
            region_tiles += 1
            if (y, x) in crown_set:
                region_crowns += 1

            for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < rows and 0 <= nx < cols:
                    if not visited[ny][nx] and grid[ny][nx] == color:
                        visited[ny][nx] = True
                        queue.append((ny, nx))

        return region_tiles * region_crowns

    for y in range(rows):
        for x in range(cols):
            if not visited[y][x]:
                total_score += bfs(y, x)

    return total_score

def testing(image_dir,template):
    num_images = 74
    results = [] 
    total_objects = 0
  
    start_pos_arr = [
    (2, 2), (4, 2), (2, 2), (2, 4), (4, 3), (2, 4), (2, 4), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2),
    (4, 2), (2, 2), (4, 2), (2, 2), (3, 1), (3, 1), (4, 4), (2, 2), (4, 4), (2, 2), (2, 2), (2, 2),
    (2, 2), (2, 2), (2, 2), (2, 2), (2, 4), (2, 0), (2, 4), (4, 0), (2, 0), (4, 4), (4, 2), (4, 0),
    (4, 4), (4, 4), (4, 0), (4, 4), (4, 0)
]

    j = 0
    for i in range(1, num_images + 1):
        img_path = image_dir / f"{i}.jpg"


        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to load image at {img_path}")
            continue
        start_pos = start_pos_arr[j]
        gamma_corrected = preprocessing.Preprocessing_tools.gamma_correction(img, preprocessing.Preprocessing_tools.find_gamma_value(img, 100))
        grid = tile_detector.output.output_for_score(gamma_corrected, start_pos, get_start_positions10x(start_pos))
        sharpened = preprocessing.Preprocessing_tools.sharpen_image(img)
        j += 1
        num_labels, crown_position, image_with_crowns = crown_detector.output.output_for_score(sharpened,template)
        crown_map = {}
        for y, x in crown_position:
            crown_map[(y, x)] = crown_map.get((y, x), 0) + 1
        print("Crown positions:", crown_position)
        print("Grid:", grid)
        points = calculate_score(grid, crown_position)
        print("Total points:", points)
        total_objects += num_labels
        results.append([points, i])

    
        df = pd.DataFrame(results, columns=["Score", "image_number"])

        csv_path = "/home/dksoren/KingD_Porj/src/crown_results.csv"
        df.to_csv(csv_path, index=False, sep=";")

        print(f"✅ Saved results to {csv_path}")
        cv2.waitKey(0)


def main():

    
    base_dir = Path(__file__).resolve().parent
    image_dir = base_dir.parent / "Cropped and perspective corrected boards"
    img = image_dir / f"28.jpg"
    img = cv2.imread(str(img))
    image_dir_template = base_dir.parent / "Images"
    template = image_dir_template / f"crown.png"
    template = cv2.imread(str(template), cv2.IMREAD_GRAYSCALE)
    
    testing(image_dir, template)
    
    cv2.imshow("Image", img)
    cv2.waitKey(0)
if __name__ == '__main__':
    main()



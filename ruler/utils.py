import os
import cv2

def get_absolute_path(path: str):
    absolute_path = os.path.dirname(__file__)
    return os.path.join(absolute_path, path)

def find_center(mask):
    m = cv2.moments(mask)
    return (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
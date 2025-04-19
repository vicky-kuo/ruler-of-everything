import os
import cv2

def getAbsolutePath(path: str):
    absolutePath = os.path.dirname(__file__)
    return os.path.join(absolutePath, path)

def find_center(mask):
    m = cv2.moments(mask)
    return (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
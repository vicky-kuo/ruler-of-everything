import os
import cv2
from skimage.io import imsave, imread


def get_absolute_path(path: str):
    absolute_path = os.path.dirname(__file__)
    return os.path.join(absolute_path, path)


def find_center(mask):
    m = cv2.moments(mask)
    return (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))


def caculate_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def downsample_image(file, target_size=640):
    image = imread(file)
    h, w = image.shape[:2]
    # Prevent upsampling
    if max(h, w) <= target_size:
        print(
            f"Image already smaller than target size {target_size}. Skipping downsampling."
        )
        return
    ratio = target_size / max(h, w)
    new_h, new_w = int(ratio * h), int(ratio * w)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    print(f"Downsampling image to {new_w}x{new_h}")
    imsave(file, resized_image)

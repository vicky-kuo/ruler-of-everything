from pathlib import Path
from skimage.io import imread, imsave
from ultralytics import YOLO
import numpy as np
import cv2
from utils.other_utils import find_center
import env


def predict(
    input_image_path: Path,
    output_image_path: Path,
) -> tuple[tuple[int, int] | None, float | None]:
    """
    Predicts ruler center and pixel length from an image, saves an annotated image.

    Args:
        input_image_path: Path to the input image.
        output_annotated_image_path: Path to save the image with ruler annotations.
        ruler_model_config_path: Path to the YOLO model for ruler detection.

    Returns:
        tuple: (
            (center_x, center_y): tuple[int, int] or None if no ruler found,
            pixel_length: float or None if no ruler found,
        )
    """
    model = YOLO(env.ruler_model_path)
    results = model(input_image_path, verbose=False)

    best_ruler_center = None
    best_ruler_width = None
    highest_confidence = -1.0
    img = None

    if not results:
        print(f"WARN! YOLO model returned no results for: {input_image_path}")
        img = imread(input_image_path)
        imsave(output_image_path, img)
        print(
            f"Saved original image to {output_image_path} as no YOLO results were found."
        )
        return None, None

    for result in results:
        img = result.plot()

        if hasattr(result.masks, "xy") and hasattr(result.boxes, "conf"):
            for mask_points, box in zip(result.masks.xy, result.boxes):
                label = model.names[int(box.cls[0])]
                confidence = float(box.conf[0])

                if label == "ruler":
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        points = np.int32([mask_points])
                        current_center = find_center(points)
                        min_x = np.min(points[:, :, 0])
                        max_x = np.max(points[:, :, 0])
                        current_width = max_x - min_x

                        best_ruler_center = current_center
                        best_ruler_width = float(current_width)
                        cv2.circle(img, best_ruler_center, 5, (0, 255, 0), -1)
        else:
            print(f"WARN! No masks or boxes found in results for: {input_image_path}")

    if highest_confidence == -1.0:
        print(f"WARN! No ruler found in picture: {input_image_path}")

    if img is not None:
        print(f"Saving ruler annotation output to: {output_image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        imsave(output_image_path, img)
    else:
        print(
            f"Error: Image data for annotation was not available for {input_image_path}. Cannot save."
        )
        return None, None

    return best_ruler_center, best_ruler_width

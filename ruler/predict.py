from pathlib import Path
from skimage.io import imread, imsave
from ultralytics import YOLO
import numpy as np
import cv2
import env


def predict(
    input_image_path: Path,
    output_image_path: Path,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    model = YOLO(env.ruler_model_path)
    results = model(input_image_path, verbose=False)

    best_ruler_endpoint1 = None
    best_ruler_endpoint2 = None
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
                        contour_points = np.array(mask_points, dtype=np.float32)

                        # Fit a rotated rectangle
                        rect = cv2.minAreaRect(contour_points)
                        box_cv_pts = cv2.boxPoints(
                            rect
                        )  # These are 4 float32 corner points

                        # The sides of the rectangle
                        # side lengths: d(v0,v1), d(v1,v2)
                        # v0, v1, v2, v3 are the corners from box_cv_pts
                        side01_len_sq = np.sum((box_cv_pts[0] - box_cv_pts[1]) ** 2)
                        side12_len_sq = np.sum((box_cv_pts[1] - box_cv_pts[2]) ** 2)

                        if side01_len_sq < side12_len_sq:
                            # side01 and side23 are shorter (these are the ends of the ruler)
                            # endpoints are midpoints of these shorter sides
                            current_endpoint1 = (box_cv_pts[0] + box_cv_pts[1]) / 2.0
                            current_endpoint2 = (box_cv_pts[2] + box_cv_pts[3]) / 2.0
                        else:
                            # side12 and side30 are shorter
                            current_endpoint1 = (box_cv_pts[1] + box_cv_pts[2]) / 2.0
                            current_endpoint2 = (box_cv_pts[3] + box_cv_pts[0]) / 2.0

                        best_ruler_endpoint1 = current_endpoint1
                        best_ruler_endpoint2 = current_endpoint2
        else:
            print(f"WARN! No masks or boxes found in results for: {input_image_path}")

    if highest_confidence == -1.0:
        print(f"WARN! No ruler found in picture: {input_image_path}")

    if img is not None:
        if best_ruler_endpoint1 is not None and best_ruler_endpoint2 is not None:
            # Draw the line representing the ruler
            pt1_draw = tuple(best_ruler_endpoint1.astype(int))
            pt2_draw = tuple(best_ruler_endpoint2.astype(int))
            cv2.line(img, pt1_draw, pt2_draw, (0, 255, 0), 2)

        print(f"Saving ruler annotation output to: {output_image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        imsave(output_image_path, img)
    else:
        print(
            f"Error: Image data for annotation was not available for {input_image_path}. Cannot save."
        )
        return None, None

    return best_ruler_endpoint1, best_ruler_endpoint2

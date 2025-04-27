from ultralytics import YOLO
import numpy as np
import os
import cv2
from utils.other_utils import find_center
import env


def predict(args, input_file: str):
    model = YOLO(env.ruler_model_path)
    results = model(input_file, verbose=False)
    name = os.path.basename(input_file).split(".")[0]

    best_ruler_center = (0, 0)
    best_ruler_width = 0
    highest_confidence = -1.0

    for result in results:
        img = result.plot()

        if hasattr(result.masks, "xy") and hasattr(result.boxes, "conf"):
            for mask, box in zip(result.masks.xy, result.boxes):
                label = model.names[box.cls.tolist().pop()]
                confidence = box.conf.tolist().pop()

                if label == "ruler":
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        points = np.int32([mask])
                        best_ruler_center = find_center(points)
                        min_x = np.min(points[:, :, 0])
                        max_x = np.max(points[:, :, 0])
                        best_ruler_width = max_x - min_x
                        cv2.circle(img, best_ruler_center, 5, (0, 255, 0), -1)

        if highest_confidence == -1.0:
            print("WARN! no ruler found in picture: " + result.path)

        output_file_path = os.path.join(args.output, f"{name}-ruler.jpg")

        print(f"Saving ruler annotation output to: {output_file_path}")
        print("Center:", best_ruler_center)
        print("Width:", best_ruler_width)
        print("-" * 20)
        cv2.imwrite(output_file_path, img)

    return best_ruler_center, best_ruler_width


def _test():
    args = type("ObjectArgs", (object,), {})()
    args.output_path = env.output_path

    filenames = [file for file in os.listdir(env.input_path)]

    for filename in filenames:
        args.name = filename.split(".")[0]
        _, _ = predict(args, os.path.join(env.input_path, filename))


# test the predict function
if __name__ == "__main__":
    _test()

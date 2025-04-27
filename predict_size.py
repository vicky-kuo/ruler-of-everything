import cv2
from pathlib import Path
from skimage.io import imsave, imread

import env
from utils.other_utils import caculate_distance
from ruler.predict import predict as ruler_predict
from predict_frame import predict as gen6d_predict


def downsample_image(file, target_size=640):
    image = imread(file)
    h, w = image.shape[:2]
    ratio = target_size / max(h, w)
    new_h, new_w = int(ratio * h), int(ratio * w)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    imsave(file, resized_image)


args = env.girl_args

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

downsample_image(args.input_file1)
downsample_image(args.input_file2)

ruler_center_1, ruler_width_1 = ruler_predict(args, args.input_file1)
ruler_center_2, ruler_width_2 = ruler_predict(args, args.input_file2)

object_center_1, object_width_1, object_height_1 = gen6d_predict(args, args.input_file1)
object_center_2, object_width_2, object_height_2 = gen6d_predict(args, args.input_file2)

ruler_move = caculate_distance(ruler_center_1, ruler_center_2)
ruler_width = (ruler_width_1 + ruler_width_2) / 2

object_move = caculate_distance(object_center_1, object_center_2)
object_width = (object_width_1 + object_width_2) / 2
object_height = (object_height_1 + object_height_2) / 2

ratio = object_move / ruler_move if ruler_move != 0 else 0
ruler_real_width = 15
predict_ruler_width = ruler_width * ratio
object_predict_width = object_width / predict_ruler_width * ruler_real_width
object_predict_height = object_height / predict_ruler_width * ruler_real_width

print(f"Ruler center 1: {ruler_center_1}, width: {ruler_width_1}")
print(f"Ruler center 2: {ruler_center_2}, width: {ruler_width_2}")
print(f"Ruler movement: {ruler_move}")
print(f"Average ruler width: {ruler_width}")
print("-" * 20)
print(f"Object center 1: {object_center_1}")
print(f"Object center 2: {object_center_2}")
print(f"Object movement: {object_move}")
print(f"Average object width: {object_width}")
print(f"Average object height: {object_height}")
print(f"Object predict width: {object_predict_width}")
print(f"Object predict height: {object_predict_height}")
print("-" * 20)

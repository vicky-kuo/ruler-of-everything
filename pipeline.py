import shutil
from pathlib import Path
import subprocess
from skimage.io import imread
import numpy as np
import tqdm

from prepare import video2image
from ruler.predict import predict as ruler_predict
from predict_frame import predict as gen6d_predict
from predict_frame import draw as gen6d_draw
from predict_size import predict as size_predict
from predict_size import draw as size_draw
from ar import render as ar_render

import env
from estimator import name2estimator
from dataset.database import parse_database_name, get_ref_point_cloud
from utils.base_utils import load_cfg
from utils.draw_utils import pts_range_to_bbox_pts
import re

logger = env.logger

obj_args_list = [env.girl_args, env.minion_args]
model_args = [env.soda_can_model, env.soda_bottle_model]
ruler_real_length_cm = 15.0
interval = 1


def main():
    video_name = Path("video_3.mp4")
    input_video_path = env.input_path / video_name
    base_output_path = Path(env.output_path) / video_name.stem
    shutil.rmtree(base_output_path, ignore_errors=True)

    # Convert video to frames
    raw_frames_path = base_output_path / "raw"
    ruler_frames_path = base_output_path / "ruler"
    size_frames_path = base_output_path / "sized"

    raw_frames_path.mkdir(parents=True, exist_ok=True)
    ruler_frames_path.mkdir(parents=True, exist_ok=True)
    size_frames_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Converting video to frames in {raw_frames_path}...")
    frame_num = video2image(
        input_video_path, raw_frames_path, interval=interval, image_size=640
    )

    frame_files = sorted(
        [f for f in raw_frames_path.iterdir() if f.is_file()],
        key=lambda p: int(re.search(r"frame(\d+)", p.stem).group(1)),
    )

    if not frame_files:
        logger.warning(f"No frames extracted from {input_video_path}. Exiting.")
        return

    logger.info(f"Found {frame_num} frames to process.")

    # --- 2. Initialize Gen6D estimators ---
    estimators = {}
    obj_bboxes_3d = {}
    min_pts = {}
    max_pts = {}
    hist_pts = {obj_args.name: [] for obj_args in obj_args_list}
    pose_inits: dict[str, np.ndarray | None] = {
        obj_args.name: None for obj_args in obj_args_list
    }
    ar_models = []
    for obj_args in obj_args_list:
        obj_name = obj_args.name

        obj_output_base_path = base_output_path / obj_name
        gen6d_frames_path = obj_output_base_path / "gen6d"
        ar_frames_path = obj_output_base_path / "ar"

        gen6d_frames_path.mkdir(parents=True, exist_ok=True)
        ar_frames_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing estimator for {obj_name}...")
        cfg = load_cfg(obj_args.cfg)

        database = parse_database_name(obj_args.database)
        obj_pts = get_ref_point_cloud(database)
        _max_pts = np.max(obj_pts, 0)
        _min_pts = np.min(obj_pts, 0)
        max_pts[obj_name] = _max_pts
        min_pts[obj_name] = _min_pts
        obj_bboxes_3d[obj_name] = pts_range_to_bbox_pts(_max_pts, _min_pts)

        estimator = name2estimator[cfg["type"]](cfg)
        estimators[obj_name] = estimator
        estimator.build(database, split_type="all")

        ar_models.append(model_args[0])

    img = imread(frame_files[0])
    h, w, _ = img.shape
    f = np.sqrt(h**2 + w**2)
    K = np.asarray([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float32)

    # --- 3 & 4. Process each frame ---
    for frame_path in tqdm.tqdm(frame_files):
        current_raw_frame_path = frame_path
        frame_name = frame_path.stem
        logger.debug("-" * 20 + frame_name + "-" * 20)

        # --- 3a. Ruler Prediction ---
        ruler_annotated_image_path = ruler_frames_path / f"{frame_name}_ruler.jpg"
        logger.debug(f"Ruler predict on {current_raw_frame_path}")

        # MODIFIED: ruler_predict now returns two endpoints
        ruler_endpoint1, ruler_endpoint2 = ruler_predict(
            input_image_path=current_raw_frame_path,
            output_image_path=ruler_annotated_image_path,
        )

        model_idx = 0
        gen6d_predictions = []
        size_predictions = []

        for obj_args in obj_args_list:
            obj_name = obj_args.name

            logger.debug("-" * 20 + f"{obj_name}" + "-" * 20)
            # --- 4a. Gen6D Prediction (predict_frame.py) ---
            logger.debug(
                f"  Gen6D predict for {obj_name} on {current_raw_frame_path.name}"
            )
            try:
                gen6d_prediction = gen6d_predict(
                    input_image_path=current_raw_frame_path,
                    K_matrix=K,
                    estimator=estimators[obj_name],
                    obj_bbox_3d=obj_bboxes_3d[obj_name],
                    hist_pts=hist_pts[obj_name],
                    pose_init=pose_inits[obj_name],
                    smoothing_num=obj_args.num,
                    smoothing_std=obj_args.std,
                )
                pose_inits[obj_name] = gen6d_prediction["pose"]
                gen6d_predictions.append(gen6d_prediction)

            except Exception as e:
                logger.error(f"  Error in Gen6D prediction for {obj_name}: {e}")
                continue

            # MODIFIED: Pass endpoints to size_predict
            if ruler_endpoint1 is not None and ruler_endpoint2 is not None:
                size_prediction = size_predict(
                    pose=gen6d_prediction["pose"],
                    K_matrix=K,
                    min_pt=min_pts[obj_name],
                    max_pt=max_pts[obj_name],
                    ruler_pt1_2d=ruler_endpoint1,  # MODIFIED
                    ruler_pt2_2d=ruler_endpoint2,  # MODIFIED
                    ruler_real_length_cm=ruler_real_length_cm,
                    obj_name=obj_name,
                )
                size_predictions.append(size_prediction)
            else:
                logger.warning(
                    f"  Skipping size prediction for {obj_name} due to missing ruler information."
                )
                # Append a placeholder or handle missing size prediction as needed
                size_predictions.append(
                    {
                        "name": obj_name,
                        "width": None,
                        "height": None,
                        "depth": None,
                    }
                )

        # --- Draw ---
        current_image_path = ruler_annotated_image_path
        model_idx = 0

        for obj_args in obj_args_list:
            obj_name = obj_args.name
            obj_output_base_path = base_output_path / obj_name
            ar_frames_path = obj_output_base_path / "ar"
            ar_annotated_image_path = ar_frames_path / f"{frame_name}_ar.jpg"

            for model in model_args:
                if (
                    size_predictions[model_idx]["width"] is None
                    or size_predictions[model_idx]["height"] is None
                    or size_predictions[model_idx]["depth"] is None
                ):
                    break
                max_axis = max(
                    size_predictions[model_idx]["width"],
                    size_predictions[model_idx]["height"],
                    size_predictions[model_idx]["depth"],
                )
                if max_axis >= model.range_cm[0] and max_axis < model.range_cm[1]:
                    ar_models[model_idx] = model
                    break

            logger.debug(
                f"AR render (using model {ar_models[model_idx].name}) on {current_image_path}"
            )
            ar_render(
                input_image_path=current_image_path,
                output_image_path=ar_annotated_image_path,
                model_args=ar_models[model_idx],
                pose_target=gen6d_predictions[model_idx]["pose"],
                K_matrix=K,
                target_min_pt=min_pts[obj_name],
                target_max_pt=max_pts[obj_name],
            )
            current_image_path = ar_annotated_image_path
            gen6d_frames_path = obj_output_base_path / "gen6d"
            gen6d_annotated_image_path = gen6d_frames_path / f"{frame_name}_gen6d.jpg"
            logger.debug(
                f"Saving Gen6D annotated image to {gen6d_annotated_image_path}"
            )
            gen6d_draw(
                input_image_path=current_image_path,
                output_image_path=gen6d_annotated_image_path,
                pts=gen6d_predictions[model_idx]["pts"],
                save=True,
            )
            current_image_path = gen6d_annotated_image_path
            model_idx += 1

        for obj_args in obj_args_list:
            obj_name = obj_args.name
            size_frames_path = base_output_path / "sized"
            size_annotated_image_path = size_frames_path / f"{frame_name}_sized.jpg"
            logger.debug(f"Saving size annotated image to {size_annotated_image_path}")
            size_draw(
                input_image_path=current_image_path,
                output_image_path=size_annotated_image_path,
                sizes=size_predictions,
                save=True,
            )

        model_idx = model_idx + 1

    list_file = base_output_path / "frames.txt"
    output_files = sorted(
        [f for f in Path(base_output_path / "sized").iterdir() if f.is_file()],
        key=lambda p: int(re.search(r"frame(\d+)_sized", p.stem).group(1)),
    )
    with open(list_file, "w") as f:
        for frame_file in output_files:
            f.write(f"file '{frame_file.absolute()}'\n")

    output_fps_str = f"{30 / interval}"
    cmd = [
        env.ffmpeg_path,
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-r",
        output_fps_str,
        "-vf",
        f"scale=640:-2,settb=AVTB,setpts=N/{output_fps_str}/TB,fps={output_fps_str}",  # Keep width 640, make height even
        "-c:v",
        "libx264",
        f"{base_output_path}/{video_name}",
    ]
    subprocess.run(cmd)

    list_file.unlink(missing_ok=True)

    logger.info("Pipeline processing complete.")
    logger.info(f"Outputs are in {base_output_path}")


if __name__ == "__main__":
    main()

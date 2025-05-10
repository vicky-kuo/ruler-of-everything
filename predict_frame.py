import numpy as np
import cv2
import warnings
from skimage.io import imsave, imread
from pathlib import Path

from estimator import Gen6DEstimator
from utils.base_utils import project_points
from utils.draw_utils import draw_bbox_3d
from utils.pose_utils import pnp

warnings.filterwarnings("ignore", category=UserWarning)


def weighted_pts(pts_list, weight_num=10, std_inv=10):
    weights = np.exp(-((np.arange(weight_num) / std_inv) ** 2))[::-1]  # wn
    pose_num = len(pts_list)
    if pose_num < weight_num:
        weights = weights[-pose_num:]
    else:
        pts_list = pts_list[-weight_num:]
    pts = np.sum(np.asarray(pts_list) * weights[:, None, None], 0) / np.sum(weights)
    return pts


def cacluate_center(pts2d):
    center = tuple(map(int, np.mean(pts2d, axis=0)))
    return center


def predict(
    input_image_path: Path,
    output_image_path: Path,
    K_matrix: np.ndarray,
    estimator: Gen6DEstimator,
    obj_bbox_3d: np.ndarray,
    hist_pts: list,
    pose_init: np.ndarray | None,
    smoothing_num: int,
    smoothing_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    if pose_init is not None:
        estimator.cfg["refine_iter"] = 1

    img = imread(input_image_path)

    pose_pr, _ = estimator.predict(img, K_matrix, pose_init=pose_init)
    pts, _ = project_points(obj_bbox_3d, pose_pr, K_matrix)

    # hist_pts = []
    print(hist_pts)
    hist_pts.append(pts)
    pts_ = weighted_pts(hist_pts, weight_num=smoothing_num, std_inv=smoothing_std)
    pose_ = pnp(obj_bbox_3d, pts_, K_matrix)  # Smoothed pose
    pts__, _ = project_points(obj_bbox_3d, pose_, K_matrix)
    bbox_img_ = draw_bbox_3d(img, pts__, (0, 0, 255))  # Draw smoothed bbox

    center = cacluate_center(pts__)
    cv2.circle(bbox_img_, center, 5, (0, 255, 0), -1)

    print(f"Saving Gen6D bbox annotation output to: {output_image_path}")
    imsave(output_image_path, bbox_img_)

    return (
        pose_,  # Return the smoothed pose
        pose_pr,  # Current raw prediction becomes pose_init for the next frame
    )

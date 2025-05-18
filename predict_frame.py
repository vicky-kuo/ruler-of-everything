import numpy as np
import warnings
from skimage.io import imread, imsave
from pathlib import Path

from estimator import Gen6DEstimator
from utils.base_utils import project_points
from utils.draw_utils import draw_bbox_3d
from utils.pose_utils import pnp

warnings.filterwarnings("ignore", category=UserWarning)


def weighted_pts(pts_list, weight_num=10, std_inv: float = 10):
    weights = np.exp(-((np.arange(weight_num) / std_inv) ** 2))[::-1]  # wn
    pose_num = len(pts_list)
    if pose_num < weight_num:
        weights = weights[-pose_num:]
    else:
        pts_list = pts_list[-weight_num:]
    pts = np.sum(np.asarray(pts_list) * weights[:, None, None], 0) / np.sum(weights)
    return pts


def predict(
    input_image_path: Path,
    K_matrix: np.ndarray,
    estimator: Gen6DEstimator,
    obj_bbox_3d: np.ndarray,
    hist_pts: list,
    pose_init: np.ndarray | None,
    smoothing_num: int,
    smoothing_std: float,
) -> dict:
    if pose_init is not None:
        estimator.cfg["refine_iter"] = 1

    img = imread(input_image_path)

    pose_pr, _ = estimator.predict(img, K_matrix, pose_init=pose_init)
    pts, _ = project_points(obj_bbox_3d, pose_pr, K_matrix)

    hist_pts.append(pts)
    pts_ = weighted_pts(hist_pts, weight_num=smoothing_num, std_inv=smoothing_std)
    pose_ = pnp(obj_bbox_3d, pts_, K_matrix)  # Smoothed pose
    pts__, _ = project_points(
        obj_bbox_3d, pose_, K_matrix
    )  # 2D points of the smoothed pose

    return {
        "pose": pose_,
        "pts": pts__,
    }


def draw(
    input_image_path: Path, output_image_path: Path, pts: np.ndarray, save: bool = True
) -> np.ndarray:
    img = imread(input_image_path)
    bbox_img = draw_bbox_3d(img, pts, (0, 0, 255))
    if save:
        imsave(output_image_path, bbox_img)
    return bbox_img

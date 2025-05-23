from pathlib import Path
from skimage.io import imread, imsave
import cv2
import numpy as np
import warnings
import env

from utils.pose_utils import pose_apply

logger = env.logger

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_3d_point_on_plane(point_2d, K, plane_point_cam, plane_normal_cam):
    """Calculates the 3D point in camera coordinates by intersecting a ray with a plane."""
    inv_K = np.linalg.inv(K)
    point_norm = inv_K @ [point_2d[0], point_2d[1], 1.0]
    ray_dir = point_norm / np.linalg.norm(point_norm)

    denom = np.dot(plane_normal_cam, ray_dir)
    if abs(denom) < 1e-6:  # Avoid division by zero or near-zero
        logger.warning("Ray is parallel or near-parallel to the plane.")
        return None  # Ray is parallel to the plane

    t = np.dot(plane_normal_cam, plane_point_cam) / denom
    if t < 0:
        logger.warning("Plane intersection is behind the camera.")
        return None  # Intersection is behind the camera

    point_3d_cam = t * ray_dir
    return point_3d_cam


def predict(
    pose: np.ndarray,
    K_matrix: np.ndarray,
    min_pt: np.ndarray,
    max_pt: np.ndarray,
    ruler_pt1_2d: np.ndarray,
    ruler_pt2_2d: np.ndarray,
    ruler_real_length_cm: float,
    obj_name: str,
):
    logger.debug(
        f"Using pre-computed ruler info: endpoint1={ruler_pt1_2d}, endpoint2={ruler_pt2_2d}"
    )

    # Calculate ruler_pixel_length from the two endpoints
    ruler_pixel_length = np.linalg.norm(ruler_pt1_2d - ruler_pt2_2d)
    if ruler_pixel_length <= 1e-6:  # Use a small epsilon for floating point comparison
        logger.error(
            "Ruler endpoints are too close or identical, resulting in zero pixel length."
        )
        return {
            "name": obj_name,
            "width": None,
            "height": None,
            "depth": None,
        }
    logger.debug(f"Calculated ruler pixel length: {ruler_pixel_length:.2f}")

    obj_center_local = (min_pt + max_pt) / 2.0
    plane_point_cam = pose_apply(pose, obj_center_local)
    plane_normal_cam = pose[:3, 2]

    logger.debug("Calculating 3D ruler points...")
    ruler_pt1_3d_cam = get_3d_point_on_plane(
        ruler_pt1_2d, K_matrix, plane_point_cam, plane_normal_cam
    )
    ruler_pt2_3d_cam = get_3d_point_on_plane(
        ruler_pt2_2d, K_matrix, plane_point_cam, plane_normal_cam
    )

    if ruler_pt1_3d_cam is None or ruler_pt2_3d_cam is None:
        logger.error("Could not determine 3D ruler points on the object plane.")
        return {
            "name": obj_name,
            "width": None,
            "height": None,
            "depth": None,
        }

    logger.debug(
        f"Estimated Ruler 3D points (camera coords): {ruler_pt1_3d_cam}, {ruler_pt2_3d_cam}"
    )

    ruler_dist_3d_cam = np.linalg.norm(ruler_pt1_3d_cam - ruler_pt2_3d_cam)
    logger.debug(
        f"Calculated 3D distance for ruler (camera units): {ruler_dist_3d_cam}"
    )

    if ruler_dist_3d_cam < 1e-6:
        logger.error("Calculated 3D ruler distance is zero or near-zero.")
        return {
            "name": obj_name,
            "width": None,
            "height": None,
            "depth": None,
        }

    units_per_cm = ruler_dist_3d_cam / ruler_real_length_cm
    logger.debug(f"Scale factor: {units_per_cm:.4f} camera units per cm")

    depth_units = max_pt[0] - min_pt[0]
    width_units = max_pt[1] - min_pt[1]
    height_units = max_pt[2] - min_pt[2]

    real_width_cm = width_units / units_per_cm
    real_height_cm = height_units / units_per_cm
    real_depth_cm = depth_units / units_per_cm

    return {
        "name": obj_name,
        "width": real_width_cm,
        "height": real_height_cm,
        "depth": real_depth_cm,
    }


def draw(
    input_image_path: Path,
    output_image_path: Path,
    sizes: list,
    save: bool,
) -> np.ndarray:
    img = imread(input_image_path)
    idx = 0
    for size in sizes:
        if size["width"] is None or size["height"] is None or size["depth"] is None:
            text_lines = [
                f"Object: {size['name']}",
                "Width: N/A",
                "Height: N/A",
                "Depth: N/A",
            ]
        else:
            text_lines = [
                f"Object: {size['name']}",
                f"Width: {size['width']:.1f} cm",
                f"Height: {size['height']:.1f} cm",
                f"Depth: {size['depth']:.1f} cm",
            ]
        y_offset = 30
        for i, line in enumerate(text_lines):
            cv2.putText(
                img,
                line,
                (10, y_offset + i * 25 + idx * 150),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        idx += 1
    if save:
        imsave(output_image_path, img)
    return img

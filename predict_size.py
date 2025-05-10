import cv2
from skimage.io import imsave, imread
from pathlib import Path
import numpy as np
import warnings

from utils.pose_utils import pose_apply
from utils.base_utils import project_points
from utils.draw_utils import draw_bbox_3d, pts_range_to_bbox_pts

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_3d_point_on_plane(point_2d, K, plane_point_cam, plane_normal_cam):
    """Calculates the 3D point in camera coordinates by intersecting a ray with a plane."""
    inv_K = np.linalg.inv(K)
    point_norm = inv_K @ [point_2d[0], point_2d[1], 1.0]
    ray_dir = point_norm / np.linalg.norm(point_norm)

    denom = np.dot(plane_normal_cam, ray_dir)
    if abs(denom) < 1e-6:  # Avoid division by zero or near-zero
        print("Warning: Ray is parallel or near-parallel to the plane.")
        return None  # Ray is parallel to the plane

    t = np.dot(plane_normal_cam, plane_point_cam) / denom
    if t < 0:
        print("Warning: Plane intersection is behind the camera.")
        return None  # Intersection is behind the camera

    point_3d_cam = t * ray_dir
    return point_3d_cam


def predict_size(
    input_image_path: Path,
    output_image_path: Path,
    pose_pr: np.ndarray,
    K_matrix: np.ndarray,
    min_pt: np.ndarray,
    max_pt: np.ndarray,
    ruler_center_2d: np.ndarray,
    ruler_pixel_length: float,
    ruler_real_length_cm: float,
    obj_args_name: str,
    idx: int,
):
    img = imread(input_image_path)
    print(
        f"Using pre-computed ruler info: center={ruler_center_2d}, px_length={ruler_pixel_length:.2f}"
    )

    if ruler_pixel_length <= 0:
        print("Error: Ruler prediction has non-positive pixel length.")
        imsave(output_image_path, img)
        print(f"Saved current image to {output_image_path} due to ruler error.")
        return output_image_path

    cv2.circle(img, tuple(ruler_center_2d), 5, (255, 0, 0), -1)

    half_length_vec = np.array([ruler_pixel_length / 2.0, 0.0])
    ruler_pt1_2d = ruler_center_2d - half_length_vec
    ruler_pt2_2d = ruler_center_2d + half_length_vec
    print(f"Estimated horizontal ruler endpoints (2D): {ruler_pt1_2d}, {ruler_pt2_2d}")

    cv2.line(
        img,
        tuple(ruler_pt1_2d.astype(int)),
        tuple(ruler_pt2_2d.astype(int)),
        (255, 255, 0),
        2,
    )

    obj_center_local = (min_pt + max_pt) / 2.0
    plane_point_cam = pose_apply(pose_pr, obj_center_local)
    plane_normal_cam = pose_pr[:3, 2]

    print("Calculating 3D ruler points...")
    ruler_pt1_3d_cam = get_3d_point_on_plane(
        ruler_pt1_2d, K_matrix, plane_point_cam, plane_normal_cam
    )
    ruler_pt2_3d_cam = get_3d_point_on_plane(
        ruler_pt2_2d, K_matrix, plane_point_cam, plane_normal_cam
    )

    if ruler_pt1_3d_cam is None or ruler_pt2_3d_cam is None:
        print("Error: Could not determine 3D ruler points on the object plane.")
        object_bbox_3d = pts_range_to_bbox_pts(max_pt, min_pt)
        pts_bbox_2d, _ = project_points(object_bbox_3d, pose_pr, K_matrix)
        img = draw_bbox_3d(img, pts_bbox_2d, (0, 0, 255))
        imsave(output_image_path, img)
        print(f"Saved image with available annotations to {output_image_path}")
        return output_image_path

    print(
        f"Estimated Ruler 3D points (camera coords): {ruler_pt1_3d_cam}, {ruler_pt2_3d_cam}"
    )

    ruler_dist_3d_cam = np.linalg.norm(ruler_pt1_3d_cam - ruler_pt2_3d_cam)
    print(f"Calculated 3D distance for ruler (camera units): {ruler_dist_3d_cam}")

    if ruler_dist_3d_cam < 1e-6:
        print("Error: Calculated 3D ruler distance is zero or near-zero.")
        object_bbox_3d = pts_range_to_bbox_pts(max_pt, min_pt)
        pts_bbox_2d, _ = project_points(object_bbox_3d, pose_pr, K_matrix)
        img = draw_bbox_3d(img, pts_bbox_2d, (0, 0, 255))
        imsave(output_image_path, img)
        print(f"Saved image with available annotations to {output_image_path}")
        return output_image_path

    units_per_cm = ruler_dist_3d_cam / ruler_real_length_cm
    print(f"Scale factor: {units_per_cm:.4f} camera units per cm")

    width_units = max_pt[0] - min_pt[0]
    height_units = max_pt[1] - min_pt[1]
    depth_units = max_pt[2] - min_pt[2]

    real_width_cm = width_units / units_per_cm
    real_height_cm = height_units / units_per_cm
    real_depth_cm = depth_units / units_per_cm

    object_bbox_3d = pts_range_to_bbox_pts(max_pt, min_pt)
    pts_bbox_2d, _ = project_points(object_bbox_3d, pose_pr, K_matrix)
    img = draw_bbox_3d(img, pts_bbox_2d, (0, 0, 255))

    text_lines = [
        f"Object: {obj_args_name}",
        f"Width: {real_width_cm:.1f} cm",
        f"Height: {real_height_cm:.1f} cm",
        f"Depth: {real_depth_cm:.1f} cm",
    ]
    y_offset = 30
    for i, line in enumerate(text_lines):
        cv2.putText(
            img,
            line,
            (10, y_offset + i * 25 + idx * 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            line,
            (10, y_offset + i * 25 + idx * 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    imsave(output_image_path, img)
    print(f"Saving visualization with dimensions to: {output_image_path}")

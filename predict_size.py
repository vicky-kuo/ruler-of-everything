import cv2
from pathlib import Path
from skimage.io import imsave, imread
import numpy as np
import warnings

import env
from ruler.predict import (
    predict as ruler_predict,
)  # Returns (center_point, pixel_length)
from predict_frame import predict as gen6d_predict
from utils.pose_utils import pose_apply
from utils.base_utils import project_points
from utils.draw_utils import draw_bbox_3d, pts_range_to_bbox_pts
from utils.other_utils import downsample_image

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


# --- Main Script ---
args = env.minion_args  # Or whichever args are appropriate
input_image_path = args.input_path
output_path = args.output_path
ruler_real_length_cm = 15.0

# 1. Downsample and run Gen6D prediction
downsample_image(
    input_image_path,
)
img_display = imread(input_image_path)
img_height, img_width = img_display.shape[:2]

print("Running Gen6D prediction...")
(
    _,  # Discard center and Gen6D dimensions
    pose_pr,
    K,
    min_pt,
    max_pt,
) = gen6d_predict(args)
print("Gen6D prediction complete.")

# 2. Run ruler prediction
print("Running ruler prediction (center and pixel length)...")
try:
    # Assuming ruler_predict returns (center_point, pixel_length)
    ruler_info = ruler_predict(args)
    if ruler_info is None or len(ruler_info) != 2:
        raise ValueError("Ruler prediction did not return center and pixel length.")
    ruler_center_2d = np.array(ruler_info[0], dtype=float)
    ruler_pixel_length = float(ruler_info[1])

    if ruler_pixel_length <= 0:
        raise ValueError("Ruler prediction returned non-positive pixel length.")

    print(f"Ruler center: {ruler_center_2d}, Pixel length: {ruler_pixel_length:.2f}")

    # Draw ruler center for verification
    cv2.circle(
        img_display, tuple(ruler_center_2d.astype(int)), 5, (255, 0, 0), -1
    )  # Blue dot at center

    # --- Estimate ruler 2D endpoints based on center and length (assuming horizontal) ---
    # !! This is an approximation. Assumes ruler is mostly horizontal in the image !!
    # !! A better approach would determine the ruler's 2D orientation !!
    half_length_vec = np.array([ruler_pixel_length / 2.0, 0.0])
    ruler_pt1_2d = ruler_center_2d - half_length_vec
    ruler_pt2_2d = ruler_center_2d + half_length_vec
    print(f"Estimated horizontal ruler endpoints (2D): {ruler_pt1_2d}, {ruler_pt2_2d}")

    # Draw estimated ruler line for verification
    cv2.line(
        img_display,
        tuple(ruler_pt1_2d.astype(int)),
        tuple(ruler_pt2_2d.astype(int)),
        (255, 255, 0),
        2,
    )  # Cyan line

except Exception as e:
    print(f"Error during ruler prediction or processing: {e}")
    print("Cannot calculate real dimensions without valid ruler info.")
    # Optionally save the image with Gen6D bbox even if ruler fails
    object_bbox_3d = pts_range_to_bbox_pts(max_pt, min_pt)
    pts_bbox_2d, _ = project_points(object_bbox_3d, pose_pr, K)
    img_display = draw_bbox_3d(img_display, pts_bbox_2d, (0, 0, 255))
    output_file_path = (
        Path(output_path) / f"{Path(input_image_path).stem}-bbox-only.jpg"
    )
    output_file_path.parent.mkdir(exist_ok=True, parents=True)
    imsave(str(output_file_path), img_display)
    print(f"Saved image with Gen6D bounding box only to {output_file_path}")
    exit()

# 3. Define object plane in camera coordinates
obj_center_local = (min_pt + max_pt) / 2.0
plane_point_cam = pose_apply(pose_pr, obj_center_local)
# Plane normal is the Z-axis of the object's coordinate system transformed by the pose's rotation.
plane_normal_cam = pose_pr[:3, 2]  # Z-axis of object pose rotation

# 4. Find 3D ruler points on the plane (using estimated 2D endpoints)
print("Calculating 3D ruler points based on estimated 2D endpoints...")
ruler_pt1_3d_cam = get_3d_point_on_plane(
    ruler_pt1_2d, K, plane_point_cam, plane_normal_cam
)
ruler_pt2_3d_cam = get_3d_point_on_plane(
    ruler_pt2_2d, K, plane_point_cam, plane_normal_cam
)

if ruler_pt1_3d_cam is None or ruler_pt2_3d_cam is None:
    print(
        "Could not determine 3D coordinates for estimated ruler endpoints on the object plane."
    )
    print("Cannot calculate real dimensions.")
    # Optionally save the image with Gen6D bbox and ruler points
    object_bbox_3d = pts_range_to_bbox_pts(max_pt, min_pt)
    pts_bbox_2d, _ = project_points(object_bbox_3d, pose_pr, K)
    img_display = draw_bbox_3d(img_display, pts_bbox_2d, (0, 0, 255))
    output_file_path = (
        Path(output_path) / f"{Path(input_image_path).stem}-no-3d-ruler.jpg"
    )
    output_file_path.parent.mkdir(exist_ok=True, parents=True)
    imsave(str(output_file_path), img_display)
    print(f"Saved image with Gen6D bounding box and 2D ruler to {output_file_path}")
    exit()
print(
    f"Estimated Ruler 3D points (camera coords): {ruler_pt1_3d_cam}, {ruler_pt2_3d_cam}"
)

# 5. Calculate 3D ruler distance in camera coordinates
ruler_dist_3d_cam = np.linalg.norm(ruler_pt1_3d_cam - ruler_pt2_3d_cam)
print(f"Calculated 3D distance for ruler (camera units): {ruler_dist_3d_cam}")

if ruler_dist_3d_cam < 1e-6:
    print("Error: Calculated 3D ruler distance is zero or near-zero.")
    print("Cannot calculate scale factor.")
    # Optionally save the image
    object_bbox_3d = pts_range_to_bbox_pts(max_pt, min_pt)
    pts_bbox_2d, _ = project_points(object_bbox_3d, pose_pr, K)
    img_display = draw_bbox_3d(img_display, pts_bbox_2d, (0, 0, 255))
    output_file_path = (
        Path(output_path) / f"{Path(input_image_path).stem}-zero-dist.jpg"
    )
    output_file_path.parent.mkdir(exist_ok=True, parents=True)
    imsave(str(output_file_path), img_display)
    print(f"Saved image with Gen6D bounding box and 2D ruler to {output_file_path}")
    exit()

# 6. Calculate scale factor (camera units per cm)
units_per_cm = ruler_dist_3d_cam / ruler_real_length_cm
print(f"Scale factor: {units_per_cm:.4f} camera units per cm")

# 7. Calculate object dimensions in local units
width_units = max_pt[0] - min_pt[0]
height_units = max_pt[1] - min_pt[1]
depth_units = max_pt[2] - min_pt[2]

# 8. Convert to real-world cm using the scale factor
real_width_cm = width_units / units_per_cm
real_height_cm = height_units / units_per_cm
real_depth_cm = depth_units / units_per_cm

# 9. Output results
print("\n--- Calculated Real-World Object Dimensions ---")
print(f"Width:  {real_width_cm:.2f} cm")
print(f"Height: {real_height_cm:.2f} cm")
print(f"Depth:  {real_depth_cm:.2f} cm")
print("(Based on assumed horizontal ruler orientation in image)")
print("-----------------------------------------------")

# Optional: Draw results on the image
# Draw Gen6D BBox
object_bbox_3d = pts_range_to_bbox_pts(max_pt, min_pt)
pts_bbox_2d, _ = project_points(object_bbox_3d, pose_pr, K)
img_display = draw_bbox_3d(img_display, pts_bbox_2d, (0, 0, 255))  # Blue bbox

# Add text with dimensions
text_lines = [
    f"W: {real_width_cm:.1f} cm",
    f"H: {real_height_cm:.1f} cm",
    f"D: {real_depth_cm:.1f} cm",
]
y_offset = 30
for i, line in enumerate(text_lines):
    cv2.putText(
        img_display,
        line,
        (10, y_offset + i * 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )  # White text w/ black outline
    cv2.putText(
        img_display,
        line,
        (10, y_offset + i * 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )  # Blue text

output_file_path = Path(output_path) / f"{Path(input_image_path).stem}-dimensions.jpg"
output_file_path.parent.mkdir(exist_ok=True, parents=True)
print(f"Saving visualization with dimensions to: {output_file_path}")
imsave(str(output_file_path), img_display)

print("Processing finished.")

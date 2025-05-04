from utils.other_utils import downsample_image
import warnings
import env
from predict_frame import predict as gen6d_predict
import cv2
import numpy as np
import open3d as o3d
from skimage.io import imread, imsave
import os
import transforms3d
from utils.base_utils import project_points
from utils.pose_utils import pose_apply

warnings.filterwarnings("ignore", category=UserWarning)

args = env.girl_args

downsample_image(args.input_path)

(
    center,
    pose_target,  # Renamed pose_pr to pose_target for clarity
    K_target,  # Renamed K to K_target
    min_pt_target,  # Renamed min_pt to min_pt_target
    max_pt_target,  # Renamed max_pt to max_pt_target
) = gen6d_predict(args)

print("AR Rendering started...")

# 1. Load Input Image
img = imread(args.input_path)
h, w, _ = img.shape
img_render = img.copy()  # Create a copy to draw on

# 2. Load Can Model
can_model_path = "input_pictures/soda.obj"
try:
    can_mesh = o3d.io.read_triangle_mesh(can_model_path)
    if not can_mesh.has_vertices() or not can_mesh.has_triangles():
        raise ValueError("Mesh is empty or invalid.")
    can_vertices = np.asarray(can_mesh.vertices)
    can_triangles = np.asarray(can_mesh.triangles)
    print(
        f"Loaded can model '{can_model_path}' with {len(can_vertices)} vertices and {len(can_triangles)} triangles."
    )
except Exception as e:
    print(f"Error loading can model: {e}")
    exit()

# 3. Calculate Target Object Dimensions and Axis
target_dims = max_pt_target - min_pt_target
# User specified target longest axis is Z (index 2)
target_longest_axis_index = 2
target_longest_dim = target_dims[target_longest_axis_index]
target_axis_vector = np.array([0, 0, 1])  # Z-axis
print(f"Target object dimensions: {target_dims}, Longest (Z): {target_longest_dim}")

# 4. Calculate Can Model Dimensions and Axis
can_min_pt = np.min(can_vertices, axis=0)
can_max_pt = np.max(can_vertices, axis=0)
can_dims = can_max_pt - can_min_pt
# User specified can longest axis is Y (index 1)
can_longest_axis_index = 1
can_longest_dim = can_dims[can_longest_axis_index]
can_axis_vector = np.array([0, 1, 0])  # Y-axis
print(f"Can model dimensions: {can_dims}, Longest (Y): {can_longest_dim}")

if can_longest_dim <= 1e-6:
    print("Error: Can model longest dimension is zero or negative.")
    exit()

# 5. Calculate Alignment Rotation (Can Y-axis to Target Z-axis)
# Rotation around X-axis by -90 degrees aligns Y with Z
R_align = transforms3d.axangles.axangle2mat([1, 0, 0], -np.pi / 2)
print("Calculated alignment rotation.")

# 6. Calculate Scaling Factor
scale = target_longest_dim / can_longest_dim
print(f"Calculated scale factor: {scale}")

# 7. Transform Can Vertices
# Center the can vertices
can_center = (can_min_pt + can_max_pt) / 2
centered_vertices = can_vertices - can_center
# Apply scale
scaled_vertices = centered_vertices * scale
# Apply alignment rotation (vertices are N x 3, R is 3 x 3, use V_new = V_old @ R.T)
aligned_vertices = scaled_vertices @ R_align.T
# aligned_vertices are now centered around (0,0,0), scaled, and rotated
# so the original Y axis aligns with the Z axis.

# Calculate center of the target object in its local coordinates
center_target_local = (min_pt_target + max_pt_target) / 2
print(f"Target object center (local coords): {center_target_local}")

# Shift the aligned can vertices by the target's local center offset.
# This prepares them for the pose_target transformation, assuming pose_target
# transforms from the target's local origin.
vertices_to_transform = aligned_vertices + center_target_local

# Apply target pose (R_target, t_target)
# R_target rotates, t_target translates the target's local origin to camera space.
# Applying this to (aligned_vertices + center_target_local) should place
# the can's center correctly at the target's center in camera space.
transformed_vertices = pose_apply(pose_target, vertices_to_transform)
print("Transformed can vertices (adjusted for target center).")

# 8. Project Can Vertices onto Image Plane
# Use identity pose as vertices are already in camera coordinates
identity_pose = np.hstack([np.eye(3), np.zeros((3, 1))])
projected_vertices, depths = project_points(
    transformed_vertices, identity_pose, K_target
)
print("Projected can vertices.")

# 9. Render Wireframe
wireframe_color = (0, 255, 0)  # Green color for wireframe
thickness = 1  # Line thickness

for triangle in can_triangles:
    # Get the 3 vertices for this triangle
    p1_idx, p2_idx, p3_idx = triangle
    p1 = projected_vertices[p1_idx]
    p2 = projected_vertices[p2_idx]
    p3 = projected_vertices[p3_idx]

    # Get depths to check if points are in front of the camera
    d1 = depths[p1_idx]
    d2 = depths[p2_idx]
    d3 = depths[p3_idx]

    # Only draw lines if both endpoints are in front of the camera and within image bounds (simple check)
    points = [(p1, d1), (p2, d2), (p3, d3)]
    edges = [(0, 1), (1, 2), (2, 0)]  # Edges of the triangle

    for i, j in edges:
        pt_a, depth_a = points[i]
        pt_b, depth_b = points[j]

        if depth_a > 0 and depth_b > 0:  # Check if points are in front of camera
            # Convert to integer coordinates for drawing
            pt_a_int = (int(round(pt_a[0])), int(round(pt_a[1])))
            pt_b_int = (int(round(pt_b[0])), int(round(pt_b[1])))
            # Ensure the y-coordinate of the point is positive
            pt_a_int = (pt_a_int[0], abs(pt_a_int[1]))
            pt_b_int = (pt_b_int[0], abs(pt_b_int[1]))
            print(f"Drawing line from {pt_a_int} to {pt_b_int}")

            cv2.line(
                img_render,
                pt_a_int,
                pt_b_int,
                wireframe_color,
                thickness,
                cv2.LINE_AA,
            )

print("Rendered wireframe.")

# 10. Save Output Image
output_dir = args.output_path
os.makedirs(output_dir, exist_ok=True)
input_basename = os.path.basename(args.input_path)
output_filename = os.path.join(
    output_dir, f"{os.path.splitext(input_basename)[0]}-ar.jpg"
)
imsave(output_filename, img_render)
print(f"Saved AR output image to: {output_filename}")

print("AR Rendering finished.")

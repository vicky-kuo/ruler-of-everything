import warnings
import numpy as np
import pyvista as pv
from pathlib import Path
from skimage.io import imread
import os
import transforms3d
import math
from utils.pose_utils import pose_apply


def caculate_axis(min_pt: np.ndarray, max_pt: np.ndarray):
    dims = max_pt - min_pt
    longest_axis_index = np.argmax(dims)
    longest_dim = dims[longest_axis_index]
    axis_vector = np.zeros(3)
    axis_vector[longest_axis_index] = 1
    return dims, longest_dim, axis_vector


def render(
    input_image_path: Path,
    output_image_path: Path,
    model_args,
    pose_target,
    K_matrix,
    target_min_pt,
    target_max_pt,
):
    """
    Renders a 3D model onto an image using AR principles.

    Args:
        object_args: Arguments related to the target object scene (not directly used for paths here).
        model_args: Arguments and paths related to the 3D model to render.
        pose_target: Pose of the target object in the scene.
        K_target: Camera intrinsics for the target scene.
        target_min_pt: Minimum point of the target object's bounding box.
        target_max_pt: Maximum point of the target object's bounding box.
        input_image_path: Path to the background image to render on.
        output_image_path: Path to save the final AR image.

    Returns:
        str: The path to the saved output image.
    """

    img = imread(input_image_path)
    h, w, _ = img.shape

    model_mesh: pv.DataSet = pv.read(model_args.obj_path)
    if model_mesh.n_points == 0 or model_mesh.n_cells == 0:
        raise ValueError("Mesh is empty or invalid.")
    model_vertices_original = model_mesh.points.copy()
    model_faces = model_mesh.faces.copy()
    print(
        f"Loaded model '{model_args.obj_path}' with {len(model_vertices_original)} vertices and {len(model_faces)} faces"
    )

    # Calculate Target Object Dimensions and Axis
    target_dims, target_longest_dim, _ = caculate_axis(target_min_pt, target_max_pt)
    print(
        f"Target object dimensions: {target_dims}, longest dimension: {target_longest_dim}"
    )

    # Calculate Model Dimensions and Axis
    model_min_pt = np.min(model_vertices_original, axis=0)
    model_max_pt = np.max(model_vertices_original, axis=0)
    model_dims, model_longest_dim, _ = caculate_axis(model_min_pt, model_max_pt)
    print(f"Model dimensions: {model_dims}, longest dimension: {model_longest_dim}")

    # Calculate Alignment Rotation
    R_align = transforms3d.axangles.axangle2mat(
        model_args.rotate_axis, model_args.rotation
    )

    # Calculate Scaling Factor
    scale = target_longest_dim / model_longest_dim
    print(f"Calculated scale factor: {scale}")

    model_center_original = (model_min_pt + model_max_pt) / 2
    center_target_local = (target_min_pt + target_max_pt) / 2

    def transform_vertices_local(vertices_to_transform_input):
        centered_vertices = vertices_to_transform_input - model_center_original
        scaled_vertices = centered_vertices * scale
        aligned_vertices = scaled_vertices @ R_align.T
        vertices_to_align_with_target = aligned_vertices + center_target_local
        transformed_v = pose_apply(pose_target, vertices_to_align_with_target)
        return transformed_v

    # Create PyVista Plotter and Set Up Scene
    plotter = pv.Plotter(off_screen=True, window_size=[w, h])
    plotter.add_background_image(input_image_path)

    # Read MTL file
    texture_paths = []
    mtl_base_dir = os.path.dirname(model_args.mtl_path)

    if os.path.exists(model_args.mtl_path):
        with open(model_args.mtl_path) as mtl_file:
            for line in mtl_file.readlines():
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                if (
                    parts[0] == "map_Kd"
                    or parts[0] == "map_Ka"
                    or parts[0] == "norm"
                    or parts[0] == "map_Pm"
                ):
                    texture_file = parts[1]
                    if not os.path.isabs(texture_file):
                        texture_file = os.path.join(mtl_base_dir, texture_file)
                    texture_paths.append(texture_file)
    else:
        print(f"Warning: MTL file not found at {model_args.mtl_path}")

    if (
        "MaterialIds" not in model_mesh.cell_data
        or model_mesh.cell_data["MaterialIds"] is None
    ):
        transformed_vertices = transform_vertices_local(model_vertices_original)
        model_polydata_transformed = pv.PolyData(
            transformed_vertices, faces=model_faces
        )

        plotter.add_mesh(model_polydata_transformed, smooth_shading=True)

    else:
        material_ids = model_mesh.cell_data["MaterialIds"]
        unique_material_ids = np.unique(material_ids)

        for current_material_id in unique_material_ids:
            mesh_part = model_mesh.extract_cells(material_ids == current_material_id)
            if mesh_part.n_points == 0 or mesh_part.n_cells == 0:
                print(
                    f"  Skipping empty mesh part for Material ID: {current_material_id}"
                )
                continue

            part_original_vertices = mesh_part.points
            mesh_part.points = transform_vertices_local(part_original_vertices)
            mesh_part_texture = None

            if 0 <= current_material_id < len(texture_paths):
                texture_file_path = texture_paths[current_material_id]
                print(
                    f"  Attempting to load texture: {texture_file_path} for material index {current_material_id}"
                )
                mesh_part_texture = pv.read_texture(texture_file_path)
            else:
                print(
                    f"  Warning: Material index {current_material_id} is out of bounds for texture_paths (len: {len(texture_paths)}) or no textures defined."
                )
            plotter.add_mesh(
                mesh_part,
                smooth_shading=True,
                texture=mesh_part_texture,
            )
            if not mesh_part_texture:
                print(
                    f"  Added mesh part for material index {current_material_id} without texture."
                )

    fy = K_matrix[1, 1]
    if fy <= 0:
        print("Error: Invalid focal length fy in K_target. Using a default.")
        fov_y = 50
    else:
        fov_y = 2 * math.atan(h / (2 * fy)) * 180 / math.pi

    cx = K_matrix[0, 2]
    cy = K_matrix[1, 2]
    window_center_x = (cx / w) * 2 - 1
    window_center_y = ((h - cy) / h) * 2 - 1

    plotter.camera.position = (0, 0, 0)
    plotter.camera.focal_point = (0, 0, 1)
    plotter.camera.up = (0, -1, 0)
    plotter.camera.view_angle = fov_y
    plotter.camera.window_center = (window_center_x, window_center_y)

    print(
        f"Configured PyVista camera: FoV={fov_y:.2f}, WindowCenter=({window_center_x:.3f}, {window_center_y:.3f})"
    )

    plotter.screenshot(output_image_path, transparent_background=False)
    print(f"Saved AR output image to: {output_image_path}")

    plotter.close()


warnings.filterwarnings("ignore", category=UserWarning)

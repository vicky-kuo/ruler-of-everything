from utils.other_utils import downsample_image
import warnings
import env
from predict_frame import predict as gen6d_predict
import numpy as np
import pyvista as pv
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


warnings.filterwarnings("ignore", category=UserWarning)

args = env.minion_args

model = env.soda_can_model

downsample_image(args.input_path)

(
    center,
    pose_target,
    K_target,
    target_min_pt,
    target_max_pt,
) = gen6d_predict(args)

print("AR Rendering started...")

# 1. Load Input Image
img = imread(args.input_path)
h, w, _ = img.shape
print(f"Loaded background image with dimensions: {w}x{h}")

# 2. Load Model

try:
    model_mesh: pv.DataSet = pv.read(model.obj_path)
    if model_mesh.n_points == 0 or model_mesh.n_cells == 0:
        raise ValueError("Mesh is empty or invalid.")

    model_vertices = model_mesh.points
    model_faces = model_mesh.faces

    print(
        f"Loaded model '{model.obj_path}' with {len(model_vertices)} vertices and {len(model_faces)} faces"
    )
except Exception as e:
    print(f"Error loading or processing model: {e}")
    exit()

# 3. Calculate Target Object Dimensions and Axis
target_dims, target_longest_dim, target_axis_vector = caculate_axis(
    target_min_pt, target_max_pt
)
print(
    f"Target object dimensions: {target_dims}, Longest dimension: {target_longest_dim}"
)

# 4. Calculate Model Dimensions and Axis
model_min_pt = np.min(model_vertices, axis=0)
model_max_pt = np.max(model_vertices, axis=0)
model_dims, model_longest_dim, model_axis_vector = caculate_axis(
    model_min_pt, model_max_pt
)
print(f"model dimensions: {model_dims}, Longest dimension: {model_longest_dim}")

# 5. Calculate Alignment Rotation
R_align = transforms3d.axangles.axangle2mat(model.rotate_axis, model.rotation)

# 6. Calculate Scaling Factor
scale = target_longest_dim / model_longest_dim
print(f"Calculated scale factor: {scale}")

# 7. Transform Model Vertices
model_center = (model_min_pt + model_max_pt) / 2
centered_vertices = model_vertices - model_center
scaled_vertices = centered_vertices * scale
aligned_vertices = scaled_vertices @ R_align.T
center_target_local = (target_min_pt + target_max_pt) / 2
vertices_to_transform = aligned_vertices + center_target_local
transformed_vertices = pose_apply(pose_target, vertices_to_transform)
print("Transformed model vertices (adjusted for target center).")

# 8. Create PyVista Plotter and Set Up Scene
plotter = pv.Plotter(off_screen=True, window_size=[w, h])
plotter.add_background_image(args.input_path)


# Read MTL file
texture_paths = []
mtl_names = []
with open(model.mtl_path) as mtl_file:
    for line in mtl_file.readlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        if parts[0] == "map_Kd" or parts[0] == "map_Ka" or parts[0] == "norm":
            texture_paths.append(os.path.join(model.input_path, parts[1]))
        elif parts[0] == "newmtl":
            mtl_names.append(parts[1])

material_ids = model_mesh.cell_data["MaterialIds"]
print(f"texture_paths: {texture_paths}")

unique_material_ids = np.unique(material_ids)

if unique_material_ids is None or len(unique_material_ids) == 0:
    print(
        "Warning: No material IDs found in mesh. Attempting to add the whole transformed mesh as a single piece."
    )
    model_polydata_fallback = pv.PolyData(transformed_vertices, faces=model_faces)

    plotter.add_mesh(model_polydata_fallback, smooth_shading=True)
else:
    for mat_id_value in unique_material_ids:
        current_material_id_int = int(
            mat_id_value
        )  # Ensure it's an integer for indexing and comparison
        print(f"Processing Material ID: {current_material_id_int}")

        # Your original skip condition for material ID 1
        # if current_material_id_int == 1:
        #     print(f"  Skipping Material ID {current_material_id_int} as per original logic.")
        #     continue

        # Extract the part of the mesh corresponding to the current material ID
        # The extract_cells method might return an UnstructuredGrid
        mesh_part: pv.UnstructuredGrid = model_mesh.extract_cells(
            material_ids == current_material_id_int
        )

        if mesh_part.n_points == 0 or mesh_part.n_cells == 0:
            print(
                f"  Skipping empty mesh part for Material ID: {current_material_id_int}"
            )
            continue

        print(
            f"  Extracted mesh part: {mesh_part.n_points} points, {mesh_part.n_cells} cells."
        )

        # Get the original vertices of this specific mesh part
        part_original_vertices = mesh_part.points

        # Apply the same sequence of transformations to these part_original_vertices
        # model_center, scale, R_align, center_target_local, pose_target are defined in the global scope of this script
        centered_part_vertices = part_original_vertices - model_center
        scaled_part_vertices = centered_part_vertices * scale
        aligned_part_vertices = scaled_part_vertices @ R_align.T
        part_vertices_to_transform = aligned_part_vertices + center_target_local
        transformed_part_mesh_vertices = pose_apply(
            pose_target, part_vertices_to_transform
        )

        # Update the points of the mesh_part in-place with the transformed vertices
        # The mesh_part's faces and texture coordinates are already relative to its points
        mesh_part.points = transformed_part_mesh_vertices

        # Handle texture for this part
        # Material IDs are typically 0-indexed and correspond to the order of materials/textures
        mesh_part_texture = None
        if 0 <= current_material_id_int < len(texture_paths):
            try:
                texture_file_path = texture_paths[current_material_id_int]
                print(
                    f"  Loading texture: {texture_file_path} for material index {current_material_id_int}"
                )
                mesh_part_texture = pv.read_texture(texture_file_path)
            except Exception as e:
                print(
                    f"  Error loading texture {texture_paths[current_material_id_int]} for material index {current_material_id_int}: {e}"
                )
                print(
                    f"  Attempting to add mesh part for material index {current_material_id_int} without its specific texture."
                )
        else:
            print(
                f"  Warning: Material index {current_material_id_int} is out of bounds for texture_paths (len: {len(texture_paths)})."
            )
            print(
                f"  Adding mesh part for material index {current_material_id_int} without texture."
            )

        # Add the transformed mesh part (which now has updated points) to the plotter
        # Its original texture coordinates (mesh_part.active_texture_coordinates) will be used.
        plotter.add_mesh(
            mesh_part,
            smooth_shading=True,
            texture=mesh_part_texture,  # This will be None if texture loading failed or index was out of bounds
            # show_edges=False # Optional, uncomment if you don't want to see edges
        )
        if mesh_part_texture:
            print(
                f"  Added mesh part for material index {current_material_id_int} with texture."
            )
        else:
            print(
                f"  Added mesh part for material index {current_material_id_int} without texture (or texture load failed)."
            )

# 9. Configure PyVista Camera based on K_target
fy = K_target[1, 1]
if fy <= 0:
    print("Error: Invalid focal length fy in K_target.")
    exit()
fov_y = 2 * math.atan(h / (2 * fy)) * 180 / math.pi

cx = K_target[0, 2]
cy = K_target[1, 2]
window_center_x = (cx / w) * 2 - 1
window_center_y = (cy / h) * 2 - 1

plotter.camera.position = (0, 0, 0)
plotter.camera.focal_point = (0, 0, 1)
plotter.camera.up = (0, -1, 0)
plotter.camera.view_angle = fov_y
plotter.camera.window_center = (window_center_x, window_center_y)

print(
    f"Configured PyVista camera: FoV={fov_y:.2f}, WindowCenter=({window_center_x:.3f}, {window_center_y:.3f})"
)

# Save Output Image
output_dir = args.output_path
os.makedirs(output_dir, exist_ok=True)
input_basename = os.path.basename(args.input_path)
output_filename = os.path.join(
    output_dir, f"{os.path.splitext(input_basename)[0]}-ar.jpg"
)

plotter.screenshot(output_filename, transparent_background=False)
print(f"Saved AR output image to: {output_filename}")

plotter.close()

del plotter
del model_mesh

print("AR Rendering finished.")

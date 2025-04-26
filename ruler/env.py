import os

task = "segment"

dataset_path = os.path.join("dataset")
generate_dataset_path = os.path.join("generate_dataset")
data_path = os.path.join(dataset_path, "data.yaml")

input_path = os.path.join("input_pictures")
output_path = os.path.join("output_pictures")

# YOLO
yolo_model_version = "train"
taskFolder = os.path.join("runs", task)
model_folder = os.path.join("runs", task, yolo_model_version)
model_path = os.path.join(model_folder, "weights", "best.pt")

# Roboflow
api_key = "AuWUC9FXJwD3HYlOXO2l"
workspace = "size-estimation"
project_name = "ruler-of-everything"
roboflow_model_version = 1

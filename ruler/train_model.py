from roboflow import Roboflow
import subprocess
import env
import utils

rf = Roboflow(env.api_key)
project = rf.workspace(env.workspace).project(env.project_name)
version = project.version(env.roboflow_model_version)

# download dataset from roboflow (Step 1)
# dataset = version.download("yolov12", utils.get_absolute_path(env.dataset_path))

# train model (Step 2)
# model = YOLO("yolo12n.pt")
# results = model.train(data=utils.get_absolute_path(env.data_path), epochs=50, imgsz=640)
subprocess.run(
    [
        "yolo",
        env.task,
        "train",
        f"project={utils.get_absolute_path(env.taskFolder)}",  # "cache=True",
        f"data={utils.get_absolute_path(env.data_path)}",
        "epochs=100",
        "imgsz=640",
        "model=yolo11n-seg.pt",
        "device=0",
    ]
)  # f"model={utils.get_absolute_path(env.model_path)}",

# upload model to roboflow (Step 3)
# version.deploy(model_type="yolov8-cls", model_path=utils.get_absolute_path(env.model_folder))

from roboflow import Roboflow
import subprocess
import shutil
import env

rf = Roboflow(env.api_key)
project = rf.workspace(env.workspace).project(env.project_name)
version = project.version(env.roboflow_model_version)

# download dataset from roboflow (Step 1)
# shutil.rmtree(env.ruler_dataset_path, ignore_errors=True)
# dataset = version.download("yolov11", env.ruler_dataset_path)

# train model (Step 2)
# model = YOLO("yolo12n.pt")
# results = model.train(data=utils.get_absolute_path(env.data_path), epochs=50, imgsz=640)
subprocess.run(
    [
        "yolo",
        env.task,
        "train",
        f"project={env.ruler_task_path}",  # "cache=True",
        f"data={env.ruler_data_path}",
        "epochs=100",
        "imgsz=640",
        "model=yolo11n-seg.pt",
        "device=0",
    ]
)  # f"model={utils.get_absolute_path(env.model_path)}",

# upload model to roboflow (Step 3)
# version.deploy(model_type="yolov8-cls", model_path=utils.get_absolute_path(env.model_folder))

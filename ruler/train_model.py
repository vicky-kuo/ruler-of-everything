from roboflow import Roboflow
from ultralytics import YOLO
import subprocess
import os
import env
import utils

rf = Roboflow(env.apiKey)
project = rf.workspace(env.workspace).project(env.projectName)
version = project.version(env.roboflowModelVersion)

#download dataset from roboflow (Step 1)
#dataset = version.download("yolov12", utils.getAbsolutePath(env.datasetPath))

#train model (Step 2)
#model = YOLO("yolo12n.pt")
#results = model.train(data=utils.getAbsolutePath(env.dataPath), epochs=50, imgsz=640)
subprocess.run(["yolo", env.task, "train", f"data={utils.getAbsolutePath(env.dataPath)}", "epochs=100", "imgsz=640", "model=yolo11n-seg.pt"]) #f"model={utils.getAbsolutePath(env.modelPath)}",

#upload model to roboflow (Step 3)
#version.deploy(model_type="yolov8-cls", model_path=utils.getAbsolutePath(env.modelFolder))

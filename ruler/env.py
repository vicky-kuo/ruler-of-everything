import os
import math

task = "segment"


datasetPath = os.path.join("dataset")
generateDatasetPath = os.path.join("generate_dataset")
dataPath = os.path.join(datasetPath, "data.yaml")

inputPath = os.path.join("input_pictures")
outputPath = os.path.join("output_pictures")

# YOLO
yoloModelVersion = "train"
modelFolder = os.path.join("runs", task, yoloModelVersion)
modelPath = os.path.join(modelFolder, "weights", "best.pt")

# Roboflow
apiKey = "AuWUC9FXJwD3HYlOXO2l"
workspace = "size-estimation"
projectName = "ruler-of-everything"
roboflowModelVersion = 1
from ultralytics import YOLO
import numpy as np
import os
import cv2
import utils
import env

files = os.listdir(utils.getAbsolutePath(env.inputPath))
inputFiles = [utils.getAbsolutePath(os.path.join(env.inputPath, file)) for file in files]

model = YOLO(utils.getAbsolutePath(env.modelPath))
results = model(inputFiles, verbose=False)
i = 0

for result in results:  
    img = result.plot()

    print("output picture: " + result.path)
    cv2.imwrite(utils.getAbsolutePath(os.path.join(env.outputPath, files[i])), img)
    i += 1

print("Done!")
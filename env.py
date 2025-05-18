import os
import numpy as np
import logging

logging.basicConfig(level=logging.ERROR, format="[%(levelname)-7s] %(message)s")
logger = logging.getLogger(__name__)

# ruler env
task = "segment"
ruler_project_path = os.path.join("ruler")
ruler_dataset_path = os.path.join(ruler_project_path, "dataset")
ruler_data_path = os.path.join(ruler_dataset_path, "data.yaml")

ruler_input_path = os.path.join(ruler_project_path, "input_pictures")
ruler_output_path = os.path.join(ruler_project_path, "output_pictures")

ruler_model_version = "train2"
ruler_task_path = os.path.join(ruler_project_path, "runs", task)
ruler_model_path = os.path.join(
    ruler_task_path, ruler_model_version, "weights", "best.pt"
)

# Roboflow
api_key = "AuWUC9FXJwD3HYlOXO2l"
workspace = "size-estimation"
project_name = "ruler-of-everything"
roboflow_model_version = 2

# Gen6D env
input_path = os.path.join("input")
output_path = os.path.join("output")

ffmpeg_path = os.path.join("C:/", "Program Files", "ffmpeg", "bin", "ffmpeg.exe")


class ObjectArgs:
    def __init__(
        self,
        name: str,
        cfg: str,
        database: str,
        input_file: str,
        output_path: str,
        num: int = 10,
        std: float = 10,
    ):
        self.name = name
        self.cfg = cfg
        self.database = database
        self.input_path = os.path.join(input_path, input_file)
        self.output_path = output_path
        self.num = num
        self.std = std


girl_args = ObjectArgs(
    name="girl",
    cfg="configs/gen6d_pretrain.yaml",
    database="custom/girl",
    input_file="girl_2.jpg",
    output_path=output_path,
)

minion_args = ObjectArgs(
    name="minion",
    cfg="configs/gen6d_pretrain.yaml",
    database="custom/minion",
    input_file="minion_1.jpg",
    output_path=output_path,
)

big_minion_args = ObjectArgs(
    name="big_minion",
    cfg="configs/gen6d_pretrain.yaml",
    database="custom/big_minion",
    input_file="big_minion_1.jpg",
    output_path=output_path,
)


class Model:
    def __init__(
        self,
        name: str,
        range_cm: tuple[float, float],
        rotations: list[tuple[float, list[float]]],
    ):
        self.name = name
        self.input_path = os.path.join(input_path, name)
        self.obj_path = os.path.join(self.input_path, name + ".obj")
        self.mtl_path = os.path.join(self.input_path, name + ".mtl")
        self.range_cm = range_cm
        self.rotations = rotations


soda_can_model = Model("soda_can", (0, 16), [(np.pi / 2, [1, 0, 0])])
soda_bottle_model = Model(
    "bottle", (16, 999), [(np.pi / 2, [1, 0, 0]), (-np.pi / 8, [0, 0, 1])]
)

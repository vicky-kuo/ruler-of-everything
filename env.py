import os

# ruler env
task = "segment"
ruler_project_path = os.path.join("ruler")
ruler_dataset_path = os.path.join(ruler_project_path, "dataset")
ruler_data_path = os.path.join(ruler_dataset_path, "data.yaml")

ruler_input_path = os.path.join(ruler_project_path, "input_pictures")
ruler_output_path = os.path.join(ruler_project_path, "output_pictures")

ruler_model_version = "train"
ruler_task_path = os.path.join(ruler_model_version, "runs", task)
ruler_model_path = os.path.join(
    ruler_task_path, ruler_model_version, "weights", "best.pt"
)

# Roboflow
api_key = "AuWUC9FXJwD3HYlOXO2l"
workspace = "size-estimation"
project_name = "ruler-of-everything"
roboflow_model_version = 2

# Gen6D env
input_path = os.path.join("input_pictures")
output_path = os.path.join("output_pictures")

ffmpeg_path = os.path.join("Program Files", "ffmpeg", "bin", "ffmpeg.exe")


class ObjectArgs:
    def __init__(
        self,
        name: str,
        cfg: str,
        database: str,
        input_file: str,
        output_path: str,
        num: int = 5,
        std: float = 2.5,
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
    input_file="girl_1.jpg",
    output_path=output_path,
)

minion_args = ObjectArgs(
    name="minion",
    cfg="configs/gen6d_pretrain.yaml",
    database="custom/minion",
    input_file="minion_1.jpg",
    output_path=output_path,
)

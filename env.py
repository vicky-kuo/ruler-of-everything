import os

# ruler env
task = "segment"
ruler_project_path = os.path.join("ruler")
ruler_dataset_path = os.path.join(ruler_project_path, "dataset")
ruler_data_path = os.path.join(ruler_dataset_path, "data.yaml")

ruler_input_path = os.path.join(ruler_project_path, "input_pictures")
ruler_output_path = os.path.join(ruler_project_path, "output_pictures")

ruler_model_version = "train"
ruler_model_path = os.path.join(
    ruler_project_path, "runs", task, ruler_model_version, "weights", "best.pt"
)

input_path = os.path.join("input_pictures")
output_path = os.path.join("output_pictures")

ffmpeg_path = os.path.join("Program Files", "ffmpeg", "bin", "ffmpeg.exe")


class ObjectArgs:
    def __init__(
        self,
        name: str,
        cfg: str,
        database: str,
        input_path: str,
        output: str,
        num: int = 5,
        std: float = 2.5,
    ):
        self.name = name
        self.cfg = cfg
        self.database = database
        self.input_path = input_path
        self.input_file1 = os.path.join(input_path, f"{str(name)}_1.jpg")
        self.input_file2 = os.path.join(input_path, f"{str(name)}_2.jpg")
        self.output = output_path
        self.num = num
        self.std = std


girl_args = ObjectArgs(
    name="girl",
    cfg="configs/gen6d_pretrain.yaml",
    database="custom/girl",
    input_path=input_path,
    output=output_path,
)

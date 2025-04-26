import os

input_path = os.path.join("input_pictures")
output_path = os.path.join("output_pictures")

ffmpeg_path = os.path.join("Program Files", "ffmpeg", "bin", "ffmpeg.exe")


class ObjectArgs:
    def __init__(
        self, name: str, cfg: str, database: str, input_path: str, output: str
    ):
        self.name = name
        self.cfg = cfg
        self.database = database
        self.input_path = input_path
        self.input_file1 = os.path.join(input_path, f"{str(name)}_1.jpg")
        self.input_file2 = os.path.join(input_path, f"{str(name)}_2.jpg")
        self.output = output_path


girl_args = ObjectArgs(
    name="girl",
    cfg="configs/gen6d_pretrain.yaml",
    database="custom/girl",
    input_path=input_path,
    output=output_path,
)

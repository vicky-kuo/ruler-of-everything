import os
import sys
import subprocess
import env
from pathlib import Path
from utils import caculate_distance
from ruler.utils import get_absolute_path
from ruler.predict import predict as ruler_predict


def run_gen6d(args, input_file: str):
    gen6d_dir = get_absolute_path("Gen6D")
    python_executable = sys.executable

    # Prepare command with any necessary arguments that Gen6D's predict_frame.py expects
    cmd = [
        python_executable,
        "-m",
        "predict_frame",
        "--name",
        args.name,
        "--input_file",
        get_absolute_path(input_file),
        "--cfg",
        args.cfg,
        "--database",
        args.database,
        "--output",
        get_absolute_path(args.output),
    ]
    print(cmd)
    print(gen6d_dir)
    result = subprocess.run(
        cmd, cwd="gen6d", capture_output=True, text=True, check=True
    )
    return result.stdout


args = env.girl_args

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

ruler_center_1, ruler_width_1 = ruler_predict(args, args.input_file1)
ruler_center_2, ruler_width_2 = ruler_predict(args, args.input_file2)

gen6d_result_1 = run_gen6d(args, args.input_file1)
gen6d_result_2 = run_gen6d(args, args.input_file2)

ruler_move = caculate_distance(ruler_center_1, ruler_center_2)
ruler_width = (ruler_width_1 + ruler_width_2) / 2

print(f"Ruler center 1: {ruler_center_1}, width: {ruler_width_1}")
print(f"Ruler center 2: {ruler_center_2}, width: {ruler_width_2}")
print(f"Ruler movement: {ruler_move}")
print(f"Average ruler width: {ruler_width}")
print("Gen6D results processed successfully")

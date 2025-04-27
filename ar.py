from utils.other_utils import downsample_image
import warnings
import env
from predict_frame import predict as gen6d_predict

warnings.filterwarnings("ignore", category=UserWarning)

args = env.girl_args

downsample_image(args.input_path)

(
    center,
    pose_pr,
    K,
    min_pt,
    max_pt,
) = gen6d_predict(args)

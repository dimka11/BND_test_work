import yaml
from ultralytics import YOLO

from perform_detection import perform_detection


if __name__ == "__main__":
    """read video, perform detection and write video with boxes"""
    with open("config.yaml") as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)

    video_path = config['path']
    out_path = config['out_video']
    model = YOLO(config['weights'])

    perform_detection(video_path, out_path, model)

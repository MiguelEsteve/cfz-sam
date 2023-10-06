import datetime
import math
import os.path
import time

import requests
import torch
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize, CenterCrop, Resize
import cv2
import numpy as np
from PIL import Image

from src.utils.videoutils import DisplayUtils

from configs import log_conf

LOGGER = log_conf.getLogger(__name__)

class ExampleResnet50:

    def __init__(self):
        self.IMAGENET_LABELS_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        self.model = torchvision.models.resnet50(weights=weights)

    @staticmethod
    def decode_predictions(labels: list, top5_indices: torch.Tensor):
        top5_labels = [labels[str(idx)][1] for idx in top5_indices.numpy()]
        return top5_labels

    @staticmethod
    def preprocess(image: np.ndarray):
        transform = Compose([
            ToTensor(),
            Resize(256, antialias=True),
            CenterCrop(size=(224, 224)),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])
        return transform(image)

    def get_imagenet_labels(self):
        response = requests.get(self.IMAGENET_LABELS_URL)
        imagenet_classes_index = response.json()
        return imagenet_classes_index


    def run_on_image(self, image_fn):
        if isinstance(image_fn, str):
            if not os.path.exists(image_fn):
                print(f"Error, {image_fn} not found")
            image = cv2.imread(image_fn)

        processed_image = torch.unsqueeze(self.preprocess(image), dim=0)

        with torch.no_grad():
            self.model.eval()
            y = self.model(processed_image)

        top5_values, top5_indices = torch.topk(y, 5)
        predicted_labels = self.decode_predictions(self.get_imagenet_labels(), top5_indices[0])

        return predicted_labels, top5_values

    def _run_on_frame(self, frame: np.ndarray):
        frame_for_inference = torch.unsqueeze(self.preprocess(frame), dim=0)
        y = self.model(frame_for_inference)
        top5_values, top5_indices = torch.topk(y, 5)
        predicted_labels = self.decode_predictions(self.get_imagenet_labels(), top5_indices[0])

        for i in range(len(top5_values[0])):
            to_draw = f"{predicted_labels[i]}  {top5_values[0][i]:.1f}"
            frame = DisplayUtils.draw_text_on_rect(frame, str(to_draw), position=(2, 50 + (i * 22)), font_scale=0.5,
                                                   padding=4)
            if i == 8:
                break
        return frame

    def run_on_video(self, video_fn: str, add_frame_number=False, read_frames_skipped: int = None, num_frames:int = None,
                     display:bool = False, save_file: str = None):

        t1 = time.time()

        cap = cv2.VideoCapture(video_fn)
        if not cap.isOpened():
            LOGGER.error(f"Error, could not open file {video_fn}")
            exit()
        FR = cap.get(cv2.CAP_PROP_FPS)

        if read_frames_skipped is not None:
            assert 0 < read_frames_skipped < FR, f"frame rate is {FR} and value required {read_frames_skipped}"

        skip_factor = FR - read_frames_skipped + 1 if read_frames_skipped is not None else 1
        LOGGER.debug(
            f"Frame rate: {FR}, read frame rate set to {read_frames_skipped}, calculated skip factor: {skip_factor}")

        skip_count = 0
        frames_count = 0
        count_num_frames = 0

        processed_frames = []

        with torch.no_grad():
            self.model.eval()

            while True:
                ret, frame = cap.read()  # Read a frame from the video
                if not ret:
                    break  # Break the loop fo we've reached the end of the video

                if skip_count % skip_factor == 0:
                    if add_frame_number:
                        frame = DisplayUtils.draw_text_on_rect(frame, str(frames_count), position=(2, 10), font_scale=0.6, padding=4)

                    frame = self._run_on_frame(frame)

                    processed_frames.append(frame)
                    if display:
                        cv2.imshow('Video frame', frame)
                    skip_count = 0
                    count_num_frames += 1
                frames_count += 1
                skip_count += 1
                if cv2.waitKey(int(FR)) & 0xFF == ord('q'):
                    break
                if num_frames is not None and count_num_frames >= num_frames:
                    break

            cap.release()
            cv2.destroyAllWindows()

        all_frames = np.stack(processed_frames)

        elapsed = datetime.timedelta(seconds=math.ceil(time.time() - t1))

        LOGGER.debug(f'Resnet50 inference on "{video_fn}" took {elapsed}')

        if save_file:
            if not os.path.exists(os.path.dirname(save_file)):
                os.makedirs(os.path.dirname(save_file), exist_ok=True)
            DisplayUtils.convert_frames_to_mosaic(frames=all_frames, save_file=save_file)
        return all_frames

class Video:
    def __init__(self):
        pass

    @staticmethod
    def display_video(video_fn: str, add_frame_number=False,
                      read_frames_skipped: int = None):
        DisplayUtils.display_video(video_fn=video_fn, add_frame_number=add_frame_number, read_frames_skipped=read_frames_skipped)


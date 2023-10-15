import math
import os.path
import time
import datetime
import numpy as np
from ultralytics.models.fastsam import FastSAM, FastSAMPrompt
import cv2
import torch
from configs import configs
from configs import log_conf
import matplotlib.pyplot as plt

from src.utils.videoutils import DisplayUtils, VideoUtils
from src.utils.fastSAMutils import FastSAMDisplay

LOGGER = log_conf.getLogger(__name__)


class ExampleFastSAM:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.model = self._get_model()
        self.current_image: np.ndarray = object
        self.prompt_process: FastSAMPrompt = object

        self.display = FastSAMDisplay()

    @property
    def checkpoint(self):
        return os.path.join(configs.PROJECT_PATH, 'FastSAM/weights/FastSAM.pt')

    def _get_model(self, checkpoint: str = None):
        checkpoint = self.checkpoint if checkpoint is None else checkpoint
        model = FastSAM(checkpoint)
        model.to(self.device)
        return model

    def _model_call(self, imgarr:np.ndarray):
        return self.model(imgarr)

    def _set_prompt_process(self, imgarr):
        everything_results = self.model(imgarr, device=self.device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
        self.prompt_process = FastSAMPrompt(imgarr, everything_results, device=self.device)
        return self.prompt_process

    def set_image(self, imgarr):
        self.current_image = imgarr
        self._set_prompt_process(imgarr)

    def get_all_masks(self, imgarr: np.ndarray = None):
        if imgarr is None:
            self.current_image = imgarr
        else:
            self.set_image(imgarr)
        return self.prompt_process.everything_prompt()

    def get_for_box_prompt(self, bbox: list = None):
        if bbox is None:
            bbox = [0, 0, 50, 50]
        return self.prompt_process.box_prompt(bbox=self)

    def get_for_text_prompt(self, text: str):
        return self.prompt_process.text_prompt(text)

    def get_for_point_prompt(self, points=None, point_label=[1]):
        if points is None:
            points = [[620, 360]]
        return self.prompt_process.point_prompt(points=[[620, 360]], point_label=[1])

    def plot_annotations(self, annots, output_path):
        self.prompt_process.plot(annotations=annots, output_path=output_path)

    def plot_to_result(self, annots, withContours=True):
        result = self.prompt_process.plot_to_result(annotations=annots, withContours=withContours)
        return result

    def _run_on_frame(self, frame: np.ndarray):

        t1 = time.time()
        annots = self.get_all_masks(frame)
        self.display.set_img_results(img=frame, results=annots)

        elapsed_inference_ms =  math.ceil((time.time()-t1)*1000)
        elapsed_inference = datetime.timedelta(milliseconds=elapsed_inference_ms)
        LOGGER.debug(f"_run_on_frame - device: {self.model.device},  elapsed inference: {str(elapsed_inference)}")

        t2 = time.time()
        frame = self.display.plot_to_result(annotations=annots) if len(annots) else frame

        elapsed_plot_ms =  math.ceil((time.time()-t2)*1000)
        elapsed_plot = datetime.timedelta(milliseconds=elapsed_plot_ms)
        LOGGER.debug(f"_run_on_frame - elapsed plot: {str(elapsed_plot)}")

        elapsed_all = elapsed_inference + elapsed_plot
        LOGGER.debug(f"_run_on_frame - elapsed all: {str(elapsed_all)}")

        return frame

    def run_on_video(self, video_fn: str, add_frame_number=False, read_frames_skipped: int = None, num_frames:int = None,
                     display:bool = False, save_file: str = None, ask_to_continue:bool = False):

        LOGGER.debug(f'Test video: "{video_fn}"')
        _, data = VideoUtils.get_video_data(video_fn)
        LOGGER.debug(f'video_data: {data}')


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
        if ask_to_continue:
            s = input('\nPress C to cancel (C)\n')
            if s == 'C':
                return


        skip_count = 0
        frames_count = 0
        count_num_frames = 0

        processed_frames = []

        with torch.no_grad():

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



if __name__ == "__main__":
    t = ExampleFastSAM()

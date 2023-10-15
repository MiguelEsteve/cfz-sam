from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import datetime
import math
import torch
import cv2
from configs import log_conf
from mobile_sam import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from src.utils.fastSAMutils import FastSAMDisplay
from src.utils.videoutils import VideoUtils

LOGGER = log_conf.getLogger(__name__)


class ExampleMobileSAM:
    def __init__(self, device:str = None):
        self.model_type = 'vit_t'
        self.sam_checkpoint = 'c:/repos/cfz-sam/MobileSAM/weights/mobile_sam.pt'
        self.device_ = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.device = self.device_ if device is None else torch.device(device)
        self.model = None
        self.imgarr: np.ndarray = None

        self.display = FastSAMDisplay()

    @staticmethod
    def read_image(image_file: str, use: str):
        if not os.path.exists(image_file):
            raise FileNotFoundError(f"{os.path.basename(image_file)} ")
        if use == "PIL":
            img = Image.open(image_file)
            img = np.asarray(img)
            return img
        if use == 'cv2':
            img = cv2.imread(image_file)
            return img
        else:
            raise Exception(f"invalid use: {use}")

    def get_model(self, model_type: str = None, sam_checkpoint: str = None, device: str = None):
        model_type = self.model_type if model_type is None else model_type
        sam_checkpoint = self.sam_checkpoint if sam_checkpoint is None else sam_checkpoint
        device = self.device if device is None else device

        mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        mobile_sam.to(device)
        mobile_sam.eval()
        return mobile_sam

    def get_all_masks(self, imgarr, model=None, with_predictor: bool = False):

        model = self.get_model() if model is None else model
        if with_predictor:
            predictor = SamPredictor(model)
            predictor.set_image(imgarr)
            masks, _, _ = predictor.predict()
        # This code does not work in SURFACE with 'cpu' # TODO check in WS with cpu 
        else:
            mask_generator = SamAutomaticMaskGenerator(model)
            masks = mask_generator.generate(imgarr)

        return masks

    @staticmethod
    def translate_masks(masks):
        masks_ = []
        for mask in masks:
            masks_.append((mask['segmentation'], mask['area']))
        sorted_masks = sorted(masks_, key=lambda x: x[1], reverse=False)
        masks = np.stack([x[0] for x in sorted_masks], axis=0)
        return masks

    @staticmethod
    def get_masks_for_promts(imgarr, prompts, model):
        predictor = SamPredictor(model)
        predictor.set_image(imgarr)
        masks, _, _ = predictor.predict(prompts)
        return masks

    @staticmethod
    def apply_single_mask(imgarr, masks, with_prediction: bool = True):
        imgarr = cv2.cvtColor(imgarr, cv2.COLOR_BGR2RGB)
        redImg = np.zeros(imgarr.shape, imgarr.dtype)
        redImg[:, :, :] = (255, 0, 0)

        image_with_mask = np.copy(imgarr)
        idx = np.random.choice(range(len(masks)), size=1)[0]

        # get one random mask, and convert from bool to unsigned int8
        if not with_prediction:
            mask = masks[idx]["segmentation"].astype(np.uint8)
        else:
            mask = masks[idx].astype(np.uint8)
        image_with_mask[mask == 1] = [255, 0, 0]
        return image_with_mask

    def apply_all_masks(self, imgarr: np.ndarray, masks: np.ndarray):
        results = torch.from_numpy(masks)
        self.display.set_img_results(img=imgarr, results=results)
        frame = self.display.plot_to_result(annotations=results)
        return frame

    @staticmethod
    def apply_single_mask_and_plot(imgarr, masks, with_prediction: bool = True):
        imgarr = cv2.cvtColor(imgarr, cv2.COLOR_BGR2RGB)
        redImg = np.zeros(imgarr.shape, imgarr.dtype)
        redImg[:, :, :] = (255, 0, 0)

        image_with_mask = np.copy(imgarr)
        idx = np.random.choice(range(len(masks)), size=1)[0]

        # get one random mask, and convert from bool to unsigned int8
        if not with_prediction:
            mask = masks[idx]["segmentation"].astype(np.uint8)
        else:
            mask = masks[idx].astype(np.uint8)
        image_with_mask[mask == 1] = [255, 0, 0]

        fig, axes = plt.subplots(1, 3)
        axes[0].imshow(imgarr)
        axes[1].imshow(mask, cmap='gray')
        axes[2].imshow(image_with_mask)
        plt.show()

    def apply_all_masks_and_plot(self, imgarr: np.ndarray, masks: np.ndarray):
        masked_frame = self.apply_all_masks(imgarr, masks)
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(imgarr)
        axes[1].imshow(masked_frame)
        plt.show()

    def run_on_frame(self, frame: np.ndarray, with_predictor=False):

        if self.model is None:
            self.model = self.get_model()

        t1 = time.time()

        masks = self.get_all_masks(frame, with_predictor=with_predictor)

        elapsed_inference_ms = math.ceil((time.time() - t1) * 1000)
        elapsed_inference = datetime.timedelta(milliseconds=elapsed_inference_ms)
        LOGGER.debug(f"_run_on_frame - device: {self.model.device},  elapsed inference: {str(elapsed_inference)}")

        t2 = time.time()

        if not with_predictor:
            masks = self.translate_masks(masks)

        frame = self.apply_all_masks(frame, masks)
        elapsed_plot_ms = math.ceil((time.time() - t2) * 1000)
        elapsed_plot = datetime.timedelta(milliseconds=elapsed_plot_ms)
        LOGGER.debug(f"_run_on_frame - elapsed plot: {str(elapsed_plot)}")

        elapsed_all = elapsed_inference + elapsed_plot
        LOGGER.debug(f"_run_on_frame - elapsed all: {str(elapsed_all)}")

        return frame

    def run_on_video(self, video_fn: str, add_frame_number=False, read_frames_skipped: int = None,
                     num_frames: int = None,
                     display: bool = False, save_file: str = None, ask_to_continue: bool = False, with_predictor=False):

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
                        frame = DisplayUtils.draw_text_on_rect(frame, str(frames_count), position=(2, 10),
                                                               font_scale=0.6, padding=4)

                    frame = self.run_on_frame(frame, with_predictor=with_predictor)

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
    pass

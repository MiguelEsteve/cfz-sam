import datetime
import time
import math
import cv2
import numpy as np
import torch
import os
import sys
sys.path.append('/home/hailo/cfz-sam')

from ultralytics.utils import ops
from ultralytics.engine.results import Results
from ultralytics.models.fastsam.utils import bbox_iou
from ultralytics.models.fastsam import FastSAM, FastSAMPrompt
from src.utils.fastSAMutils import FastSAMDisplay
from src.utils.videoutils import VideoUtils, DisplayUtils
import onnxruntime

from configs.configs import FASTSAM_CHECKPOINTS
from configs import log_conf

LOGGER = log_conf.getLogger(__name__)


class ONNXFastSAM:
    def __init__(self):

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.input_names = None
        self.output_names = None

        self.w_pt = os.path.join(FASTSAM_CHECKPOINTS, 'FastSAM.pt')
        self.w_onnx = os.path.join(FASTSAM_CHECKPOINTS, 'FastSAM.onnx')

        self.onnx_session: onnxruntime.InferenceSession = None
        self.display = FastSAMDisplay()

    def run_pt(self, img_fn):
        model = FastSAM(self.w_pt)
        model.to(self.device)
        y = model(img_fn)
        return y

    def prepare_input(self, imgarr):
        input_img = cv2.cvtColor(imgarr, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, self.get_input_w_h())
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        original_img = torch.from_numpy(imgarr)
        original_img = torch.unsqueeze(original_img, 0)
        original_img = torch.permute(original_img, dims=[0, 3, 1, 2]) / 255.0

        return input_tensor, original_img

    def initialize_model_onnx(self):
        self.onnx_session = onnxruntime.InferenceSession(self.w_onnx, providers=['CPUExecutionProvider'])
        self.get_input_details()
        self.get_output_details()

    def get_input_details(self):
        model_inputs = self.onnx_session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

    def get_output_details(self):
        model_outputs = self.onnx_session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def get_input_w_h(self):
        model_inputs = self.onnx_session.get_inputs()
        return model_inputs[0].shape[3], model_inputs[0].shape[2]

    def inference(self, input_tensor):
        outputs = self.onnx_session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs

    @staticmethod
    def postprocess(preds, img, orig_imgs, conf: float = 0.4, iou: float = 0.9, agnostic_nms: bool = False,
                    max_det: int = 300, img_path: str = None, retina_masks: bool = False):
        # sometimes conf = 0.4, iou = 0.9
        """
        Perform post-processing steps on predictions, including non-max suppression and scaling boxes to original image
        size, and returns the final results.

        Args:
            preds (list): The raw output predictions from the model.
            img (torch.Tensor): The processed image tensor.
            orig_imgs (list | torch.Tensor): The original image or list of images.

        Returns:
            (list): A list of Results objects, each containing processed boxes, masks, and other metadata.
        """
        if isinstance(preds[0], np.ndarray):
            preds[0] = torch.from_numpy(preds[0])

        p = ops.non_max_suppression(
            preds[0],
            conf,
            iou,
            agnostic_nms,
            max_det,
            nc=1)  # set to 1 class since SAM has no class predictions

        full_box = torch.zeros(p[0].shape[1], device=p[0].device)
        full_box[2], full_box[3], full_box[4], full_box[6:] = img.shape[3], img.shape[2], 1.0, 1.0
        full_box = full_box.view(1, -1)
        critical_iou_index = bbox_iou(full_box[0][:4], p[0][:, :4], iou_thres=0.9, image_shape=img.shape[2:])
        if critical_iou_index.numel() != 0:
            full_box[0][4] = p[0][critical_iou_index][:, 4]
            full_box[0][6:] = p[0][critical_iou_index][:, 6:]
            p[0][critical_iou_index] = full_box

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        results = []

        if isinstance(preds[1], np.ndarray):
            preds[1] = torch.from_numpy(preds[1])

        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
        for i, pred in enumerate(p):
            orig_img = orig_imgs[i]
            img_path = img_path
            if not len(pred):  # save empty boxes
                masks = None
            elif retina_masks:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            model_names = {0: 'object'}
            results.append(Results(orig_img, path=img_path, names=model_names, boxes=pred[:, :6], masks=masks))
        return results

    @staticmethod
    def results_to_maks(results: Results) -> np.ndarray:
        return results[0].masks.cpu().numpy()

    def get_all_masks(self, imgarr: np.ndarray, everything_results: Results):
        prompt_process = FastSAMPrompt(imgarr, everything_results, device=self.device)
        return prompt_process.everything_prompt()

    def plot(self, imgarr: np.ndarray, results: Results, output: str = './images', with_contours: bool = True):
        prompt_process = FastSAMPrompt(imgarr, results, device=self.device)
        annotations = prompt_process.everything_prompt()[0]
        annotations.path = 'fastSAM.jpg'

        prompt_process.plot(annotations=prompt_process.everything_prompt(), output=output, with_contours=with_contours)

    def plot_to_results(self, imgarr: np.ndarray, results: Results):

        self.display.set_img_results(img=imgarr, results=self.results_to_maks(results))
        return self.display.plot_to_result(annotations=self.results_to_maks(results))

    def run_on_frame(self, frame: np.ndarray):

        t1 = time.time()
        preprocessed, original = self.prepare_input(frame)
        elapsed_preprocessed_ms = math.ceil((time.time() - t1) * 1000)
        elapsed_preprocessed = datetime.timedelta(microseconds=elapsed_preprocessed_ms)
        LOGGER.debug(f"_run_on_frame - elapsed preprocessed: {str(elapsed_preprocessed)}")

        t2 = time.time()
        predictions = self.inference(preprocessed)
        elapsed_inference_ms = math.ceil((time.time() - t2) * 1000)
        elapsed_inference = datetime.timedelta(milliseconds=elapsed_inference_ms)
        LOGGER.debug(f"_run_on_frame - elapsed inference: {str(elapsed_inference)}")

        t3 = time.time()
        results = self.postprocess(predictions, preprocessed, original)
        masks = self.results_to_maks(results)
        elapsed_postprocess_ms = math.ceil((time.time() - t3) * 1000)
        elapsed_postproces = datetime.timedelta(microseconds=elapsed_postprocess_ms)
        LOGGER.debug(f"_run_on_frame - elapsed postprocessed: {str(elapsed_postproces)}")

        self.display.set_img_results(img=frame, results=masks)
        frame = self.display.plot_to_result(annotations=masks) if len(masks) else frame

        elapsed_all_ms = math.ceil((time.time() - t1) * 1000)
        elapsed_all = datetime.timedelta(milliseconds=elapsed_all_ms)
        LOGGER.debug(f"_run_on_frame - elapsed all: {str(elapsed_all)}")

        return frame

    def run_on_video(self, video_fn: str, add_frame_number=False, read_frames_skipped: int = None,
                     num_frames: int = None,
                     display: bool = False, save_file: str = None, ask_to_continue: bool = False):

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

                    frame = self.run_on_frame(frame)

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

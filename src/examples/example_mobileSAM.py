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

LOGGER = log_conf.getLogger(__name__)


class ExampleMobileSAM:
    def __init__(self):
        self.model_type = 'vit_t'
        self.sam_checkpoint = 'c:/repos/cfz-sam/MobileSAM/weights/mobile_sam.pt'
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model = None
        self.imgarr: np.ndarray = None

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

    def get_all_masks(self, imgarr, model=None, with_predictor: bool = True):

        model = self.get_model() if model is None else model
        if with_predictor:
            predictor = SamPredictor(model)
            predictor.set_image(imgarr)
            masks, _, _ = predictor.predict()
        """
        # This code does not work in SURFACE with 'cpu' # TODO check in WS with cpu 
        else:
            mask_generator = SamAutomaticMaskGenerator(model)
            masks = mask_generator.generate(imgarr)
        """
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

    @staticmethod
    def apply_all_masks(imgarr:np.ndarray, masks:np.ndarray):
        f = FastSAMDisplay()
        results = torch.from_numpy(masks)
        f.set_img_results(img=imgarr, results=results)
        frame = f.plot_to_result(annotations=results)


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

    def run_on_frame(self, frame: np.ndarray):

        if self.model is None:
            self.model = self.get_model()

        t1 = time.time()
        masks = self.get_all_masks(frame)

        elapsed_inference_ms = math.ceil((time.time() - t1) * 1000)
        elapsed_inference = datetime.timedelta(milliseconds=elapsed_inference_ms)
        LOGGER.debug(f"_run_on_frame - elapsed inference: {str(elapsed_inference)}")

        frame = self.apply_single_mask(frame, masks)

        t2 = time.time()
        elapsed_plot_ms = math.ceil((time.time() - t2) * 1000)
        elapsed_plot = datetime.timedelta(milliseconds=elapsed_plot_ms)
        LOGGER.debug(f"_run_on_frame - elapsed plot: {str(elapsed_plot)}")

        elapsed_all = elapsed_inference + elapsed_plot
        LOGGER.debug(f"_run_on_frame - elapsed all: {str(elapsed_all)}")

        return frame


if __name__ == "__main__":
    pass

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import cv2

from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from mobile_sam import SamAutomaticMaskGenerator

class ExampleMobileSAM:

    @staticmethod
    def read_image(image_file: str, use: str):
        if not os.path.exists(image_file):
            raise FileNotFoundError(f"{os.path.basename(image_file)} ")
        if use == "PIL":
            img = Image.open(image_file)
            return np.asarray(img)
        if use == 'cv2':
            return cv2.imread(image_file)
        else:
            raise Exception(f"invalid use: {use}")

    @staticmethod
    def get_model(model_type, sam_checkpoint, device):
        mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        mobile_sam.to(device)
        mobile_sam.eval()
        return mobile_sam

    @staticmethod
    def get_all_masks(imgarr, model):
        mask_generator = SamAutomaticMaskGenerator(model)
        masks = mask_generator.generate(imgarr)
        return masks

    @staticmethod
    def get_masks_for_promts(imgarr, prompts, model):
        predictor = SamPredictor(model)
        predictor.set_image(imgarr)
        masks, _, _ = predictor.predict(prompts)
        return masks

    @staticmethod
    def apply_mask_and_plot(imgarr, masks):
        imgarr = cv2.cvtColor(imgarr, cv2.COLOR_BGR2RGB)
        redImg = np.zeros(imgarr.shape, imgarr.dtype)
        redImg[:, :, :] = (255, 0, 0)

        image_with_mask = np.copy(imgarr)
        idx = np.random.choice(range(len(masks)), size=1)[0]

        # get one random mask, and convert from bool to unsigned int8
        mask = masks[idx]["segmentation"].astype(np.uint8)
        image_with_mask[mask == 1] = [255,0,0]

        fig, axes = plt.subplots(1, 3)
        axes[0].imshow(imgarr)
        axes[1].imshow(mask, cmap='gray')
        axes[2].imshow(image_with_mask)
        plt.show()


if __name__ == "__main__":
    pass

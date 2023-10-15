import os
import sys
import cv2
import numpy as np
import torch


class FastSAMDisplay:

    def __init__(self, image: np.ndarray = None, results: torch.Tensor = None):

        self.img = image
        self.results = results

    def set_img_results(self, img: np.ndarray, results: torch.Tensor):
        self.img = img
        self.results = results

    def plot_to_result(self,
                       annotations,
                       bboxes=None,
                       points=None,
                       point_label=None,
                       mask_random_color=True,
                       better_quality=True,
                       retina=False,
                       withContours=True) -> np.ndarray:

        image = self.img.copy()
        original_h, original_w, _ = image.shape

        morphed_annotations = []
        if better_quality:
            for i, mask in enumerate(annotations):
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                morphed_annotations.append(
                    cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8)))
            morphed_annotations = np.array(morphed_annotations)

            # Convert the list back to a tensor if you need to
            if isinstance(annotations, np.ndarray):
                annotations = torch.tensor(morphed_annotations)
            else:
                annotations = morphed_annotations

        mask_image = self.fast_show_mask(
            annotation=annotations,
            random_color=mask_random_color,
            bboxes=bboxes,
            points=points,
            pointlabel=point_label,
            retinamask=retina,
            target_height=original_h,
            target_width=original_w,
        )
        alpha = 0.6

        mask_rgb = (mask_image[:, :, :3] * 255).astype(np.uint8)
        image = cv2.addWeighted(image, 1 - alpha, mask_rgb, alpha, 0.5)

        if withContours:

            for i, mask in enumerate(annotations):
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                annotation = mask.astype(np.uint8)
                if not retina:
                    annotation = cv2.resize(annotation, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
                contours, _ = cv2.findContours(annotation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, (255, 0, 0), 2)

        if bboxes is not None:
            for bbox in bboxes:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        if points is not None:
            for i, point in enumerate(points):
                if point_label[i] == 1:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (255, 0, 255)  # Magenta
                cv2.circle(image, (int(point[0]), int(point[1])), 10, color, -1)

        return image

    @staticmethod
    def fast_show_mask(
            annotation,
            random_color=True,
            bboxes=None,
            points=None,
            pointlabel=None,
            retinamask=True,
            target_height=960,
            target_width=960,
    ):
        if isinstance(annotation, torch.Tensor):
            annotation = annotation.cpu().numpy()

        height = annotation.shape[1]
        weight = annotation.shape[2]

        # Sort annotations based on area.
        areas = np.sum(annotation, axis=(1, 2))
        sorted_indices = np.argsort(areas)
        sorted_indices_ = sorted_indices[::-1]

        annotation = annotation[sorted_indices_]

        # Prepare a blank RGBA image to overlay masks
        show = np.zeros((height, weight, 4), dtype=np.uint8)

        for i, mask in enumerate(annotation):
            if random_color:
                # Generate a new random color for each mask
                color = np.random.randint(0, 256, (1, 1, 3), dtype=np.uint8)

            else:
                color = np.array([30, 144, 255], dtype=np.uint8).reshape(1, 1, 3)  # Default blue color

            transparency = np.array([int(0.2 * 255)], dtype=np.uint8).reshape(1, 1, 1)  # 20% transparency
            rgba_color = np.concatenate([color, transparency], axis=-1)

            mask_indices = mask.astype(bool)
            show[mask_indices] = rgba_color  # Set the RGBA values to the color where the mask is True

        if bboxes is not None:
            for bbox in bboxes:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(show, (x1, y1), (x2, y2), (255, 0, 0), 1)  # blue color for bboxes

        # draw points
        if points is not None:
            for i, point in enumerate(points):
                if pointlabel[i] == 1:
                    cv2.circle(show, (int(point[0]), int(point[1])), 10, (0, 255, 255), -1)  # yellow for label 1
                elif pointlabel[i] == 0:
                    cv2.circle(show, (int(point[0]), int(point[1])), 10, (255, 0, 255), -1)  # magenta for label 0

        if not retinamask:
            show = cv2.resize(show, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

        return show

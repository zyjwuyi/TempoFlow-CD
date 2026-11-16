import os
import warnings
from typing import List, Tuple
import numpy as np
from collections.abc import Sized
import cv2

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import albumentations as alb


class ChangeDetectionDataset(Dataset, Sized):
    def __init__(self, data_path: str, mode: str, return_geometry: bool = False,
                 geometry_target: str = "mask_distance") -> None:


        self._mode = mode
        self.return_geometry = return_geometry
        self.geometry_target = geometry_target


        split_path = os.path.join(data_path, mode)
        split_list = os.path.join(split_path, "list", mode + ".txt")
        split_directories = all(
            os.path.isdir(os.path.join(split_path, directory))
            for directory in ("A", "B", "label")
        )
        self._data_path = (
            split_path if os.path.isfile(split_list) or split_directories else data_path
        )
        self._A = os.path.join(self._data_path, "A")
        self._B = os.path.join(self._data_path, "B")
        self._label = os.path.join(self._data_path, "label")


        self._list_images = self._read_images_list(self._data_path)


        if mode == 'train' or mode == 'trainval':
            self._augmentation = _create_shared_augmentation()
            self._aberration = _create_aberration_augmentation()


        self._normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, idx):

        img_name = self._list_images[idx].strip('\n')


        x_ref = cv2.cvtColor(cv2.imread(os.path.join(self._A, img_name)), cv2.COLOR_BGR2RGB) / 255
        x_test = cv2.cvtColor(cv2.imread(os.path.join(self._B, img_name)), cv2.COLOR_BGR2RGB) / 255
        x_mask = _binarize(cv2.imread(os.path.join(self._label, img_name))[:, :, 0] / 255)


        if self._mode == 'train' or self._mode == 'trainval':
            x_ref, x_test, x_mask = self._augment(x_ref, x_test, x_mask)

        x_ref, x_test, x_mask = self._to_tensors(x_ref, x_test, x_mask)

        if self.return_geometry:
            from utils.geometry_target import build_geometry_target
            geometry = build_geometry_target(x_mask.unsqueeze(0), self.geometry_target).squeeze(0)
            return x_ref, x_test, x_mask, geometry
        return x_ref, x_test, x_mask

    def __len__(self):
        return len(self._list_images)

    def get_name(self, idx: int) -> str:
        return self._list_images[idx].strip().split()[0]

    def _read_images_list(self, data_path: str) -> List[str]:
        images_list_file = os.path.join(data_path, 'list', self._mode + ".txt")
        if os.path.isfile(images_list_file):
            with open(images_list_file, "r") as f:
                listed_names = [line.strip().split()[0] for line in f if line.strip()]
            resolved_names = []
            incomplete = []
            for listed_name in listed_names:
                candidates = (listed_name, self._mode + listed_name)
                resolved = None
                for candidate in candidates:
                    paths = [
                        os.path.join(directory, candidate)
                        for directory in (self._A, self._B, self._label)
                    ]
                    if all(
                        os.path.isfile(os.path.join(directory, candidate))
                        for directory in (self._A, self._B, self._label)
                    ):


                        if candidate != listed_name and any(
                            cv2.imread(path, cv2.IMREAD_UNCHANGED) is None
                            for path in paths
                        ):
                            continue
                        resolved = candidate
                        break
                if resolved is None:
                    incomplete.append(listed_name)
                else:
                    resolved_names.append(resolved)
            if incomplete:
                warnings.warn(
                    f"skipping {len(incomplete)} incomplete manifest entries in "
                    f"{images_list_file}: {', '.join(incomplete[:8])}",
                    RuntimeWarning,
                )
            if not resolved_names:
                raise ValueError(
                    f"manifest contains no complete A/B/label samples: {images_list_file}"
                )
            return [name + "\n" for name in resolved_names]


        filename_sets = {}
        for key, directory in (("A", self._A), ("B", self._B), ("label", self._label)):
            if not os.path.isdir(directory):
                raise FileNotFoundError(
                    f"missing dataset manifest {images_list_file!r} and directory {directory!r}"
                )
            filename_sets[key] = {
                name for name in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, name))
            }
        if not filename_sets["A"]:
            raise ValueError(f"dataset split contains no images: {data_path}")
        if not (filename_sets["A"] == filename_sets["B"] == filename_sets["label"]):
            counts = ", ".join(
                f"{key}={len(names)}" for key, names in filename_sets.items()
            )
            raise ValueError(f"A/B/label filename mismatch in {data_path}: {counts}")
        return [name + "\n" for name in sorted(filename_sets["A"])]

    def _augment(self, x_ref: np.ndarray, x_test: np.ndarray,
                 x_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        transformed = self._augmentation(image=x_ref, image0=x_test, x_mask0=x_mask)
        x_ref = transformed["image"]
        x_test = transformed["image0"]
        x_mask = transformed["x_mask0"]


        x_ref = self._aberration(image=x_ref)["image"]
        x_test = self._aberration(image=x_test)["image"]

        return x_ref, x_test, x_mask

    def _to_tensors(self, x_ref: np.ndarray, x_test: np.ndarray, x_mask: np.ndarray) -> Tuple[Tensor, Tensor, Tensor]:
        return (
            self._normalize(torch.tensor(x_ref).permute(2, 0, 1)),
            self._normalize(torch.tensor(x_test).permute(2, 0, 1)),
            torch.tensor(x_mask),
        )


def _create_shared_augmentation():
    return alb.Compose([
        alb.HorizontalFlip(p=0.5),
        alb.VerticalFlip(p=0.5),
        alb.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5)
    ], additional_targets={"image0": "image", "x_mask0": "mask"})


def _create_aberration_augmentation():
    return alb.Compose([
        alb.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        alb.GaussianBlur(blur_limit=(3, 5), p=0.5)
    ])


def _binarize(mask: np.ndarray) -> np.ndarray:
    return np.clip(mask * 255, 0, 1).astype(int)

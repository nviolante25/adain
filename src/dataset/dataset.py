import os
import numpy as np
from PIL import Image
from pathlib import Path
from collections import OrderedDict
from torch.utils.data import Dataset, Sampler


class ConfigDict(OrderedDict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]


def is_image_ext(filename: str):
    Image.init()
    ext = str(filename).split(".")[-1].lower()
    return f".{ext}" in Image.EXTENSION


class Transform:
    """Converts image as np.array into torch.tensor and optionally applies augmentation"""

    def __init__(self, transform, augmentation=None):
        self._transform = transform
        self._augmentation = augmentation

    def __call__(self, pil_image):
        tensor_image = self._transform(pil_image)
        if self._augmentation is not None:
            tensor_image = self._augmentation(tensor_image)
        return tensor_image


class ImageDataset(Dataset):
    def __init__(self, source_dir, transform):
        super().__init__()
        self._transform = transform
        self._image_paths = self._get_image_paths(source_dir)
        self._image_shape = list(self[0].shape)
        self._info = ConfigDict(
            dir=source_dir,
            total_images=len(self),
            image_shape=self._image_shape,
        )

    def _get_image_paths(self, source_dir):
        paths = [
            str(f)
            for f in Path(source_dir).rglob("*")
            if is_image_ext(f) and os.path.isfile(f)
        ]
        if not len(paths) > 0:
            raise ValueError(f"No images found in {source_dir}")
        return paths

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        image_array = Image.open(self._image_paths[idx])
        image_tensor = self._transform(image_array)
        return image_tensor

    @property
    def info(self):
        return self._info

    def imshow(self, idx):
        Image.open(self._image_paths[idx]).show()

    def __repr__(self):
        s = f"Directory   : {self._info.dir}\n"
        s += f"Total images: {self._info.total_images}\n"
        s += f"Image shape : {self._info.image_shape}\n"
        s += f"Label shape :"
        return s

    def __str__(self):
        return self.__repr__()

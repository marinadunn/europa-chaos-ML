import random
import torch

from torchvision.transforms import functional as F
from typing import List, Optional, Tuple, Union



def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target
    
# class ColorJitter(torch.nn.Module):
#     def __init__(
#         self,
#         brightness: Union[float, Tuple[float, float]] = 0,
#         contrast: Union[float, Tuple[float, float]] = 0,
#         saturation: Union[float, Tuple[float, float]] = 0,
#         hue: Union[float, Tuple[float, float]] = 0,
#     ) -> None:
#         super().__init__()
#         _log_api_usage_once(self)
#         self.brightness = self._check_input(brightness, "brightness")
#         self.contrast = self._check_input(contrast, "contrast")
#         self.saturation = self._check_input(saturation, "saturation")
#         self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

#     @torch.jit.unused
#     def _check_input(self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
#         if isinstance(value, numbers.Number):
#             if value < 0:
#                 raise ValueError(f"If {name} is a single number, it must be non negative.")
#             value = [center - float(value), center + float(value)]
#             if clip_first_on_zero:
#                 value[0] = max(value[0], 0.0)
#         elif isinstance(value, (tuple, list)) and len(value) == 2:
#             value = [float(value[0]), float(value[1])]
#         else:
#             raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

#         if not bound[0] <= value[0] <= value[1] <= bound[1]:
#             raise ValueError(f"{name} values should be between {bound}, but got {value}.")

#         # if value is 0 or (1., 1.) for brightness/contrast/saturation
#         # or (0., 0.) for hue, do nothing
#         if value[0] == value[1] == center:
#             return None
#         else:
#             return tuple(value)


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

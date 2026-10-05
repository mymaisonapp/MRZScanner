from typing import List

import cv2
import numpy as np

from .. import runtime as rt
from ..utils import DecodeMode, TextDecoder

__all__ = ['Inference']


class Inference:

    configs = {
        '20240919': {
            'model_path': 'mobilenetv4_conv_small_bifpn1_l6_d256_p12345_finetune_20240919_fp32.onnx',
            'img_size_infer': (512, 512),
        },
    }

    def __init__(
        self,
        gpu_id: int = 0,
        backend: rt.Backend = rt.Backend.cpu,
        model_cfg: str = '20240919',
        **kwargs
    ) -> None:
        self.model_cfg = model_cfg
        self.cfg = cfg = self.configs[model_cfg]
        self.image_size = cfg['img_size_infer']
        directory = rt.package_directory(__file__)
        self.model_path = rt.bundled_model(directory, cfg['model_path'])
        self.model = rt.ONNXEngine(self.model_path, gpu_id, backend, **kwargs)
        self.input_key = list(self.model.input_infos.keys())[0]
        self.output_key = list(self.model.output_infos.keys())[0]

        # Text en/de-coding
        keys = ["<PAD>", "<EOS>"] + \
            list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<&")
        chars_dict = {
            k: i
            for i, k in enumerate(keys)
        }

        self.text_dec = TextDecoder(
            chars_dict=chars_dict,
            decode_mode=DecodeMode.Normal
        )

    def preprocess(self, img: np.ndarray, normalize: bool) -> np.ndarray:

        # Padding
        if img.shape[0] < img.shape[1]:  # H < W
            pad = (img.shape[1] - img.shape[0]) // 2
            padding = (pad, pad, 0, 0)
            img = cv2.copyMakeBorder(
                img, *padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            pad = (img.shape[0] - img.shape[1]) // 2
            padding = (0, 0, pad, pad)
            img = cv2.copyMakeBorder(
                img, *padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        tensor = rt.imresize(img, size=tuple(self.image_size))
        tensor = np.transpose(tensor, axes=(2, 0, 1)).astype('float32')
        tensor = tensor[None] / 255.0 if normalize else tensor[None]

        return {self.input_key: tensor}

    def __call__(self, img: np.ndarray, normalize: bool = True) -> List[str]:
        tensor = self.preprocess(img, normalize=normalize)
        x = self.model(**tensor)
        x = x[self.output_key].argmax(-1)
        result = self.text_dec(x)[0]
        result = result.split('&')
        return result

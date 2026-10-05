from pathlib import Path

import cv2
import numpy as np
import pytest

from mrzscanner import runtime as rt
from mrzscanner.spotting.infer import Inference


class DummyONNXEngine:
    def __init__(self, model_path, gpu_id, backend, **kwargs):
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.backend = backend
        self.input_infos = {"input": "dummy"}
        self.output_infos = {"output": "dummy"}

    def __call__(self, **kwargs):
        # Return a dummy output array with shape (1, 10, 20)
        return {"output": np.zeros((1, 10, 20), dtype=np.float32)}


def dummy_imresize(img, size):
    # Return an array of the given size with the same number of channels as the input.
    channels = img.shape[2] if img.ndim == 3 else 1
    return np.zeros((size[0], size[1], channels), dtype=img.dtype)


def _use_fake_runtime(monkeypatch, engine=DummyONNXEngine):
    monkeypatch.setattr(rt, 'bundled_model', lambda directory, filename: Path(filename))
    monkeypatch.setattr(rt, 'ONNXEngine', engine)
    monkeypatch.setattr(rt, 'imresize', dummy_imresize)


def test_init_uses_packaged_weights(monkeypatch):
    """A present weight file is opened locally and nothing is downloaded."""
    _use_fake_runtime(monkeypatch)
    inf = Inference()
    assert inf.image_size == (512, 512)
    assert inf.input_key == "input"
    assert inf.output_key == "output"
    assert hasattr(inf, "text_dec")


def test_missing_model_is_not_downloaded(tmp_path, monkeypatch):
    """A missing weight file fails without contacting Google Drive."""
    monkeypatch.setattr(rt, 'package_directory', lambda _file: tmp_path)
    with pytest.raises(FileNotFoundError, match='Google Drive'):
        Inference()


def test_preprocess_padding_horizontal(monkeypatch):
    # Test when image height < width (H < W)
    _use_fake_runtime(monkeypatch)

    record = {}

    def dummy_copyMakeBorder(img, top, bottom, left, right, borderType, value):
        record["top"] = top
        record["bottom"] = bottom
        record["left"] = left
        record["right"] = right
        return img
    monkeypatch.setattr(cv2, "copyMakeBorder", dummy_copyMakeBorder)

    inf = Inference()
    # Create an image with shape (200, 300, 3) where H < W.
    img = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
    tensor_dict = inf.preprocess(img, normalize=True)
    # Expected padding: pad = (300 - 200) // 2 = 50 -> top=50, bottom=50, left=0, right=0.
    assert record["top"] == 50
    assert record["bottom"] == 50
    assert record["left"] == 0
    assert record["right"] == 0
    tensor = tensor_dict[inf.input_key]
    assert tensor.shape == (1, 3, 512, 512)
    # Since dummy_imresize returns zeros, after normalization the tensor remains zeros.
    assert np.all(tensor == 0)


def test_preprocess_padding_vertical(monkeypatch):
    # Test when image height >= width (H >= W)
    _use_fake_runtime(monkeypatch)

    record = {}

    def dummy_copyMakeBorder(img, top, bottom, left, right, borderType, value):
        record["top"] = top
        record["bottom"] = bottom
        record["left"] = left
        record["right"] = right
        return img
    monkeypatch.setattr(cv2, "copyMakeBorder", dummy_copyMakeBorder)

    inf = Inference()
    # Create an image with shape (300, 200, 3) where H >= W.
    img = np.random.randint(0, 256, (300, 200, 3), dtype=np.uint8)
    tensor_dict = inf.preprocess(img, normalize=False)
    # Expected padding: pad = (300 - 200) // 2 = 50 -> left=50, right=50, top=0, bottom=0.
    assert record["top"] == 0
    assert record["bottom"] == 0
    assert record["left"] == 50
    assert record["right"] == 50
    tensor = tensor_dict[inf.input_key]
    assert tensor.shape == (1, 3, 512, 512)
    # Without normalization, dummy_imresize still returns zeros.
    assert np.all(tensor == 0)


def test_call(monkeypatch):

    class DummyONNXEngineCall:
        def __init__(self, model_path, gpu_id, backend, **kwargs):
            self.input_infos = {"input": "dummy"}
            self.output_infos = {"output": "dummy"}

        def __call__(self, **kwargs):
            # Return a dummy output array.
            return {"output": np.zeros((1, 10, 20), dtype=np.float32)}

    _use_fake_runtime(monkeypatch, DummyONNXEngineCall)

    inf = Inference()
    # Override text_dec to return a fixed string.
    inf.text_dec = lambda x: ("HELLO&WORLD",)
    img = np.random.randint(0, 256, (300, 200, 3), dtype=np.uint8)
    result = inf(img, normalize=True)
    assert isinstance(result, list)
    assert result == ["HELLO", "WORLD"]

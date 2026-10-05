"""ONNX runtime and image helpers for the packaged MRZ models.

The weight files live in each model's ``ckpt`` directory and are installed
with the package. Startup never downloads weights, including from Google Drive.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Sequence, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort

_PolygonLike = Union[np.ndarray, Sequence, 'Polygon']


class EnumCheckMixin:
    """Accept a member, its name, or its value."""

    @classmethod
    def obj_to_enum(cls, obj: Any):
        if isinstance(obj, str):
            try:
                return getattr(cls, obj)
            except AttributeError:
                pass
        elif isinstance(obj, cls):
            return obj
        elif isinstance(obj, int):
            try:
                return cls(obj)
            except ValueError:
                pass
        raise ValueError(f'{obj} is not correct for {cls.__name__}')


class Backend(EnumCheckMixin, Enum):
    cpu = 0
    cuda = 1


class ONNXEngine:
    """Run a local ONNX file on CPU, or on CUDA when that provider exists."""

    def __init__(
        self,
        model_path: Union[str, Path],
        gpu_id: int = 0,
        backend: Union[str, int, Backend] = Backend.cpu,
        **_kwargs,
    ):
        backend = Backend.obj_to_enum(backend)
        if backend == Backend.cuda:
            providers = [
                ('CUDAExecutionProvider', {'device_id': gpu_id}),
                'CPUExecutionProvider',
            ]
        else:
            providers = ['CPUExecutionProvider']

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 2
        self.model_path = str(model_path)
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=options,
            providers=providers,
        )
        self.input_infos = {
            item.name: {'shape': item.shape, 'dtype': item.type}
            for item in self.session.get_inputs()
        }
        self.output_infos = {
            item.name: {'shape': item.shape, 'dtype': item.type}
            for item in self.session.get_outputs()
        }

    def __call__(self, **arrays) -> dict:
        names = list(self.output_infos)
        values = self.session.run(names, arrays)
        return dict(zip(names, values))

    def __repr__(self) -> str:
        provider = ', '.join(self.session.get_providers())
        return f'ONNXEngine(path={self.model_path}, provider={provider})'


def package_directory(file: str) -> Path:
    """Directory that contains the calling module."""
    return Path(file).resolve().parent


def bundled_model(directory: Union[str, Path], filename: str) -> Path:
    """Return a weight file shipped in ``directory/ckpt``.

    Raises FileNotFoundError instead of fetching the file from Google Drive.
    """
    path = Path(directory) / 'ckpt' / filename
    if not path.is_file():
        raise FileNotFoundError(
            f'MRZ model {filename} is missing at {path}. '
            'The weights ship inside the mrzscanner package and are not '
            'downloaded from Google Drive.'
        )
    return path


def is_numpy_img(image: Any) -> bool:
    """True for a grayscale or 1/3-channel numpy image."""
    return isinstance(image, np.ndarray) and (
        image.ndim == 2 or (image.ndim == 3 and image.shape[-1] in (1, 3))
    )


def imresize(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize to ``size`` as (height, width), bilinearly."""
    height, width = size
    return cv2.resize(img, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)


def imbinarize(img: np.ndarray) -> np.ndarray:
    """Otsu-binarize a gray or BGR image."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def centercrop(img: np.ndarray) -> np.ndarray:
    """Crop the largest square centered in the image."""
    height, width = img.shape[:2]
    side = min(height, width)
    y_start = (height - side) // 2
    x_start = (width - side) // 2
    return img[y_start:y_start + side, x_start:x_start + side].copy()


def order_points_clockwise(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    if points.shape != (4, 2):
        raise ValueError('Input array `pts` must be of shape (4, 2).')
    by_x = points[np.argsort(points[:, 0]), :]
    left = by_x[:2, :]
    right = by_x[2:, :]
    top_left, bottom_left = left[np.argsort(left[:, 1]), :]
    top_right, bottom_right = right[np.argsort(right[:, 1]), :]
    return np.stack([top_left, top_right, bottom_right, bottom_left])


class Polygon:
    """A contour stored as an ``(N, 2)`` float32 array."""

    def __init__(self, array: _PolygonLike):
        self._array = self._as_points(array)

    def __len__(self) -> int:
        return int(self._array.shape[0])

    def __getitem__(self, item):
        return self._array[item]

    def __iter__(self):
        return iter(self._array)

    def numpy(self) -> np.ndarray:
        return self._array.copy()

    @property
    def area(self) -> float:
        return float(cv2.moments(self._array)['m00'])

    @property
    def min_box_wh(self) -> Tuple[float, float]:
        _center, (width, height), _angle = cv2.minAreaRect(self._array)
        return float(width), float(height)

    def to_min_boxpoints(self) -> 'Polygon':
        boxed = cv2.boxPoints(cv2.minAreaRect(self._array)).round(4)
        return Polygon(order_points_clockwise(np.asarray(boxed, dtype=np.float32)))

    @staticmethod
    def _as_points(array: _PolygonLike) -> np.ndarray:
        if isinstance(array, Polygon):
            array = array.numpy()
        points = np.asarray(array, dtype=np.float32)
        if points.ndim == 3 and points.shape[1] == 1:
            points = np.squeeze(points, axis=1)
        if points.ndim == 1 and points.size == 0:
            points = points.reshape(0, 2)
        if points.ndim != 2 or points.shape[1] != 2:
            raise TypeError('A polygon must be an (N, 2) array of points.')
        return points


class Polygons:
    """A list of contours, with the indexing used by MRZ detection."""

    def __init__(self, polygons: Sequence):
        self._polygons = [Polygon(polygon) for polygon in polygons]

    def __len__(self) -> int:
        return len(self._polygons)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._polygons[item]
        if isinstance(item, np.ndarray) and item.dtype == bool:
            item = np.argwhere(item).flatten()
        if isinstance(item, np.ndarray):
            return Polygons([self._polygons[int(index)] for index in item])
        raise TypeError('Polygons indexes must be an int or a boolean array.')

    @property
    def area(self) -> np.ndarray:
        return np.array([polygon.area for polygon in self._polygons])

    @classmethod
    def from_image(cls, image: np.ndarray) -> 'Polygons':
        if not isinstance(image, np.ndarray):
            raise TypeError('Input image must be a np.ndarray.')
        contours, _hierarchy = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if contour.shape[0] > 1]
        return cls(contours)


def imwarp_quadrangle(img: np.ndarray, polygon: Union[Polygon, np.ndarray]) -> np.ndarray:
    """Perspective-warp ``polygon`` so the MRZ band is horizontal."""
    if isinstance(polygon, np.ndarray):
        polygon = Polygon(polygon)
    if not isinstance(polygon, Polygon):
        raise TypeError(f'Input type of polygon {type(polygon)} not supported.')
    if len(polygon) != 4:
        raise ValueError('Input polygon, which is not contain 4 points is invalid.')

    width, height = polygon.min_box_wh
    if width < height:
        width, height = height, width
    width = int(width)
    height = int(height)
    source = order_points_clockwise(polygon.numpy()).astype(np.float32)
    destination = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height],
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(img, matrix, (width, height))

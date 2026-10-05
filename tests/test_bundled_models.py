from pathlib import Path

import mrzscanner
from mrzscanner.det.infer import Inference as Detection
from mrzscanner.rec.infer import Inference as Recognition
from mrzscanner.spotting.infer import Inference as Spotting


def test_weights_ship_with_the_package():
    """Each published model config points at an ONNX file in the package."""
    package = Path(mrzscanner.__file__).resolve().parent
    for inference in (Detection, Recognition, Spotting):
        component = inference.__module__.split('.')[1]
        for config in inference.configs.values():
            assert 'file_id' not in config
            weight = package / component / 'ckpt' / config['model_path']
            assert weight.is_file()
            assert weight.stat().st_size > 1_000_000


def test_library_source_does_not_fetch_google_drive():
    """The installed package must not contain a Google Drive download path."""
    package = Path(mrzscanner.__file__).resolve().parent
    for path in package.rglob('*.py'):
        text = path.read_text(encoding='utf-8')
        assert 'download_from_google' not in text
        assert 'drive.google.com' not in text

import sys
from pathlib import Path

import mrzscanner
from mrzscanner.det.infer import Inference as Detection
from mrzscanner.rec.infer import Inference as Recognition
from mrzscanner.runtime import bundled_model
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


def test_slugignore_keeps_the_packaged_weights():
    """The Heroku slug must keep the ONNX files the process imports."""
    slugignore = Path(mrzscanner.__file__).resolve().parents[1] / '.slugignore'
    patterns = [
        line.strip()
        for line in slugignore.read_text(encoding='utf-8').splitlines()
        if line.strip() and not line.lstrip().startswith('#')
    ]
    assert not any('onnx' in line for line in patterns)


def test_bundled_model_uses_installed_copy_when_checkout_was_stripped(tmp_path, monkeypatch):
    """A checkout without weights can still load the installed package copy."""
    checkout = tmp_path / 'app' / 'mrzscanner' / 'det'
    checkout.mkdir(parents=True)
    installed = tmp_path / 'venv' / 'mrzscanner' / 'det' / 'ckpt'
    installed.mkdir(parents=True)
    weight = installed / 'model.onnx'
    weight.write_bytes(b'weights')
    monkeypatch.setattr(sys, 'path', [str(tmp_path / 'app'), str(tmp_path / 'venv')])

    assert bundled_model(checkout, 'model.onnx') == weight


def test_library_source_does_not_fetch_google_drive():
    """The installed package must not contain a Google Drive download path."""
    package = Path(mrzscanner.__file__).resolve().parent
    for path in package.rglob('*.py'):
        text = path.read_text(encoding='utf-8')
        assert 'download_from_google' not in text
        assert 'drive.google.com' not in text

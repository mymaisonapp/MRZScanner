import cv2
import numpy as np
from fastapi.testclient import TestClient

from mrzscanner import ErrorCodes
from mrzscanner.api import create_app

TD3_LINE_1 = 'P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<'
TD3_LINE_2 = 'L898902C36UTO7408122F1204159ZE184226B<<<<<10'


class FakeScanner:
    """Callable stand-in for MRZScanner."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, image, do_center_crop=False, do_postprocess=False):
        assert isinstance(image, np.ndarray)
        key = (do_center_crop, do_postprocess)
        self.calls.append(key)
        return self.responses[key]


def _jpeg_bytes():
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode('.jpg', image)
    assert ok
    return encoded.tobytes()


def _result(lines, message=ErrorCodes.NO_ERROR):
    return {
        'mrz_polygon': np.array([[1.5, 2.0], [8.0, 2.0], [8.0, 4.0], [1.5, 4.0]], dtype=np.float32),
        'mrz_texts': lines,
        'msg': message,
    }


def _client(scanner):
    app = create_app(lambda: scanner)
    return TestClient(app)


def test_health():
    """The health route does not load a recognition model."""
    with _client(FakeScanner({})) as client:
        response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_root_points_at_the_scan_endpoint():
    """Opening the service URL explains where to send a document."""
    with _client(FakeScanner({})) as client:
        response = client.get('/')
    assert response.status_code == 200
    body = response.json()
    assert body['scan'] == '/v1/scan'
    assert body['docs'] == '/docs'


def test_main_keeps_a_single_worker_on_heroku(monkeypatch):
    """WEB_CONCURRENCY must not turn one dyno into multiple model copies."""
    import uvicorn

    captured = {}

    def fake_run(*args, **kwargs):
        captured['args'] = args
        captured['kwargs'] = kwargs

    monkeypatch.setenv('WEB_CONCURRENCY', '2')
    monkeypatch.setenv('PORT', '4321')
    monkeypatch.setattr(uvicorn, 'run', fake_run)

    from mrzscanner.api import main
    main()

    assert captured['args'] == ('mrzscanner.api:app',)
    assert captured['kwargs']['workers'] == 1
    assert captured['kwargs']['port'] == 4321
    assert captured['kwargs']['host'] == '0.0.0.0'


def test_scan_returns_raw_mrz_and_parsed_fields():
    """A document image comes back as raw MRZ text plus parsed JSON."""
    scanner = FakeScanner({
        (False, True): _result([TD3_LINE_1, TD3_LINE_2]),
    })
    with _client(scanner) as client:
        response = client.post(
            '/v1/scan',
            files={'image': ('passport.jpg', _jpeg_bytes(), 'image/jpeg')},
        )

    assert response.status_code == 200
    body = response.json()
    assert body['raw_mrz'] == f'{TD3_LINE_1}\n{TD3_LINE_2}'
    assert body['mrz_lines'] == [TD3_LINE_1, TD3_LINE_2]
    assert body['parsed']['surname'] == 'ERIKSSON'
    assert body['parsed']['given_names'] == 'ANNA MARIA'
    assert body['parsed']['document_number'] == 'L898902C3'
    assert body['parsed']['valid'] is True
    assert body['message'] == 'No error.'
    assert body['parse_error'] is None
    assert body['mrz_polygon'][0] == [1.5, 2.0]
    assert scanner.calls == [(False, True)]


def test_auto_retries_until_check_digits_pass():
    """A failed first read is replaced by a later read that validates."""
    scanner = FakeScanner({
        (False, True): _result(['NOT-AN-MRZ']),
        (False, False): _result(['STILLBAD', 'ALSOBAD']),
        (True, True): _result([TD3_LINE_1, TD3_LINE_2]),
    })
    with _client(scanner) as client:
        response = client.post(
            '/v1/scan',
            files={'image': ('passport.jpg', _jpeg_bytes(), 'image/jpeg')},
        )

    assert response.status_code == 200
    assert response.json()['parsed']['valid'] is True
    assert scanner.calls == (
        [(False, True)] * 4
        + [(False, False), (True, True)]
    )


def test_sideways_document_is_rotated_until_it_validates():
    """A card photographed on its side is read after a quarter turn."""

    class OrientationScanner:
        def __init__(self):
            self.shapes = []

        def __call__(self, image, do_center_crop=False, do_postprocess=False):
            self.shapes.append(image.shape[:2])
            if image.shape[0] > image.shape[1]:
                return _result([TD3_LINE_1, TD3_LINE_2])
            return _result(['NOT-AN-MRZ'])

    scanner = OrientationScanner()
    with _client(scanner) as client:
        response = client.post(
            '/v1/scan',
            files={'image': ('card.jpg', _jpeg_bytes(), 'image/jpeg')},
        )

    body = response.json()
    assert response.status_code == 200
    assert body['parsed']['valid'] is True
    assert scanner.shapes[0] == (32, 48)
    assert scanner.shapes[1][0] > scanner.shapes[1][1]
    assert body['mrz_polygon'][0] == [45.0, 1.5]


def test_auto_can_be_disabled():
    """Without auto, only the requested scanner settings run."""
    scanner = FakeScanner({
        (False, True): _result(['NOT-AN-MRZ']),
    })
    with _client(scanner) as client:
        response = client.post(
            '/v1/scan?auto=false',
            files={'image': ('passport.jpg', _jpeg_bytes(), 'image/jpeg')},
        )

    body = response.json()
    assert response.status_code == 200
    assert body['raw_mrz'] == 'NOT-AN-MRZ'
    assert body['parsed'] is None
    assert body['parse_error']
    assert scanner.calls == [(False, True)]


def test_invalid_image_is_rejected():
    """Non-image bytes are a client error and never reach the scanner."""
    scanner = FakeScanner({})
    with _client(scanner) as client:
        response = client.post(
            '/v1/scan',
            files={'image': ('notes.txt', b'not an image', 'text/plain')},
        )

    assert response.status_code == 400
    assert scanner.calls == []


def test_empty_upload_is_rejected():
    """An empty file is rejected before scanning."""
    with _client(FakeScanner({})) as client:
        response = client.post(
            '/v1/scan',
            files={'image': ('empty.jpg', b'', 'image/jpeg')},
        )
    assert response.status_code == 400


def test_missing_mrz_is_unprocessable():
    """A decoded image with no MRZ text returns 422 and an empty raw field."""
    scanner = FakeScanner({
        (False, True): _result(['']),
        (False, False): _result(None),
        (True, True): _result(['   ']),
        (True, False): _result([]),
    })
    with _client(scanner) as client:
        response = client.post(
            '/v1/scan',
            files={'image': ('blank.jpg', _jpeg_bytes(), 'image/jpeg')},
        )

    assert response.status_code == 422
    body = response.json()
    assert body['raw_mrz'] == ''
    assert body['parsed'] is None
    assert body['message'] == 'No MRZ was found in the image.'

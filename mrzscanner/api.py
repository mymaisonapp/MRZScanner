"""HTTP API for scanning a document image into raw and parsed MRZ data."""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from typing import Callable, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from . import __version__
from .scanner import MRZScanner
from .service import empty_mrz_message, scan_image

MAX_IMAGE_BYTES = 20 * 1024 * 1024
_INIT_LOCK = threading.Lock()


def _default_scanner() -> MRZScanner:
    return MRZScanner()


def create_app(
    scanner_factory: Optional[Callable] = None,
    *,
    warmup: bool = False,
) -> FastAPI:
    """Build the MRZ scanning API.

    `scanner_factory` replaces model loading in tests. `warmup` loads the
    recognition models during startup instead of on the first scan.
    """
    factory = scanner_factory or _default_scanner

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.scanner_factory = factory
        app.state.scanner = factory() if warmup else None
        yield

    app = FastAPI(
        title='MRZ Scanner',
        version=__version__,
        summary='Scan a travel document image and read its machine-readable zone.',
        description=(
            'Upload a passport, visa, or identity-card image. '
            'The response contains the raw MRZ text and the fields parsed '
            'from ICAO 9303 (TD1, TD2, and TD3), including check digits.'
        ),
        lifespan=lifespan,
    )

    @app.get('/health')
    def health():
        """Report that the API process is running."""
        return {'status': 'ok'}

    @app.post(
        '/v1/scan',
        summary='Scan a document image',
        responses={
            400: {'description': 'The upload is empty or not a readable image.'},
            413: {'description': 'The upload is larger than 20 MB.'},
            422: {'description': 'The image was decoded but no MRZ was found.'},
        },
    )
    async def scan_document(
        request: Request,
        image: UploadFile = File(
            ...,
            description='Passport, visa, or ID card image (JPEG, PNG, WEBP, or similar).',
        ),
        do_center_crop: bool = Query(
            False,
            description='Center-crop the image before scanning. Leave this off when the MRZ is near the edge.',
        ),
        do_postprocess: bool = Query(
            True,
            description='Correct characters that are illegal in a given MRZ field.',
        ),
        auto: bool = Query(
            True,
            description='Retry other crop and post-process settings when check digits fail.',
        ),
    ):
        """Read the MRZ from an uploaded document image."""
        payload = await _read_image(image)
        result = scan_image(
            _get_scanner(request.app),
            payload,
            do_center_crop=do_center_crop,
            do_postprocess=do_postprocess,
            auto=auto,
        )
        if not result['raw_mrz']:
            result['message'] = empty_mrz_message(result)
            return JSONResponse(status_code=422, content=result)
        return result

    return app


async def _read_image(upload: UploadFile) -> np.ndarray:
    contents = await upload.read()
    if not contents:
        raise HTTPException(status_code=400, detail='Empty image upload.')
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail='Image exceeds the 20 MB limit.')

    encoded = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail='Could not decode the image. Upload a JPEG, PNG, or similar file.',
        )
    return image


def _get_scanner(app: FastAPI):
    if app.state.scanner is not None:
        return app.state.scanner
    with _INIT_LOCK:
        if app.state.scanner is None:
            app.state.scanner = app.state.scanner_factory()
        return app.state.scanner


app = create_app()


def main() -> None:
    """Run the API. Models load before the server accepts requests."""
    import uvicorn

    host = os.environ.get('MRZ_HOST', '0.0.0.0')
    # Heroku sets PORT. MRZ_PORT remains available for local runs.
    port = int(os.environ.get('PORT', os.environ.get('MRZ_PORT', '8000')))
    uvicorn.run(
        create_app(warmup=True),
        host=host,
        port=port,
    )


if __name__ == '__main__':
    main()

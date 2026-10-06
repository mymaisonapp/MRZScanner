"""Turn an MRZScanner result into JSON-ready raw text and parsed fields."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .parser import MRZParseError, parse_mrz
from .scanner import ErrorCodes

Attempt = Tuple[bool, bool]

# Non-cropped, post-processed reads are tried first. Center-cropping a photo
# whose MRZ sits near the edge clips the zone, so it is only a fallback.
_FALLBACK_ATTEMPTS: Sequence[Attempt] = (
    (False, True),
    (False, False),
    (True, True),
    (True, False),
)


def scan_image(
    scanner,
    image: np.ndarray,
    *,
    do_center_crop: bool = False,
    do_postprocess: bool = True,
    auto: bool = True,
) -> Dict:
    """Scan `image` and return the raw MRZ plus parsed ICAO fields.

    When `auto` is set and a read does not pass every check digit, the other
    orientations are tried before the other crop and post-process settings.
    The fully valid read wins; otherwise the read with the most valid check
    digits is returned. Polygon points are mapped back onto the uploaded image.
    """
    primary = (bool(do_center_crop), bool(do_postprocess))
    attempts: List[Attempt] = [primary]
    if auto:
        for attempt in _FALLBACK_ATTEMPTS:
            if attempt not in attempts:
                attempts.append(attempt)

    best_payload = None
    best_score = None

    def consider(frame, rotation, crop, postprocess) -> bool:
        nonlocal best_payload, best_score
        result = scanner(
            frame,
            do_center_crop=crop,
            do_postprocess=postprocess,
        )
        payload = build_scan_payload(result)
        payload['mrz_polygon'] = _unmap_polygon(payload['mrz_polygon'], rotation)
        score = _score(payload)
        if best_score is None or score > best_score:
            best_payload = payload
            best_score = score
        return bool((payload.get('parsed') or {}).get('valid'))

    if consider(image, None, primary[0], primary[1]):
        return best_payload
    if not auto:
        return best_payload

    height, width = image.shape[:2]
    rotations = (
        (cv2.ROTATE_90_COUNTERCLOCKWISE, 'ccw'),
        (cv2.ROTATE_90_CLOCKWISE, 'cw'),
        (cv2.ROTATE_180, 'half'),
    )
    for rotate_code, kind in rotations:
        frame = cv2.rotate(image, rotate_code)
        if consider(frame, (kind, width, height), primary[0], primary[1]):
            return best_payload

    for crop, postprocess in attempts:
        if (crop, postprocess) == primary:
            continue
        if consider(image, None, crop, postprocess):
            return best_payload

    return best_payload


def _unmap_polygon(polygon, rotation):
    """Map a polygon from a rotated frame back to the uploaded image."""
    if not polygon or rotation is None:
        return polygon
    kind, width, height = rotation
    mapped = []
    for x_coord, y_coord in polygon:
        if kind == 'ccw':
            mapped.append([float(width - 1 - y_coord), float(x_coord)])
        elif kind == 'cw':
            mapped.append([float(y_coord), float(height - 1 - x_coord)])
        else:
            mapped.append([
                float(width - 1 - x_coord),
                float(height - 1 - y_coord),
            ])
    return mapped


def build_scan_payload(result: Dict) -> Dict:
    """Convert one scanner result into a JSON-serializable payload."""
    lines = coerce_mrz_lines(result.get('mrz_texts'))
    parsed = None
    parse_error = None
    if lines:
        try:
            parsed = parse_mrz(lines)
        except MRZParseError as exc:
            parse_error = str(exc)

    return {
        'raw_mrz': '\n'.join(lines),
        'mrz_lines': lines,
        'parsed': parsed,
        'mrz_polygon': coerce_polygon(result.get('mrz_polygon')),
        'message': error_message(result.get('msg')),
        'parse_error': parse_error,
    }


def coerce_mrz_lines(mrz_texts) -> List[str]:
    """Normalize scanner text output to a list of MRZ lines."""
    if mrz_texts is None:
        return []
    if isinstance(mrz_texts, str):
        parts = mrz_texts.splitlines()
    else:
        parts = list(mrz_texts)

    lines = []
    for part in parts:
        text = str(part).strip().upper()
        if text:
            lines.append(text)
    return lines


def coerce_polygon(polygon) -> Optional[List[List[float]]]:
    """Convert a detection polygon to plain floats, preserving None."""
    if polygon is None:
        return None
    points = np.asarray(polygon, dtype=float)
    if points.size == 0:
        return []
    points = points.reshape(-1, 2)
    return [[float(x_coord), float(y_coord)] for x_coord, y_coord in points]


def error_message(msg) -> str:
    """Stringify an ErrorCodes value."""
    if msg is None:
        return ''
    return getattr(msg, 'value', str(msg))


def _score(payload: Dict) -> Tuple:
    parsed = payload.get('parsed') or {}
    checks = parsed.get('checks') or {}
    valid_checks = 0
    for item in checks.values():
        if isinstance(item, dict) and item.get('valid'):
            valid_checks += 1
    return (
        1 if parsed.get('valid') else 0,
        valid_checks,
        1 if payload.get('raw_mrz') else 0,
        len(payload.get('mrz_lines') or []),
    )


def empty_mrz_message(payload: Dict) -> str:
    """Message used when the scanner returns no MRZ text."""
    if payload.get('message') in ('', ErrorCodes.NO_ERROR.value):
        return 'No MRZ was found in the image.'
    return payload['message']

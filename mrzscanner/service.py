"""Turn an MRZScanner result into JSON-ready raw text and parsed fields."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

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

    When `auto` is set and the first read does not pass every check digit,
    the other crop and post-process combinations are tried. The fully valid
    read wins; otherwise the read with the most valid check digits is returned.
    """
    primary = (bool(do_center_crop), bool(do_postprocess))
    attempts: List[Attempt] = [primary]
    if auto:
        for attempt in _FALLBACK_ATTEMPTS:
            if attempt not in attempts:
                attempts.append(attempt)

    best_payload = None
    best_score = None
    for crop, postprocess in attempts:
        result = scanner(
            image.copy(),
            do_center_crop=crop,
            do_postprocess=postprocess,
        )
        payload = build_scan_payload(result)
        score = _score(payload)
        if best_score is None or score > best_score:
            best_payload = payload
            best_score = score
        parsed = payload.get('parsed') or {}
        if parsed.get('valid'):
            return payload

    return best_payload


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

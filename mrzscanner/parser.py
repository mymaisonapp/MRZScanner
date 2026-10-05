"""Parse ICAO 9303 machine-readable zones into structured fields."""

from __future__ import annotations

import re
from datetime import date
from typing import Dict, List, Optional, Sequence, Union

MRZInput = Union[str, Sequence[str]]

_WEIGHTS = (7, 3, 1)
_CHAR_VALUES = {str(digit): digit for digit in range(10)}
_CHAR_VALUES.update({
    letter: index
    for index, letter in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ', start=10)
})
_CHAR_VALUES['<'] = 0

_DOCUMENT_TYPE_NAMES = {
    'P': 'passport',
    'I': 'identity_card',
    'A': 'residence_permit',
    'C': 'identity_card',
    'V': 'visa',
}

_DATE_PATTERN = re.compile(r'\d{6}')


class MRZParseError(ValueError):
    """Raised when text is not a supported TD1, TD2, or TD3 MRZ."""


def calculate_check_digit(value: str) -> Optional[str]:
    """Return the ICAO 7-3-1 check digit, or None when `value` has an invalid character."""
    total = 0
    for index, char in enumerate(value):
        try:
            char_value = _CHAR_VALUES[char]
        except KeyError:
            return None
        total += char_value * _WEIGHTS[index % 3]
    return str(total % 10)


def parse_mrz(mrz: MRZInput, *, today: Optional[date] = None) -> Dict:
    """Parse two-line TD2/TD3 or three-line TD1 MRZ text.

    `today` controls the century chosen for a date of birth. Expiry dates are
    always read as 20YY, which matches travel documents currently in circulation.
    """
    lines = _normalize_lines(mrz)
    reference = today or date.today()
    lengths = tuple(len(line) for line in lines)

    if lengths == (44, 44):
        return _parse_td3(lines, reference)
    if lengths == (36, 36):
        return _parse_td2(lines, reference)
    if lengths == (30, 30, 30):
        return _parse_td1(lines, reference)

    raise MRZParseError(
        f'Unsupported MRZ layout: {len(lines)} line(s) with lengths {list(lengths)}.'
    )


def _normalize_lines(mrz: MRZInput) -> List[str]:
    if isinstance(mrz, str):
        raw_lines = mrz.splitlines()
    else:
        raw_lines = list(mrz)

    lines = []
    for raw_line in raw_lines:
        text = str(raw_line).strip().upper()
        if text:
            lines.append(text)
    if not lines:
        raise MRZParseError('MRZ text is empty.')
    return lines


def _parse_td3(lines: Sequence[str], today: date) -> Dict:
    line1, line2 = lines
    # Composite input skips nationality (10:13) and sex (20).
    composite_source = line2[0:10] + line2[13:20] + line2[21:43]
    return _build_record(
        fmt='TD3',
        document_type=line1[0],
        document_subtype=line1[1],
        issuing_country=line1[2:5],
        name_field=line1[5:44],
        number_field=line2[0:9],
        number_digit=line2[9],
        nationality=line2[10:13],
        birth=line2[13:19],
        birth_digit=line2[19],
        sex=line2[20],
        expiry=line2[21:27],
        expiry_digit=line2[27],
        optional_fields=(line2[28:42],),
        optional_digit=line2[42],
        composite_source=composite_source,
        composite_digit=line2[43],
        today=today,
    )


def _parse_td2(lines: Sequence[str], today: date) -> Dict:
    line1, line2 = lines
    # TD2 optional data has no check digit of its own; it is part of the composite.
    composite_source = line2[0:10] + line2[13:20] + line2[21:35]
    return _build_record(
        fmt='TD2',
        document_type=line1[0],
        document_subtype=line1[1],
        issuing_country=line1[2:5],
        name_field=line1[5:36],
        number_field=line2[0:9],
        number_digit=line2[9],
        nationality=line2[10:13],
        birth=line2[13:19],
        birth_digit=line2[19],
        sex=line2[20],
        expiry=line2[21:27],
        expiry_digit=line2[27],
        optional_fields=(line2[28:35],),
        optional_digit=None,
        composite_source=composite_source,
        composite_digit=line2[35],
        today=today,
    )


def _parse_td1(lines: Sequence[str], today: date) -> Dict:
    line1, line2, line3 = lines
    composite_source = line1[5:30] + line2[0:7] + line2[8:15] + line2[18:29]
    return _build_record(
        fmt='TD1',
        document_type=line1[0],
        document_subtype=line1[1],
        issuing_country=line1[2:5],
        name_field=line3[0:30],
        number_field=line1[5:14],
        number_digit=line1[14],
        nationality=line2[15:18],
        birth=line2[0:6],
        birth_digit=line2[6],
        sex=line2[7],
        expiry=line2[8:14],
        expiry_digit=line2[14],
        optional_fields=(line1[15:30], line2[18:29]),
        optional_digit=None,
        composite_source=composite_source,
        composite_digit=line2[29],
        today=today,
    )


def _build_record(
    *,
    fmt: str,
    document_type: str,
    document_subtype: str,
    issuing_country: str,
    name_field: str,
    number_field: str,
    number_digit: str,
    nationality: str,
    birth: str,
    birth_digit: str,
    sex: str,
    expiry: str,
    expiry_digit: str,
    optional_fields: Sequence[str],
    optional_digit: Optional[str],
    composite_source: str,
    composite_digit: str,
    today: date,
) -> Dict:
    surname, given_names = _split_name(name_field)
    optional_data = _join_optional(optional_fields)
    date_of_birth = _parse_yymmdd(birth, birth=True, today=today)
    date_of_expiry = _parse_yymmdd(expiry, birth=False, today=today)
    optional_check = None
    if optional_digit is not None:
        optional_check = _evaluate_check(
            optional_fields[0], optional_digit, allow_filler=True)

    checks = {
        'document_number': _evaluate_check(number_field, number_digit),
        'date_of_birth': _evaluate_check(birth, birth_digit),
        'date_of_expiry': _evaluate_check(expiry, expiry_digit),
        'optional_data': optional_check,
        'composite': _evaluate_check(composite_source, composite_digit),
    }
    check_values = [item['valid'] for item in checks.values() if item is not None]
    dates_ok = date_of_birth is not None and date_of_expiry is not None

    return {
        'format': fmt,
        'document_type': document_type,
        'document_subtype': document_subtype,
        'document_type_name': _DOCUMENT_TYPE_NAMES.get(document_type),
        'issuing_country': issuing_country,
        'surname': surname,
        'given_names': given_names,
        'document_number': number_field.rstrip('<'),
        'nationality': nationality,
        'date_of_birth': date_of_birth,
        'date_of_birth_raw': birth,
        'sex': None if sex == '<' else sex,
        'date_of_expiry': date_of_expiry,
        'date_of_expiry_raw': expiry,
        'optional_data': optional_data,
        'checks': checks,
        'valid': dates_ok and all(check_values),
    }


def _evaluate_check(field: str, digit: str, *, allow_filler: bool = False) -> Dict:
    calculated = calculate_check_digit(field)
    filler_ok = allow_filler and digit == '<' and set(field) <= {'<'}
    return {
        'digit': digit,
        'calculated': calculated,
        'valid': filler_ok or (calculated is not None and calculated == digit),
    }


def _split_name(name_field: str) -> tuple:
    primary, separator, secondary = name_field.partition('<<')
    surname = _fillers_to_spaces(primary)
    given_names = _fillers_to_spaces(secondary) if separator else ''
    return surname, given_names


def _fillers_to_spaces(value: str) -> str:
    return ' '.join(value.replace('<', ' ').split())


def _join_optional(fields: Sequence[str]) -> str:
    parts = [field.rstrip('<') for field in fields]
    return ' '.join(part for part in parts if part)


def _parse_yymmdd(value: str, *, birth: bool, today: date) -> Optional[str]:
    if _DATE_PATTERN.fullmatch(value) is None:
        return None
    year = 2000 + int(value[0:2])
    month = int(value[2:4])
    day = int(value[4:6])
    try:
        parsed = date(year, month, day)
    except ValueError:
        return None
    if birth and parsed > today:
        try:
            parsed = date(year - 100, month, day)
        except ValueError:
            return None
    return parsed.isoformat()

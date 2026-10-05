from datetime import date

import pytest

from mrzscanner import MRZParseError, calculate_check_digit, parse_mrz

# ICAO specimen passport from the machine-readable passport examples.
TD3_LINE_1 = 'P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<'
TD3_LINE_2 = 'L898902C36UTO7408122F1204159ZE184226B<<<<<10'

# TD1 specimen. Composite digit is part of the published example.
TD1_LINE_1 = 'I<UTOD231458907<<<<<<<<<<<<<<<'
TD1_LINE_2 = '7408122F1204159UTO<<<<<<<<<<<6'
TD1_LINE_3 = 'ERIKSSON<<ANNA<MARIA<<<<<<<<<<'

TODAY = date(2026, 10, 5)


def test_check_digit_vectors():
    """ICAO 7-3-1 weights for the specimen passport fields."""
    assert calculate_check_digit('L898902C3') == '6'
    assert calculate_check_digit('740812') == '2'
    assert calculate_check_digit('120415') == '9'
    assert calculate_check_digit('ZE184226B<<<<<') == '1'
    assert calculate_check_digit('') == '0'
    assert calculate_check_digit('L898902C3*') is None


def test_parse_td3_specimen():
    """A TD3 passport yields names, dates, and a valid check-digit set."""
    parsed = parse_mrz(f'{TD3_LINE_1}\n{TD3_LINE_2}', today=TODAY)

    assert parsed['format'] == 'TD3'
    assert parsed['document_type'] == 'P'
    assert parsed['document_subtype'] == '<'
    assert parsed['document_type_name'] == 'passport'
    assert parsed['issuing_country'] == 'UTO'
    assert parsed['surname'] == 'ERIKSSON'
    assert parsed['given_names'] == 'ANNA MARIA'
    assert parsed['document_number'] == 'L898902C3'
    assert parsed['nationality'] == 'UTO'
    assert parsed['date_of_birth'] == '1974-08-12'
    assert parsed['date_of_birth_raw'] == '740812'
    assert parsed['sex'] == 'F'
    assert parsed['date_of_expiry'] == '2012-04-15'
    assert parsed['optional_data'] == 'ZE184226B'
    assert parsed['checks']['document_number']['valid'] is True
    assert parsed['checks']['date_of_birth']['valid'] is True
    assert parsed['checks']['date_of_expiry']['valid'] is True
    assert parsed['checks']['optional_data']['valid'] is True
    assert parsed['checks']['composite']['digit'] == '0'
    assert parsed['checks']['composite']['valid'] is True
    assert parsed['valid'] is True


def test_birth_date_century_pivot():
    """Birth dates that would land in the future roll back one century."""
    line2 = 'L898902C36UTO0501011F1204159ZE184226B<<<<<10'
    # 050101 is 2005, which is not in the future relative to TODAY.
    recent = parse_mrz([TD3_LINE_1, line2], today=TODAY)
    assert recent['date_of_birth'] == '2005-01-01'
    assert recent['checks']['date_of_birth']['valid'] is False

    line2_old = 'L898902C36UTO7911138F1204159ZE184226B<<<<<10'
    older = parse_mrz([TD3_LINE_1, line2_old], today=TODAY)
    assert older['date_of_birth'] == '1979-11-13'


def test_parse_td1_specimen():
    """A three-line TD1 card keeps the name on the final line."""
    parsed = parse_mrz([TD1_LINE_1, TD1_LINE_2, TD1_LINE_3], today=TODAY)

    assert parsed['format'] == 'TD1'
    assert parsed['document_type_name'] == 'identity_card'
    assert parsed['surname'] == 'ERIKSSON'
    assert parsed['given_names'] == 'ANNA MARIA'
    assert parsed['document_number'] == 'D23145890'
    assert parsed['nationality'] == 'UTO'
    assert parsed['date_of_birth'] == '1974-08-12'
    assert parsed['sex'] == 'F'
    assert parsed['date_of_expiry'] == '2012-04-15'
    assert parsed['optional_data'] == ''
    assert parsed['checks']['optional_data'] is None
    assert parsed['checks']['document_number']['valid'] is True
    assert parsed['checks']['composite']['valid'] is True
    assert parsed['valid'] is True


def test_parse_td2_round_trip_fields():
    """TD2 uses 36-character lines and has no separate optional-data check."""
    line1 = 'V<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<'
    number = 'L898902C3'
    birth = '740812'
    expiry = '120415'
    optional = '<<<<<<<'
    body = (
        f'{number}{calculate_check_digit(number)}'
        f'UTO{birth}{calculate_check_digit(birth)}'
        f'F{expiry}{calculate_check_digit(expiry)}'
        f'{optional}'
    )
    composite_source = body[0:10] + body[13:20] + body[21:35]
    line2 = body + calculate_check_digit(composite_source)
    assert len(line1) == 36
    assert len(line2) == 36

    parsed = parse_mrz([line1, line2], today=TODAY)
    assert parsed['format'] == 'TD2'
    assert parsed['document_type_name'] == 'visa'
    assert parsed['surname'] == 'ERIKSSON'
    assert parsed['given_names'] == 'ANNA MARIA'
    assert parsed['document_number'] == 'L898902C3'
    assert parsed['checks']['optional_data'] is None
    assert parsed['valid'] is True


def test_tampered_check_digit_is_invalid():
    """A wrong composite digit is reported and fails validation."""
    tampered = TD3_LINE_2[:-1] + '1'
    parsed = parse_mrz([TD3_LINE_1, tampered], today=TODAY)
    assert parsed['checks']['composite']['valid'] is False
    assert parsed['valid'] is False
    assert parsed['surname'] == 'ERIKSSON'


def test_filler_optional_check_is_accepted():
    """An unused TD3 optional field may use '<' as its check digit."""
    optional = '<' * 14
    prefix = TD3_LINE_2[:28]
    composite_source = prefix[0:10] + prefix[13:20] + prefix[21:28] + optional + '<'
    line2 = prefix + optional + '<' + calculate_check_digit(composite_source)
    parsed = parse_mrz([TD3_LINE_1, line2], today=TODAY)
    assert parsed['optional_data'] == ''
    assert parsed['checks']['optional_data']['digit'] == '<'
    assert parsed['checks']['optional_data']['valid'] is True
    assert parsed['valid'] is True


def test_unsupported_layout():
    """Lines that are not TD1, TD2, or TD3 are rejected."""
    with pytest.raises(MRZParseError, match='Unsupported MRZ layout'):
        parse_mrz(['ABCDEF', 'GHIJKL'])


def test_empty_mrz():
    """Blank input is rejected."""
    with pytest.raises(MRZParseError, match='empty'):
        parse_mrz('   \n  ')


def test_invalid_calendar_date_fails_validation():
    """Check digits can match while the date itself is impossible."""
    birth = '741332'
    prefix = 'L898902C36UTO'
    rest_after_birth_check = 'F1204159ZE184226B<<<<<10'
    line2 = prefix + birth + calculate_check_digit(birth) + rest_after_birth_check
    # Rebuild the composite so only the calendar date is wrong.
    line2 = line2[:43]
    composite_source = line2[0:10] + line2[13:20] + line2[21:43]
    line2 = line2 + calculate_check_digit(composite_source)
    assert len(line2) == 44

    parsed = parse_mrz([TD3_LINE_1, line2], today=TODAY)
    assert parsed['date_of_birth'] is None
    assert parsed['checks']['date_of_birth']['valid'] is True
    assert parsed['valid'] is False

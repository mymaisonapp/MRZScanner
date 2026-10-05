from .parser import MRZParseError, calculate_check_digit, parse_mrz
from .scanner import ErrorCodes, ModelType, MRZScanner, SpottingInference
from .utils import replace_digits, replace_letters, replace_sex

__version__ = '1.1.0'

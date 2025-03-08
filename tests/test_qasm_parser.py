import math

from gospel.scripts.qasm_parser import parse_angle


def test_parse_angle() -> None:
    assert parse_angle("2*pi/3") == 2 * math.pi / 3

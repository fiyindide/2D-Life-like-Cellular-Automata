import numpy as np
from rule import encode, decode, parse, format_rule


def test_encode_decode_roundtrip():
    B = [3]
    S = [2, 3]

    mask = encode(B, S)
    birth, survive = decode(mask)

    # Convert boolean arrays back to sets of indices
    B_decoded = {i for i, v in enumerate(birth) if v}
    S_decoded = {i for i, v in enumerate(survive) if v}

    assert B_decoded == set(B)
    assert S_decoded == set(S)


def test_parse_format_rule_roundtrip():
    rule_str = "B3/S23"

    B, S = parse(rule_str)
    mask = encode(B, S)
    birth, survive = decode(mask)
    formatted = format_rule(birth, survive)

    assert formatted == rule_str

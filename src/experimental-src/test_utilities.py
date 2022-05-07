import pytest

import utilities

def test_binarize():
    a_decimal = 4
    expected = [1, 0, 0]
    actual = utilities.binarize(a_decimal)
    assert actual == expected

def test_binarize_inverse():
    a_binarized = [1, 0, 0]
    expected = 4
    actual = utilities.binarize_inverse(a_binarized)
    assert actual == expected


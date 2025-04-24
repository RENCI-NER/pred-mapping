import pytest
from src.biolink_predicate_lookup import extract_mapped_predicate

def test_extract_valid_json_mapping():
    response = '{"mapped_predicate": "treats"}'
    choices = {"treats": "used to treat", "prevents": "used to prevent"}
    result = extract_mapped_predicate(response, choices)
    assert result == "biolink:treats"


def test_extract_loose_format():
    response = "mapped_predicate: 'prevents'"
    choices = {"treats": "used to treat", "prevents": "used to prevent"}
    result = extract_mapped_predicate(response, choices)
    assert result == "biolink:prevents"


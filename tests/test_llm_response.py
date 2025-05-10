import pytest
from src.biolink_predicate_lookup import extract_mapped_predicate


def test_extract_valid_json_mapping():
    response = '{"mapped_predicate": "treats", "negated": false}'
    choices = {"treats": "used to treat", "prevents": "used to prevent"}
    result = extract_mapped_predicate(response, choices)
    assert result.get("mapped_predicate", None) == "biolink:treats"

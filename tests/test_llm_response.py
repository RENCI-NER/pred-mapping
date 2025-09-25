import pytest
from src.biolink_predicate_lookup import extract_mapped_predicate


def test_extract_valid_json_mapping():
    response = '{"mapped_predicate": "treats", "negated": "False"}'
    result = extract_mapped_predicate(response)
    assert result.get("mapped_predicate", None) == "treats"

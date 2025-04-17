import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_query_endpoint():
    test_payload = [
        {
            "subject": "mutation",
            "object": "ALD gene",
            "abstract": (
                "Fragments of the adrenoleukodystrophy (ALD) cDNA from a patient with adolescent ALD "
                "were amplified by polymerase chain reaction and subcloned. Bidirectional sequencing of the "
                "entire coding ALD gene disclosed a cytosine to guanine transversion at nucleotide 1451 in exon five, "
                "resulting in substitution of proline 484 by arginine."
            ),
            "relationship": "found in"
        }
    ]

    response = client.post("/query/", json=test_payload)

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) == 1
    assert "top_choice" in data["results"][0]

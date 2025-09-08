# export PYTHONPATH="$PYTHONPATH:$PWD"
import os
import json
import pytest
from fastapi.testclient import TestClient
from src.chemprot_server import APP, RetrievalMethod

client = TestClient(APP)


def test_query_endpoint():
    """ query endpoint with SapBERT enabled"""
    DIR = os.path.dirname(os.path.abspath(__file__))
    with open(f"{DIR}/newest_chemprot_test_file.json") as f:
        test_payload = json.load(f)

    response = client.post("/query/", json=test_payload[:1], params={
        "retrieval_method": RetrievalMethod.knn.value,
        "use_sapbert": False
    })

    assert response.status_code == 200
    data = response.json()

    with open(f"{DIR}/chemprot_protocol_results_1024_dim_bge.json", "w") as f:
        json.dump(data, f, indent=4)

    assert "results" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) == len(test_payload)
    assert "top_choice" in data["results"][0]


def test_query_endpoint_with_sapbert():
    """ query endpoint with SapBERT enabled"""
    DIR = os.path.dirname(os.path.abspath(__file__))
    with open(f"{DIR}/newest_chemprot_test_file.json") as f:
        test_payload = json.load(f)

    response = client.post("/query/", json=test_payload, params={
        "retrieval_method": RetrievalMethod.knn.value,
        "use_sapbert": True
    })

    assert response.status_code == 200
    data = response.json()

    # with open(f"{DIR}/chemprot_protocol_results_1024_dim_bgeplus.json", "w") as f:
    #     json.dump(data, f, indent=4)

    assert "results" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) == len(test_payload)
    assert "top_choice" in data["results"][0]
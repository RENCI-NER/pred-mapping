import pytest
import torch
import asyncio
from unittest.mock import AsyncMock, MagicMock
from src.predicate_database import PredicateDatabase, transform_embedding

EMBEDDINGS = [
    {"predicate": "P1", "text": "Text about relationship", "embedding": [0.1] * 768},
    {"predicate": "P2", "text": "RE is cool", "embedding": [0.6] * 768},
    {"predicate": "P3", "text": "LitCoin Textual RELATE", "embedding": [0.3] * 768}
]


@pytest.fixture
def dummy_client():
    mock = MagicMock()
    mock.get_embedding = AsyncMock(return_value=torch.tensor([0.2] * 768, dtype=torch.float32))
    return mock


def test_populate_default(dummy_client):
    db = PredicateDatabase(dummy_client)
    db.populate_db(EMBEDDINGS)
    assert len(db.all_pred_emb) == 3
    assert db.all_pred[1] == "P2"


def test_search_default_mode(dummy_client):
    db = PredicateDatabase(dummy_client)
    db.populate_db(EMBEDDINGS)
    result = asyncio.run(db.search("RE is cool"))

    assert isinstance(result, dict)
    assert all("mapped_predicate" in val for val in result.values())


def test_search_nn_mode(dummy_client):
    db = PredicateDatabase(dummy_client)
    db.populate_db(EMBEDDINGS)
    result = asyncio.run(db.search("search for relationship", num_results=1))
    assert len(result) <= 3
    assert isinstance(result, dict)


def test_similarity_score_1(dummy_client):
    embedding_vector = [1.0] * 768
    db = PredicateDatabase(dummy_client)
    dummy_embeddings = [
        {"predicate": "ExactMatch", "text": "Same vector", "embedding": embedding_vector}
    ]
    dummy_client.get_embedding.return_value = transform_embedding(embedding_vector)
    db.populate_db(dummy_embeddings)

    results = asyncio.run(db.search("Any query", embedding=transform_embedding(embedding_vector), num_results=1))
    score = results[0]["score"]

    assert abs(score - 1.0) < 1e-6, f"Expected 1.0, got {score}"


def test_similarity_score_0(dummy_client):
    vec1 = [1.0] + [0.0] * 767
    vec2 = [0.0, 1.0] + [0.0] * 766

    db = PredicateDatabase(dummy_client)
    dummy_embeddings = [
        {"predicate": "Orthogonal", "text": "Orthogonal vector", "embedding": vec2}
    ]
    dummy_client.get_embedding.return_value = transform_embedding(vec1)
    db.populate_db(dummy_embeddings)

    results = asyncio.run(db.search("Orthogonal test", embedding=transform_embedding(vec1), num_results=1))
    score = results[0]["score"]

    assert abs(score - 0.0) < 1e-6, f"Expected 0.0, got {score}"


def test_search_performance(dummy_client):
    import time
    db = PredicateDatabase(dummy_client)
    db.populate_db(EMBEDDINGS * 10)
    db.is_nn=False
    db.is_vdb=False
    start = time.perf_counter()
    _ = asyncio.run(db.search("Text about relationship", num_results=5))
    duration = time.perf_counter() - start

    print(f"\nSearch took: {duration:.6f} seconds")
    assert duration < 1.0

import os
import time
import json
import pytest
import torch
import asyncio
from src.predicate_database import PredicateDatabase, transform_embedding


def is_ci_env():
    return os.environ.get("CI", "false").lower() == "true"


class DummyClient:
    async def get_embedding(self, _):
        return torch.tensor([0.5] * 768, dtype=torch.float32)


def get_embedding_client():
    if is_ci_env():
        return DummyClient()
    else:
        from src.biolink_predicate_lookup import PredicateClient
        return PredicateClient()


def benchmark_search(db, query="Test", num_results=10, embedding=None):
    start = time.perf_counter()
    result = asyncio.run(db.search(query, embedding=embedding, num_results=num_results))
    end = time.perf_counter()
    return end - start, result


def benchmark_with_real_data(json_path, queries, mode_name="default", **db_kwargs):
    client = get_embedding_client()
    db = PredicateDatabase(client, **db_kwargs)
    db.load_db_from_json(json_path)

    total_time = 0.0
    for query in queries:
        start = time.perf_counter()
        _ = asyncio.run(db.search(query, num_results=5))
        total_time += time.perf_counter() - start

    avg_time = total_time / len(queries)
    print("*******")
    print(f"{mode_name} avg latency: {avg_time:.6f} sec/query")


@pytest.mark.skipif(not is_ci_env(), reason="only runs in CI")
def test_main():
    EMBEDDINGS = [
        {"predicate": f"P{i}", "text": f"Text {i}", "embedding": [float(i % 2)] * 768}
        for i in range(1000)
    ]
    client = get_embedding_client()
    embedding = transform_embedding([0.5] * 768)

    print("\n...Benchmarking cosine method...")
    db = PredicateDatabase(client)
    db.populate_db(EMBEDDINGS)
    t, _ = benchmark_search(db, embedding=embedding)
    print(f"Cosine: {t:.6f} sec")

    print("...Benchmarking NearestNeighbors method...")
    db = PredicateDatabase(client, is_nn=True)
    db.populate_db(EMBEDDINGS)
    t, _ = benchmark_search(db, embedding=embedding)
    print(f"NearestNeighbors: {t:.6f} sec")

    print("...Benchmarking vector DB method...")
    db = PredicateDatabase(client, is_vdb=True)
    db.populate_db(EMBEDDINGS)
    t, _ = benchmark_search(db, embedding=embedding)
    print(f"VectorDB: {t:.6f} sec")


@pytest.mark.skipif(is_ci_env(), reason="only runs in locally")
def test_main2():
    DIR = os.path.dirname(os.path.abspath(__file__))
    with open(f"{DIR}/sample_input.json") as f:
        data = json.load(f)
    queries = [d["relationship"] for d in data[:20]]
    print(f"\nTotal Queries: {len(queries)}")
    benchmark_with_real_data("../data/all_biolink_mapped_vectors.json", queries, mode_name="Similarities")
    benchmark_with_real_data("../data/all_biolink_mapped_vectors.json", queries, mode_name="NN", is_nn=True)
    benchmark_with_real_data("../data/all_biolink_mapped_vectors.json", queries, mode_name="VDB", is_vdb=True)


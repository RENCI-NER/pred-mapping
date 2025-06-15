import argparse
import json
import asyncio
import time
import httpx
from tqdm import tqdm

from src.llm_client import HEALpacaClient


class EmbeddingClient(HEALpacaClient):
    def __init__(self, **kwargs):
        super().__init__()


async def embed_biolink_predicates(infile, outfile, use_lowercase=False):
    start_time = time.time()
    client = EmbeddingClient()
    with open(infile, "r") as file:
        data = json.load(file)

    results = []
    tasks = []
    max_conns = 10
    limits = httpx.Limits(max_connections=max_conns, max_keepalive_connections=5)
    async with httpx.AsyncClient(timeout=httpx.Timeout(max_conns * 60), limits=limits) as asynclient:
        for predicate, text_list in tqdm(data.items(), position=0, desc="Biolink predicates"):
            if isinstance(text_list, list) and text_list:
                for text in text_list:
                    if use_lowercase:
                        text = text.lower()
                    results.append({
                        "predicate": "biolink:" + predicate.strip().replace(" ", "_"),
                        "text": text})
                    tasks.append(asyncio.create_task(client.get_async_embedding(text, asynclient=asynclient, max_conns=max_conns)))
            elif isinstance(text_list, str):
                results.append({
                    "predicate": "biolink:" + predicate.strip().replace(" ", "_"),
                    "text": text_list})
                tasks.append(asyncio.create_task(client.get_async_embedding(predicate, text_list, asynclient=asynclient, max_conns=max_conns, use_lowercase=use_lowercase)))

        vectors = await asyncio.gather(*tasks, return_exceptions=True)

    for result, vector in zip(results, vectors):
        result["embedding"] = vector.get("embedding", [])

    with open(outfile, "w") as file:
        file.write(json.dumps(results, indent=2))

    end_time = time.time()
    print(f"Execution Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mappings", default="llama_all_biolink_mappings.json", help="Input biolink mapping file")
    parser.add_argument("-e", "--embeddings", default="llama_all_biolink_mapped_vectors.json", help="Output biolink embedding file")
    parser.add_argument("--lowercase", action="store_true", default=False, help="Use lowercase mappings")
    args = parser.parse_args()
    mappings_file = args.mappings
    embeddings_file = args.embeddings
    use_lowercase = args.lowercase
    asyncio.run(embed_biolink_predicates(mappings_file, embeddings_file, use_lowercase))

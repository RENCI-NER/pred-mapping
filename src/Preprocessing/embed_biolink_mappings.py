import argparse
import json
import asyncio
import time
from tqdm import tqdm
from src.utils import chunked, safe_limited_embedding
from src.llm_client import HEALpacaAsyncClient
from clean_mappings import cull_mapped_predicates

class EmbeddingClient(HEALpacaAsyncClient):
    def __init__(self, **kwargs):
        super().__init__()


async def embed_biolink_predicates(infile, infile_hierarchy_file, outfile, use_lowercase=False):
    start_time = time.time()
    client = EmbeddingClient()

    with open(infile, "r") as file:
        data = json.load(file)

    results = []
    failed_count = 0

    for predicate, text_list in tqdm(data.items(), position=0, desc="Biolink predicates Embedding"):
        predicate_uri = predicate.strip().replace(" ", "_")

        if isinstance(text_list, list) and text_list:
            if use_lowercase:
                text_list = [text.lower() for text in text_list]

            vectors = []
            chunked_texts = chunked(text_list, 25)
            for batch in chunked_texts:
                vector = await safe_limited_embedding(client, batch, retries=3)
                vectors.extend(vector)

            for text, vector in zip(text_list, vectors):
                if vector is not None:
                    results.append({
                        "predicate": predicate_uri,
                        "text": text,
                        "embedding": vector
                    })
                else:
                    # Let's retry once more abeg!
                    try:
                        vector = await client.get_embedding(text)
                    except Exception as e:
                        print(f"Retry failed for text: {text}, Error: {e}")
                        vector = []

                    results.append({
                        "predicate": predicate_uri,
                        "text": text,
                        "embedding": vector if isinstance(vector, (list, tuple)) else []
                    })
                    if not isinstance(vector, (list, tuple)) or not vector:
                        failed_count += 1

        elif isinstance(text_list, str):
            if use_lowercase:
                text_list = text_list.lower()

            try:
                vector = await client.get_embedding(text_list)
                results.append({
                    "predicate": predicate_uri,
                    "text": text_list,
                    "embedding": vector
                })
            except Exception as e:
                failed_count += 1
                print(f"Failed to get embedding for single text: {text_list}, Error: {e}")
                results.append({
                    "predicate": predicate_uri,
                    "text": text_list,
                    "embedding": None
                })

    with open(outfile, "w") as file:
        json.dump(results, file, indent=2)

    end_time = time.time()
    print(f"Execution Time: {end_time - start_time:.2f} seconds")
    print(f"Total results: {len(results)}")
    print(f"Failed embeddings: {failed_count}")
    if failed_count > 0:
        print(f"Success rate: {((len(results) - failed_count) / len(results)) * 100:.1f}%")
    else:
        print("Success rate: 100.0%")

    # with open(infile_hierarchy_file, "r") as file:
    #     infile_hierarchy = json.load(file)
    #
    # culledresults = cull_mapped_predicates(results, infile_hierarchy)
    # with open(f"culled_{outfile}", "w") as file:
    #     json.dump(culledresults, file, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mappings", default="all_chemprot_mappings.json", help="Input biolink mapping file")
    parser.add_argument("-q", "--qualified_mappings", default="qualified_predicate_mappings.json", help="Input biolink mapping file")
    parser.add_argument("-e", "--embeddings", default="all_chemprot_mapped_vectors.json", help="Output biolink embedding file")
    parser.add_argument("--lowercase", action="store_true", default=False, help="Use lowercase mappings")
    args = parser.parse_args()
    mappings_file = args.mappings
    qualified_mappings_file = args.qualified_mappings
    embeddings_file = args.embeddings
    use_lowercase = args.lowercase
    asyncio.run(embed_biolink_predicates(mappings_file, qualified_mappings_file, embeddings_file, use_lowercase))

import json
from vectordb import InMemoryExactNNVectorDB
from docarray import BaseDoc, DocList
from docarray.typing import NdArray


class PredicateText(BaseDoc):
    predicate: str = ''
    text: str = ''
    embedding: NdArray[768]


class PredicateDatabase:
    def __init__(self, client):
        self.db = None
        self.client = client

    def load_db_from_json(self, embeddings_file):
        print("Loading json")
        with open(embeddings_file, "r") as f:
            embeddings = json.load(f)

        self.populate_db(embeddings)

    def populate_db(self, embeddings):
        doc_list = []
        # print(f"Number of entries: {len(embeddings)}")
        for entry in embeddings:
            if len(entry["text"]) != 0:
                doc_list.append(
                    PredicateText(
                        predicate=entry["predicate"],
                        text=entry["text"],
                        embedding=entry["embedding"]
                    )
                )

        print("Load vectordb")
        self.db = InMemoryExactNNVectorDB[PredicateText](workspace='./workspace')
        self.db.index(inputs=DocList[PredicateText](doc_list))
        print("Ready")

    def search(self, text, embedding=None, num_results=10):
        if embedding is None:
            embedding = self.client.get_embedding(text)

        if len(embedding) == 0:
            return None

        query = PredicateText(text=text, embedding=embedding)
        results = self.db.search(inputs=DocList[PredicateText]([query]), limit=num_results)

        texts = [match.text for match in results[0].matches]
        predicates = [match.predicate for match in results[0].matches]
        scores = [score for score in results[0].scores]

        results_dict = {
            i: {"text": t, "mapped_predicate": p, "score": round(s, 4)} for i, (t, p, s) in enumerate(zip(texts, predicates, scores))
        }
        return results_dict

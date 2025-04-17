import json
import numpy as np
import torch
from scipy.spatial.distance import cdist
from vectordb import InMemoryExactNNVectorDB
from docarray import BaseDoc, DocList
from docarray.typing import NdArray


class PredicateText(BaseDoc):
    predicate: str = ''
    text: str = ''
    embedding: NdArray[768]


class PredicateDatabase:
    def __init__(self, client, is_vdb = False):
        self.all_pred_emb = None
        self.all_pred_texts = None
        self.all_pred = None
        self.db = None
        self.client = client
        self.is_vdb = is_vdb

    def load_db_from_json(self, embeddings_file):
        print("Loading json")
        with open(embeddings_file, "r") as f:
            embeddings = json.load(f)
        self.populate_db(embeddings)

    def populate_db(self, embeddings):
        if self.is_vdb:
            doc_list = []
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
        self.all_pred_texts = [e.get("text", "") for e in embeddings]
        self.all_pred = [e.get("predicate", "") for e in embeddings]
        self.all_pred_emb = [e.get("embedding", []) for e in embeddings]
        self.all_pred_emb = transform_embedding(self.all_pred_emb)
        print("Ready")

    def search(self, text, embedding=None, num_results=10):
        if embedding is None:
            embedding = self.client.get_embedding(text)
        if not embedding:
            return None

        if self.is_vdb:
            query = PredicateText(text=text, embedding=embedding)
            results = self.db.search(inputs=DocList[PredicateText]([query]), limit=num_results)

            texts = [match.text for match in results[0].matches]
            predicates = [match.predicate for match in results[0].matches]
            scores = [score for score in results[0].scores]

            results_dict = {
                i: {"text": t, "mapped_predicate": p, "score": round(s, 4)} for i, (t, p, s) in enumerate(zip(texts, predicates, scores))
            }
            return results_dict
        embedding = transform_embedding(embedding)

        dist = 1 - cdist([embedding.cpu().detach().numpy()], self.all_pred_emb, metric="cosine")
        count_dist = np.argpartition(dist, num_results, axis=None)
        result_dist = np.sort(dist[0, count_dist[:num_results]], axis=None)
        indices = [list(np.asarray(np.where(dist.flatten() == d)).flatten())
                   for d in result_dist]
        indices = sum(indices, [])
        return {
            idx:
            {
                "text": self.all_pred_texts[idx],
                "mapped_predicate": self.all_pred[idx],
                "score": round(dist[0, idx], 4)
            }
            for idx in indices[:num_results]
        }


def temp_transform_embedding(embedding):
    return [transform_embedding(entry) for entry in embedding]


def transform_embedding(embedding):
    return torch.tensor(embedding, dtype=torch.float32)
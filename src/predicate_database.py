import numpy as np
import logging
from src.utils import load_from_json
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
logger = logging.getLogger(__name__)


class PredicateDatabase:
    def __init__(self, client, embedding_dim=None, is_vdb=False, is_knn=False):
        self.embedding_dim = embedding_dim
        self.all_pred_emb = None
        self.all_pred_texts = None
        self.all_pred = None
        self.db = None
        self.client = client
        self.is_vdb = is_vdb
        self.is_knn = is_knn
        # self.nn_model = None

    def load_db_from_json(self, embedding_mappings_file):
        embedding_mappings = load_from_json(embedding_mappings_file)
        self.populate_db(embedding_mappings)

    def populate_db(self, embedding_mappings):
        logger.info(f"Initialized DB with {len(embedding_mappings)} predicate embeddings.... ")
        self.all_pred_texts = [e.get("text", "") for e in embedding_mappings]
        self.all_pred = [e.get("predicate", "") for e in embedding_mappings]
        self.all_pred_emb = [e.get("embedding", []) for e in embedding_mappings]
        self.all_pred_emb = transform_embedding(self.all_pred_emb)
        if self.is_knn:
            self.nn_model = NearestNeighbors(n_neighbors=10, metric="cosine")
            self.nn_model.fit(self.all_pred_emb)

    async def search(self, text: str = None, embedding=None, num_results: int = 10):
        if embedding is None or (hasattr(embedding, '__len__') and len(embedding) == 0):
            embedding = await self.client.get_embedding(text)
            if embedding is None:
                return None
        results = await self.batch_search([embedding], num_results)
        return results[0] if results else []

    async def batch_search(self, embeddings=None, num_results: int = 10):
        """Search using pre-computed embeddings."""
        if not embeddings:
            raise RuntimeError("To do a vector search input embeddings cannot be empty")

        embeddings = transform_embedding(embeddings)

        if self.is_knn:
            if self.nn_model is None:
                raise ValueError("NearestNeighbors model not initialized. Call populate_db() first.")
            distances, indices = self.nn_model.kneighbors(embeddings, n_neighbors=num_results)
            similarities = 1 - distances
            return [
                [
                    {
                        "text": self.all_pred_texts[i],
                        "mapped_predicate": self.all_pred[i],
                        "score": round(float(similarities[q_idx][j]), 3)
                    }
                    for j, i in enumerate(neighbor_idxs)
                ]
                for q_idx, neighbor_idxs in enumerate(indices)
            ]
        distance = cdist(embeddings, self.all_pred_emb, "cosine")
        similarities = 1 - distance
        top_indices = np.argsort(-similarities, axis=1)[:, :num_results]
        return [
            [
                {
                    "text": self.all_pred_texts[i],
                    "mapped_predicate": self.all_pred[i],
                    "score": round(float(similarities[q_idx][i]), 3)
                }
                for i in top_k_indices
            ]
            for q_idx, top_k_indices in enumerate(top_indices)
        ]


def transform_embedding(embedding):
    try:
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"Embedding transformation failed: {e}")

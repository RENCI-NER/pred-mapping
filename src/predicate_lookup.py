# src/predicate_lookup.py
import json
import logging
import numpy as np
from tqdm import tqdm
from typing import Union
from pydantic import BaseModel
from src.llm_client import HEALpacaAsyncClient
from bmt import Toolkit
from src.utils import chunked, safe_limited_chat_completion, safe_limited_embedding
from src.predicate_database import PredicateDatabase
from src.ontology_config import get_current_config

logger = logging.getLogger(__name__)

t = Toolkit()

# Global variables for lazy loading (now ontology-aware)
sapbert_data_cache = {}
sapbert_available_cache = {}


class PredicateMapping(BaseModel):
    mapped_predicate: str
    negated: str = "False"


def load_sapbert_data():
    """Load SapBERT data for current ontology"""
    config = get_current_config()
    ontology_name = config.name

    if ontology_name in sapbert_data_cache:
        return sapbert_available_cache[ontology_name]

    paths = config.sapbert_paths
    if not paths or not all(paths):
        logger.warning(f"SapBERT not configured for ontology: {ontology_name}")
        sapbert_available_cache[ontology_name] = False
        sapbert_data_cache[ontology_name] = {}
        return False

    bio_link_path, dict_path, embedding_path, model_folder = paths

    try:
        # Import based on ontology
        if ontology_name == "chemprot":
            from src.Chemprot_SapBert.utils import sapbert_predict as sp, sapbert_score_batch as ssb, get_labels
        elif ontology_name == "biolink":
            from src.Biolink_SapBert.utils import sapbert_predict as sp, sapbert_score_batch as ssb, get_labels
        else:
            raise ImportError(f"No SapBERT module for ontology: {ontology_name}")

        all_rels, all_rels_id = get_labels(str(bio_link_path), str(dict_path))
        all_rels_emb = np.load(str(embedding_path))

        sapbert_data_cache[ontology_name] = {
            'all_rels': all_rels,
            'all_rels_id': all_rels_id,
            'all_rels_emb': all_rels_emb,
            'sapbert_predict': sp,
            'sapbert_score_batch': ssb,
            'model_folder': str(model_folder)
        }

        sapbert_available_cache[ontology_name] = True
        logger.info(f"SapBERT data loaded for ontology: {ontology_name}")

    except (ImportError, FileNotFoundError) as e:
        logger.warning(f"SapBERT not available for {ontology_name}: {e}")
        sapbert_available_cache[ontology_name] = False
        sapbert_data_cache[ontology_name] = {}

    return sapbert_available_cache[ontology_name]


def get_sapbert_data():
    """Get SapBERT data for current ontology"""
    config = get_current_config()
    return sapbert_data_cache.get(config.name, {})


def get_prompt(subject, object, relationship, abstract, predicate_choices, **kwargs):
    relationship_system_prompt = f"""
        Given this input:
            subject = {subject}
            object = {object}
            relationship = {relationship}
            abstract = {abstract}
            predicate_choices = {predicate_choices}

        For each key in predicate_choices, the corresponding value is the description of the key.

        Your Task:
            1. Select the most appropriate key from predicate_choices to replace the given relationship.
            2. Ensure the replacement preserves both **meaning** and **directionality** of the subject-object pair.
            3. Understand that relationships may be **negated**:
                - If a predicate in `predicate_choices` directly matches the **negated meaning**, use that.
                - If a predicate matches the base meaning but you must negate it to capture the intended meaning, select that predicate and set `"negated": "True"` in the response e.g. "does not cause" where causes is in the choices implies that mapped_predicate is causes and negated is True.
                - Otherwise, use `"negated": "False"`.

        Output:
            A JSON object with these exact keys and format:
            {{"mapped_predicate": "Top one predicate choice" if a good match exists, otherwise "none", "negated": "True" or "False"}}

        Do not include any other output or explanation. Only output the JSON object.
    """
    return relationship_system_prompt


def extract_mapped_predicate(response_text):
    """Extract JSON using simple Pydantic parsing"""

    if not response_text or isinstance(response_text, Exception):
        logger.warning(f"No response or exception: {response_text}")
        return None

    try:
        # Find JSON in the response
        start = response_text.find('{')
        end = response_text.rfind('}') + 1

        if start == -1 or end == 0:
            logger.warning("No JSON found in response")
            return None

        json_str = response_text[start:end]
        data = json.loads(json_str)

        parsed = PredicateMapping(**data)

        return {
            "mapped_predicate": parsed.mapped_predicate,
            "negated": parsed.negated
        }

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error: {e}")
        return None


class PredicateClient(HEALpacaAsyncClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.qualified_predicates = None

    async def rerank_relationship_choices(self, relationships_json: list[dict], qualified_predicates: dict, is_nn: bool = False, chunk_size: int = 10) -> list[dict]:
        self.qualified_predicates = qualified_predicates
        prompts = [get_prompt(**r) for r in relationships_json]
        llm_responses = []
        chunked_relationship = chunked(prompts, chunk_size)

        for batch_prompts in tqdm(chunked_relationship, desc="LLM (Predicate Candidate) Reranking",
                                  total=(len(prompts) + chunk_size - 1) // chunk_size):
            responses = await safe_limited_chat_completion(self, batch_prompts)
            llm_responses.extend(responses)

        response_relationship_pairs = list(zip(relationships_json, llm_responses))
        results = []

        for i in tqdm(range(0, len(response_relationship_pairs), chunk_size), desc="LLM Reranking (Postprocessing)"):
            batch = response_relationship_pairs[i:i + chunk_size]
            batch_results = [
                self._format_relationship_result(r_json, response, is_nn)
                for r_json, response in batch
            ]
            for result in batch_results:
                if isinstance(result, dict):
                    results.append(result)
                else:
                    logger.error(f"Failed task in batch {i // chunk_size}: {result}")

        return results

    def _format_relationship_result(self, relationship_json, llm_response, is_nn):
        predicate_choices = relationship_json.get("predicate_choices", {})
        choices = list(predicate_choices.keys())

        if not choices:
            logger.warning(f"No predicate candidates found for relationship: {relationship_json.get('relationship')}")
            relationship_json["top_choice"] = {
                "predicate": " ",
                "object_aspect_qualifier": " ",
                "object_direction_qualifier": " ",
                "negated": "False",
                "selector": " "
            }
            return relationship_json

        parsed_response = extract_mapped_predicate(llm_response)

        if parsed_response is None or parsed_response.get("mapped_predicate") == "none":
            logger.warning(
                f"No valid mapping found for relationship: {relationship_json.get('relationship')}. Using first choice: {choices[0]}")
            predicate = choices[0].strip()
            negated = "False"
            selector = "nearest_neighbors" if is_nn else "scipy"
        else:
            predicate = parsed_response.get("mapped_predicate")
            negated = parsed_response.get("negated", "False")
            selector = self.chat_model

        predicate, oaq, odq = self.is_qualified(predicate)
        relationship_json["top_choice"] = {
            "predicate": predicate,
            "object_aspect_qualifier": oaq,
            "object_direction_qualifier": odq,
            "negated": negated,
            "selector": selector
        }
        relationship_json.pop("predicate_choices", None)
        return relationship_json

    def is_qualified(self, predicate):
        p = self.qualified_predicates.get(predicate, None)
        if p is None:
            return predicate, "", ""
        return p.get("predicate", ""), p.get("object_aspect_qualifier", ""), p.get("object_direction_qualifier", "")


def parse_new_llm_response(llm_response: Union[str, list[dict]]) -> list[dict]:
    if isinstance(llm_response, str):
        with open(llm_response, "r") as f:
            if llm_response.endswith(".jsonl"):
                parsed = [json.loads(line) for line in f]
            elif llm_response.endswith(".json"):
                parsed = json.load(f)
            else:
                raise ValueError("Unsupported file type: must be .json or .jsonl")
    elif isinstance(llm_response, list):
        parsed = llm_response
    else:
        raise TypeError("Input must be a path (str) or a list of dicts")

    return parsed


def relationship_queries_to_batch(query_results: list[dict], descriptions, is_vdb, is_nn) -> list[dict]:
    method = "vectorDb" if is_vdb else ("nearest_neighbors" if is_nn else "similarities")
    return [
        {
            **edge,
            "Top_n_retrieval_method": method,
            "predicate_choices": {k: descriptions.get(k, k) for k in edge.get("Top_n_candidates", {})},
            "Top_n_candidates": {
                i: {"mapped_predicate": k, "score": v}
                for i, (k, v) in enumerate(edge.get("Top_n_candidates", {}).items())
            },
        }
        for edge in query_results
    ]


async def lookup_unique_predicates(
        parsed_data: list[dict],
        db: PredicateDatabase,
        output_file: str = None,
        num_results: int = 10,
        batch_size: int = 25,
        use_sapbert: bool = True
) -> list[dict]:
    input_relationships = list(set(e["relationship"] for e in parsed_data))

    sapbert_results_dict = {}
    if use_sapbert:
        sapbert_available = load_sapbert_data()

        if sapbert_available:
            try:
                sapbert_data = get_sapbert_data()
                sapbert_relationship_embs = sapbert_data['sapbert_predict'](
                    sapbert_data['model_folder'], input_relationships, use_gpu=False
                )
                sapbert_topk_results = sapbert_data['sapbert_score_batch'](
                    sapbert_relationship_embs,
                    sapbert_data['all_rels_emb'],
                    sapbert_data['all_rels_id'],
                    sapbert_data['all_rels'],
                    num_results
                )
                sapbert_results_dict = dict(zip(input_relationships, sapbert_topk_results))
            except Exception as e:
                logger.error(f"SapBERT prediction failed: {e}")
                sapbert_results_dict = {}

    chunked_relationship = chunked(input_relationships, batch_size)
    relationship_embeddings = []
    for batch in tqdm(chunked_relationship, desc="Embedding Relationship Batches"):
        result = await safe_limited_embedding(db.client, batch)
        relationship_embeddings.extend(result)

    search_results = await db.batch_search(
        embeddings=relationship_embeddings,
        num_results=num_results
    )
    search_results_dict = dict(zip(input_relationships, search_results))

    updated_data = format_result(parsed_data, search_results_dict, sapbert_results_dict)

    if output_file is not None:
        with open(output_file, "w") as out_file:
            out_file.writelines(json.dumps(edge) + "\n" for edge in updated_data)

    return updated_data


def format_result(edges: list[dict], search_results: dict, sapbert_results: dict = None) -> list[dict]:
    if sapbert_results is None:
        sapbert_results = {}

    for edge in edges:
        rel = edge.get("relationship")
        try:
            unique_predicates = {}

            rel_search_results = search_results.get(rel, [])
            rel_sapbert_results = sapbert_results.get(rel, [])

            combined_results = rel_search_results + rel_sapbert_results

            for result in combined_results:
                pred = result["mapped_predicate"].replace("biolink:", "").replace("_NEG", "").replace("_", " ")
                score = round(result["score"], 5)
                if pred not in unique_predicates or score > unique_predicates[pred]:
                    unique_predicates[pred] = score

            for predicate in list(unique_predicates):
                try:
                    inverse = t.get_element(predicate).inverse
                    if inverse and inverse not in unique_predicates:
                        unique_predicates[inverse] = unique_predicates[predicate]
                except AttributeError:
                    continue

            edge["Top_n_candidates"] = dict(
                sorted(unique_predicates.items(), key=lambda item: item[1], reverse=True)
            )

        except Exception as e:
            logger.error(f"Search failed for edge '{rel}': {e}")
            edge["Top_n_candidates"] = {}
    return edges

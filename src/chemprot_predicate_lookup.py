import re
import json
import ast
import os
import logging
import numpy as np
from tqdm import tqdm
from typing import Union, Optional
from difflib import get_close_matches
from src.llm_client import HEALpacaAsyncClient
from bmt import Toolkit
from src.utils import chunked, safe_limited_chat_completion, safe_limited_embedding
from src.predicate_database import PredicateDatabase

logger = logging.getLogger(__name__)

t = Toolkit()

# SapBERT configuration/ paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BIO_LINK_PATH = os.path.join(BASE_DIR, "Chemprot_SapBert/data/mappings_dataset.txt")
DICT_PATH = os.path.join(BASE_DIR, "Chemprot_SapBert/data/mappings_dictionary.txt")
EMBEDDING_PATH = os.path.join(BASE_DIR, "Chemprot_SapBert/data/embedding_mappings.npy")
MODEL_FOLDER = os.path.join(BASE_DIR, "Chemprot_SapBert/model")

# Global variables for lazy loading
sapbert_data_loaded = False
sapbert_available = None
ontology_predicate_texts = None
ontology_predicate_labels = None
ontology_predicate_embeddings = None
sapbert_predict = None
sapbert_score_batch = None


def load_sapbert_data():
    """Lazy load SapBERT data only when needed"""
    global sapbert_data_loaded, sapbert_available, ontology_predicate_texts, ontology_predicate_labels, ontology_predicate_embeddings
    global sapbert_predict, sapbert_score_batch

    if sapbert_data_loaded:
        return sapbert_available

    try:
        logger.info("Loading SapBERT data...")
        from src.Chemprot_SapBert.utils import sapbert_predict, sapbert_score_batch, get_labels

        ontology_predicate_texts, ontology_predicate_labels = get_labels(BIO_LINK_PATH, DICT_PATH)
        ontology_predicate_embeddings = np.load(EMBEDDING_PATH)
        sapbert_predict = sapbert_predict
        sapbert_score_batch = sapbert_score_batch

        sapbert_available = True
        logger.info("SapBERT data loaded successfully")

    except (ImportError, FileNotFoundError) as e:
        logger.warning(f"SapBERT not available: {e}")
        sapbert_available = False

    sapbert_data_loaded = True
    return sapbert_available


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


class PredicateClient(HEALpacaAsyncClient):
    def __init__( self, **kwargs ):
        super().__init__(**kwargs)
        self.qualified_predicates = None

    async def rerank_relationship_choices( self, relationships_json: list[dict], qualified_predicates: dict,
                                           is_vdb: bool = False, is_nn: bool = False, chunk_size: int = 10 ) -> list[
        dict]:
        """ Send options for a single relationship to LLM in concurrent chunked batches """
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
                self._format_relationship_result(r_json, response, is_vdb, is_nn)
                for r_json, response in batch
            ]
            for result in batch_results:
                if isinstance(result, dict):
                    results.append(result)
                else:
                    logger.error(f"Failed task in batch {i // chunk_size}: {result}")

        return results

    def _format_relationship_result(self, relationship_json, llm_response, is_vdb, is_nn):
        predicate_choices = relationship_json.get("predicate_choices", {})
        choices = list(predicate_choices.keys())
        if not choices:
            logger.warning(
                f"No predicate candidate(s) found for relationship: {relationship_json.get('relationship')}. Cannot proceed.")
            relationship_json["top_choice"] = {
                "predicate": " ",
                "object_aspect_qualifier": " ",
                "object_direction_qualifier": " ",
                "negated": "False",
                "selector": " "
            }
            return relationship_json

        top_choice = extract_mapped_predicate(llm_response, predicate_choices)

        if top_choice is None or top_choice.get("mapped_predicate") is None:
            # Parsing failure or malformed LLM response
            logger.warning(
                f"Malformed LLM response, cannot parse JSON: {llm_response} for relationship: {relationship_json.get('relationship')}. Falling back to: {choices[0]}")
            predicate = f'{choices[0].strip()}'
            negated = "False"
            selector = "vectorDB" if is_vdb else "nearest_neighbors" if is_nn else "scipy"
        else:
            # Parsing when LLM explicitly returned "none" or one of the choices
            predicate = top_choice.get("mapped_predicate")
            negated = top_choice.get("negated", "False")
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


async def lookup_unique_predicates(parsed_data: list[dict], db: PredicateDatabase, output_file: str = None, num_results: int = 10, batch_size: int = 25, use_sapbert: bool = True) -> list[dict]:
    """
    Look up predicate candidates for all relationships using vector database and optionally SapBERT.

    Args:
        parsed_data: List of relationship dictionaries
        db: PredicateDatabase instance
        output_file: Optional output file path
        num_results: Number of top results to return
        batch_size: Batch size for processing
        use_sapbert: Whether to use SapBERT predictions (loads data if needed)

    Returns:
        List of updated relationship dictionaries with predicate candidates
    """
    logger.info("Looking up predicate candidates for all relationships...")
    input_relationships = list(set(e["relationship"] for e in parsed_data))

    # Get SapBERT results if requested and available
    sapbert_results_dict = {}
    if use_sapbert:
        sapbert_available = load_sapbert_data()  # Lazy load

        if sapbert_available:
            logger.info("Computing SapBERT predictions...")
            try:
                sapbert_relationship_embs = sapbert_predict(MODEL_FOLDER, input_relationships, use_gpu=False)
                sapbert_topk_results = sapbert_score_batch(sapbert_relationship_embs, ontology_predicate_embeddings, ontology_predicate_labels, ontology_predicate_texts, num_results)
                sapbert_results_dict = dict(zip(input_relationships, sapbert_topk_results))
                logger.info(f"SapBERT predictions computed for {len(sapbert_results_dict)} relationships")
            except Exception as e:
                logger.error(f"SapBERT prediction failed: {e}")
                sapbert_results_dict = {}
        else:
            logger.warning("SapBERT requested but not available, skipping...")

    # Get vector database embeddings and search results
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

    # Format results combining both sources
    updated_data = format_result(parsed_data, search_results_dict, sapbert_results_dict)

    if output_file is not None:
        with open(output_file, "w") as out_file:
            out_file.writelines(json.dumps(edge) + "\n" for edge in updated_data)

    return updated_data


def format_result( edges: list[dict], search_results: dict, sapbert_results: dict = None ) -> list[dict]:
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


def extract_mapped_predicate( response_text, choices ):
    def find_key_from_value( val, options ):
        try:
            for key, value in options.items():
                if val.lower() == value.lower() or val.lower() in value.lower():
                    return f'{key.strip()}'
        except Exception as e:
            logger.warning(f"Exception: {e} for {val}")
            return None

    def _validate_negated( negated_value ):
        """Validate and normalize negated field to boolean string"""
        if isinstance(negated_value, bool):
            return str(negated_value)
        elif isinstance(negated_value, str):
            negated_lower = negated_value.strip().lower()
            if negated_lower in ['true', 'yes', '1']:
                return "True"
            elif negated_lower in ['false', 'no', '0']:
                return "False"
            else:
                logger.warning(f"Invalid negated value: '{negated_value}', defaulting to False")
                return "False"
        else:
            logger.warning(f"Unexpected negated type: {type(negated_value)}, defaulting to False")
            return "False"

    def _format_if_valid( mapped_pred, normalized_options, allow_raw=False ):
        if mapped_pred in normalized_options:
            return f'{mapped_pred.strip()}'

        reverse = find_key_from_value(mapped_pred, normalized_options)
        if reverse:
            return reverse

        match_pred = get_close_matches(mapped_pred, normalized_options.keys(), n=1)
        if match_pred:
            return f'{match_pred[0].strip()}'

        if allow_raw:
            return mapped_pred

        return None

    default = {"mapped_predicate": None, "negated": "False"}

    if not response_text or isinstance(response_text, Exception):
        logger.warning(f"[extract_mapped_predicate] No response or exception: {response_text}")
        return default

    cleaned_text = re.sub(r'```(?:json)?\n?', '', response_text.strip()).strip("` \n")
    json_patterns = [
        # Complete JSON with both fields?
        r'\{\s*["\']mapped_predicate["\']\s*:\s*["\'][^"\']*["\']\s*,\s*["\']negated["\']\s*:\s*["\'][^"\']*["\']\s*\}',
        # JSON with mapped_predicate only?
        r'\{\s*["\']mapped_predicate["\']\s*:\s*["\'][^"\']*["\']\s*\}',
        # Fallback: any JSON-like structure with mapped_predicate?
        r'\{[^{}]*["\']mapped_predicate["\']\s*:\s*[^{}]*?\}',
    ]

    for pattern in json_patterns:
        match = re.search(pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
        if match:
            json_candidate = match.group().strip()
            try:
                # Clean up common quote issues
                json_candidate = json_candidate.replace('"', '"').replace('"', '"')
                json_candidate = json_candidate.replace(''', "'").replace(''', "'")
                parsed = json.loads(json_candidate)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(json_candidate)
                except Exception as e:
                    logger.warning(f"JSON parsing failed for: {json_candidate[:100]}... Error: {e}")
                    continue

            if not isinstance(parsed, dict) or 'mapped_predicate' not in parsed:
                logger.warning(f"Invalid parsed structure: {parsed}")
                continue

            mapped = parsed.get("mapped_predicate", "").strip().lower()
            negated_raw = parsed.get("negated", "False")
            negated = _validate_negated(negated_raw)

            if mapped == "none":
                return {"mapped_predicate": "none", "negated": negated}

            normalized_choices = {k.strip().lower(): v.strip().lower() for k, v in choices.items()}
            formatted = _format_if_valid(mapped, normalized_choices)
            if not formatted:
                logger.warning(
                    f"[extract_mapped_predicate] Failed to map: '{mapped}' from response:\n{response_text}\n")
            return {"mapped_predicate": formatted or None, "negated": negated}

    logger.warning(f"[extract_mapped_predicate] No valid JSON structure found in:\n{response_text}\n")
    return default
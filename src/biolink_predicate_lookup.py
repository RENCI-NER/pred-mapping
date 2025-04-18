import re
import json
import ast
import time
from typing import Union
from src.llm_client import HEALpacaClient
import logging
logger = logging.getLogger(__name__)
logging.getLogger("linkml_runtime").setLevel(logging.WARNING)
logging.getLogger("docarray").setLevel(logging.ERROR)
from bmt import Toolkit
from src.predicate_database import PredicateDatabase


def get_prompt(in_json):
    relationship_system_prompt = f"""
        given this input:
            subject = {in_json["subject"]} ,
            object = {in_json["object"]} ,
            relationship = {in_json["relationship"]} ,
            abstract = {in_json["abstract"]} ,
            predicate_choices = {in_json["predicate_choices"]}
        
        For each key in predicate_choices, the corresponding value is the description of the key

        Your Task:
            Select the most appropriate key from predicate_choices to replace the relationship based on the abstract while maintaining meaning and directionality of the subject and object.

        Output: Respond strictly and ending with a JSON object in this format:
        {{"mapped_predicate": "Top one predicate choice"}}
        
        If any of the predicate_choices key is not a good replacement for the relationship, respond with:
        {{"mapped_predicate": "none"}}
    """
    return relationship_system_prompt


class PredicateClient(HEALpacaClient):
    def __init__( self, **kwargs ):
        super().__init__(**kwargs)

    def check_relationship(self, relationships_json: list[dict], is_vdb = False, is_nn= False) -> list:
        """ Send options for a single relationship to OpenAI LLM """
        """ Send options for a single relationship to LLM """
        model = self.chat_model
        relationships = []
        for relationship_json in relationships_json:
            prompt = get_prompt(relationship_json)
            ai_response = self.get_chat_completion(prompt)
            choices = list(relationship_json.get("predicate_choices").keys())
            top_choice = extract_mapped_predicate(ai_response, choices)
            relationship_json["top_choice"] = {
                "predicate": top_choice or f'biolink:{choices[0].replace(" ", "_")}',
                "qualifier": "",
                "selector": (
                    model if top_choice
                    else "vectorDB" if is_vdb
                    else "similarities" if is_nn
                    else "nearest_neighbors"
                )
            }
            relationship_json.pop("predicate_choices", None)
            relationships.append(relationship_json)
            time.sleep(1.0)
        return relationships


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
    batch_data = []
    batch_keys = [
        "Top-n candidates",
        "subject",
        "object",
        "relationship",
        "abstract",
    ]
    method = "vectorDb" if is_vdb else ("nearest_neighbors" if is_nn else "similarities")
    for edge in query_results:
        batch_edge = {key: val for key, val in edge.items() if key in batch_keys}
        batch_edge["Top-n retrieval_method"] = method
        predicate_choices = batch_edge["Top-n candidates"].keys()
        predicate_choices = {k: descriptions.get(k, k) for k in predicate_choices}
        batch_edge["predicate_choices"] = predicate_choices
        batch_edge["Top-n candidates"] = {i : {"mapped_predicate": k,  "score": v} for i, (k, v) in enumerate(batch_edge["Top-n candidates"].items())}
        batch_data.append(batch_edge)
    return batch_data


def lookup_unique_predicates(parsed_data: list[dict], db: PredicateDatabase, output_file: str = None,
                              num_results: int = 10) -> list[dict]:
    print("Looking up mapped predicates for all relationships")
    need_embeddings = sum(["relationship_embedding" in list(edge.keys()) for edge in parsed_data])
    print(f"Embeddings found: {need_embeddings}. Sending {len(parsed_data) - need_embeddings} relationships to model.")

    t = Toolkit()
    for edge in parsed_data:
        try:
            if "relationship_embedding" not in list(edge.keys()):
                edge["relationship_embedding"] = db.client.get_embedding(edge["relationship"])

            search_results = db.search(
                text=edge["relationship"],
                embedding=edge["relationship_embedding"],
                num_results=num_results
            )

            # we'd like to keep the scores
            unique_predicates = {search_results[key]["mapped_predicate"].replace("biolink:", "").replace("_NEG", ""):
                                     search_results[key]["score"] for key in search_results}

            for predicate in unique_predicates.copy():
                try:  # Avoid failing when attempting to add inverse of qualifiedPredicates which 'as it is' doesn't exist in biolink
                    if t.get_element(predicate).inverse is not None:
                        unique_predicates.update({t.get_element(predicate).inverse: unique_predicates[predicate]})
                except AttributeError:
                    pass

            edge["Top-n candidates"] = {
                predicate.replace("_", " "): score
                for predicate, score in sorted(unique_predicates.items(), key=lambda item: item[1], reverse=True)
            }
        except KeyError as e:
            print(e)
            print(json.dumps(edge, indent=2))

    if output_file is not None:
        with open(output_file, "w") as out_file:
            out_file.writelines(json.dumps(edge) + "\n" for edge in parsed_data)
    return parsed_data


def extract_mapped_predicate(response_text, choices):
    if response_text is None:
        return None

    # Preprocess: normalize and strip known wrapping artifacts
    cleaned_text = re.sub(r'```(?:json)?\n?', '', response_text.strip()).strip("` \n")

    # Create case-insensitive lookup for choice keys
    normalized_choices = {k.lower(): k for k in choices}

    # === 1. Try to extract valid JSON/dict-style format ===
    json_patterns = [
        r'\{[^{}]*["\']mapped_predicate["\']\s*:\s*[^{}]*?\}',  # JSON/dict format
    ]

    for pattern in json_patterns:
        match = re.search(pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
        if match:
            json_candidate = match.group().strip()

            # Handle mapped_predicate: null
            if 'null' in json_candidate.lower():
                return "none"

            try:
                parsed = json.loads(json_candidate)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(json_candidate)
                except Exception as e:
                    logger.warning(f"Fallback literal_eval failed: {e}")
                    continue

            mapped = parsed.get("mapped_predicate")
            if not mapped:
                return "none"

            return _format_if_valid(mapped, normalized_choices, choices)

    # === 2. Fallback: Loose pattern match e.g., mapped_predicate: "treats" ===
    loose_match = re.search(
        r'["\']?mapped[_ ]predicate["\']?\s*:\s*["\']([^"\'}\n\r]+)["\']?',
        cleaned_text,
        re.IGNORECASE
    )
    if loose_match:
        mapped = loose_match.group(1).strip()
        return _format_if_valid(mapped, normalized_choices, choices)

    # === 3. Final fallback: Try natural-language phrasing ===
    nl_matches = re.findall(
        r'(?:mapped[_ ]predicate[^a-zA-Z0-9]*)?[`\'"]([a-zA-Z0-9_ \-]+)[`\'"]',
        cleaned_text,
        flags=re.IGNORECASE
    )
    for match in reversed(nl_matches):
        mapped = match.strip()
        result = _format_if_valid(mapped, normalized_choices, choices)
        if result:
            return result

    # === 4. Last-resort: match full text to a choice ===
    mapped = cleaned_text.strip()
    return _format_if_valid(mapped, normalized_choices, choices, allow_raw=True)


def _format_if_valid(mapped, normalized_choices, original_choices, allow_raw=False):
    """Helper to validate and format the mapped predicate."""
    if not isinstance(mapped, str):
        return "none"

    mapped_lower = mapped.lower()
    if mapped_lower in normalized_choices:
        canonical_key = normalized_choices[mapped_lower]
        return f'biolink:{canonical_key.replace(" ", "_")}'

    # If it's a value match (reverse-lookup)
    reverse = find_key_from_value(mapped, original_choices)
    if reverse:
        return reverse

    if allow_raw:
        logger.warning(f"Returning fallback raw mapped predicate: '{mapped}'")
        return mapped

    logger.warning(f"Mapped predicate '{mapped}' not found in choices.")
    return "none"


def find_key_from_value(val, choices):
    try:
        val = val.strip().lower()
        for key, value in choices.items():
            if isinstance(value, str) and val == value.strip().lower():
                return f'biolink:{key.replace(" ", "_")}'
    except Exception as e:
        print(f" Exception: {e} for {val}")
    return None

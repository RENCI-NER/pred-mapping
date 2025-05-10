import re
import json
import ast
import asyncio
import requests
import yaml
from collections import defaultdict
from typing import Union
from src.llm_client import HEALpacaAsyncClient
import logging
logger = logging.getLogger(__name__)
logging.getLogger("linkml_runtime").setLevel(logging.WARNING)
logging.getLogger("docarray").setLevel(logging.ERROR)
from bmt import Toolkit
from src.predicate_database import PredicateDatabase

t = Toolkit()

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
            3. Understand that relationships may be **negated** (e.g., "does not cause", "fails to inhibit").
                - If a predicate in `predicate_choices` directly matches the **negated meaning**, use that.
                - If a predicate matches the base meaning but you must negate it to capture the intended meaning, select that predicate and set `"negated": "True"` in the response.
                - Otherwise, use `"negated": "False"`.

        Output:
            A JSON object with these exact keys and format:
            {{"mapped_predicate": "Top one predicate choice" if a good match exists, otherwise "none", "negated": "True" or "False"}}

        Do not include any other output or explanation. Only output the JSON object.
    """
    return relationship_system_prompt


# def get_prompt(subject, object, relationship, abstract, predicate_choices, **kwargs):
#     relationship_system_prompt = f"""
#         given this input:
#             subject = {subject} ,
#             object = {object} ,
#             relationship = {relationship} ,
#             abstract = {abstract} ,
#             predicate_choices = {predicate_choices}
#
#         For each key in predicate_choices, the corresponding value is the description of the key
#
#         Your Task:
#             Select the most appropriate key from predicate_choices to replace the relationship based on the abstract while maintaining meaning and directionality of the subject and object.
#
#         Output: Response strictly ending with a JSON object in this format:
#         {{"mapped_predicate": "Top one predicate choice"}}
#
#         If any of the predicate_choices key is not a good replacement for the relationship, respond with:
#         {{"mapped_predicate": "none"}}
#     """
#     return relationship_system_prompt


class PredicateClient(HEALpacaAsyncClient):
    def __init__( self, **kwargs ):
        super().__init__(**kwargs)
        self.qualified_predicates = None

    async def check_relationship(self, relationships_json: list[dict], qualified_predicates: dict, is_vdb = False, is_nn= False) -> list:
        """ Send options for a single relationship to LLM """
        self.qualified_predicates = qualified_predicates
        tasks = []
        for relationship_json in relationships_json:
            prompt = get_prompt(**relationship_json)
            task = asyncio.create_task(self._process_single_relationship(relationship_json, prompt, is_vdb, is_nn))
            tasks.append(task)
        return await asyncio.gather(*tasks)

    async def _process_single_relationship(self, relationship_json, prompt, is_vdb, is_nn):
        ai_response = await self.get_chat_completion(prompt)
        return self._format_relationship_result(relationship_json, ai_response, is_vdb, is_nn)

    def _format_relationship_result( self, relationship_json, ai_response, is_vdb, is_nn ):
        choices = list(relationship_json.get("predicate_choices").keys())
        top_choice = extract_mapped_predicate(ai_response, relationship_json.get("predicate_choices"))
        logger.info(f"""
        [LLM]: {self.chat_model}
        [Input]: {relationship_json.get('relationship')}
        [LLM Raw Response]: {ai_response}
        [Parsed Predicate]: {top_choice}
        """)
        if not top_choice:
            logger.warning(
                f"No valid mapping for relationship: {relationship_json.get('relationship')}. Falling back to: {choices[0]}")
        negated = top_choice.get("negated", False)
        top_choice = top_choice.get("mapped_predicate", None)
        predicate = top_choice or f'biolink:{choices[0].replace(" ", "_")}'
        predicate, oaq, odq = self.is_qualified(predicate)
        relationship_json["top_choice"] = {
            "predicate": predicate,
            "object_aspect_qualifier": oaq,
            "object_direction_qualifier": odq,
            "negated": negated,
            "selector":  self.chat_model if top_choice else "vectorDB" if is_vdb else "nearest_neighbors" if is_nn else "similarities"
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
    batch_data = []
    batch_keys = [
        "Top_n_candidates",
        "subject",
        "object",
        "relationship",
        "abstract",
    ]
    method = "vectorDb" if is_vdb else ("nearest_neighbors" if is_nn else "similarities")
    for edge in query_results:
        batch_edge = {key: val for key, val in edge.items() if key in batch_keys}
        batch_edge["Top_n_retrieval_method"] = method
        predicate_choices = batch_edge.get("Top_n_candidates", {}).keys()
        predicate_choices = {k: descriptions.get(k, k) for k in predicate_choices}
        batch_edge["predicate_choices"] = predicate_choices
        batch_edge["Top_n_candidates"] = {i : {"mapped_predicate": k,  "score": v} for i, (k, v) in enumerate(batch_edge.get("Top_n_candidates", {}).items())}
        batch_data.append(batch_edge)
    return batch_data


async def lookup_unique_predicates(parsed_data: list[dict], db: PredicateDatabase, output_file: str = None,
                              num_results: int = 10) -> list[dict]:
    print("Looking up mapped predicates for all relationships")

    tasks = [process_single_edge(edge, db, num_results) for edge in parsed_data]
    updated_data = await asyncio.gather(*tasks)

    need_embeddings = sum(["relationship_embedding" in list(edge.keys()) for edge in parsed_data])
    print(f"Embeddings found: {need_embeddings}. Sending {len(parsed_data) - need_embeddings} relationships to model.")

    if output_file is not None:
        with open(output_file, "w") as out_file:
            out_file.writelines(json.dumps(edge) + "\n" for edge in updated_data)

    return updated_data


async def process_single_edge( edge, db, num_results ):
    try:
        if "relationship_embedding" not in edge:
            edge["relationship_embedding"] = await db.client.get_embedding(edge["relationship"])

        search_results = await db.search(
            text=edge["relationship"],
            embedding=edge["relationship_embedding"],
            num_results=num_results
        )

        if search_results:
            unique_predicates = {
                search_results[key]["mapped_predicate"].replace("biolink:", "").replace("_NEG", ""):
                    round(search_results[key]["score"], 5)
                for key in search_results
            }

            for predicate in unique_predicates.copy():
                try:
                    if t.get_element(predicate).inverse is not None:
                        unique_predicates[t.get_element(predicate).inverse] = unique_predicates[predicate]
                except AttributeError:
                    pass

            edge["Top_n_candidates"] = {
                predicate.replace("_", " "): score
                for predicate, score in sorted(unique_predicates.items(), key=lambda item: item[1], reverse=True)
            }

    except KeyError as e:
        print(f"KeyError: {e}\n{json.dumps(edge, indent=2)}")

    return edge


def extract_mapped_predicate(response_text, choices):
    def find_key_from_value(val, choices):
        try:
            val = val.strip().lower()
            for key, value in choices.items():
                if isinstance(value, str) and (val == value.strip().lower() or val in value.strip().lower()):
                    return f'biolink:{key.replace(" ", "_")}'
        except Exception as e:
            print(f"Exception: {e} for {val}")
        return None

    def _format_if_valid(mapped, normalized_choices, original_choices, allow_raw=False):
        if not isinstance(mapped, str):
            return None

        mapped_lower = mapped.lower()
        if mapped_lower in normalized_choices:
            canonical_key = normalized_choices[mapped_lower]
            return f'biolink:{canonical_key.replace(" ", "_")}'

        reverse = find_key_from_value(mapped, original_choices)
        if reverse:
            return reverse

        if allow_raw:
            return mapped

        return None

    if response_text is None:
        return None

    choices_keys = choices.keys()
    cleaned_text = re.sub(r'```(?:json)?\n?', '', response_text.strip()).strip("` \n")
    normalized_choices = {k.lower(): k for k in choices_keys}

    json_patterns = [
        r'\{[^{}]*["\']mapped_predicate["\']\s*:\s*[^{}]*?["\']\s*,\s*["\']negated["\']\s*:\s*[^{}]*?\}',  # includes negated
    ]

    for pattern in json_patterns:
        match = re.search(pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
        if match:
            json_candidate = match.group().strip()
            if 'null' in json_candidate.lower():
                return {"mapped_predicate": None, "negated": "False"}

            try:
                json_candidate = json_candidate.replace("‘", "'").replace("’", "'").replace('“', '"').replace('”', '"')
                parsed = json.loads(json_candidate)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(json_candidate)
                except Exception as e:
                    print(f"Fallback literal_eval failed: {e}")
                    continue

            mapped = parsed.get("mapped_predicate")
            negated = parsed.get("negated", "False")
            if not mapped:
                return {"mapped_predicate": None, "negated": "False"}

            formatted = _format_if_valid(mapped, normalized_choices, choices)
            return {"mapped_predicate": formatted or None, "negated": str(negated).capitalize()}

    return {"mapped_predicate": None, "negated": "False"}

# def extract_mapped_predicate(response_text, choices):
#     def find_key_from_value( val, choices ):
#         try:
#             val = val.strip().lower()
#             for key, value in choices.items():
#                 if isinstance(value, str) and val == value.strip().lower() or val in value.strip().lower():
#                     return f'biolink:{key.replace(" ", "_")}'
#         except Exception as e:
#             print(f" Exception: {e} for {val}")
#         return None
#
#     def _format_if_valid( mapped, normalized_choices, original_choices, allow_raw=False ):
#         """Helper to validate and format the mapped predicate."""
#         if not isinstance(mapped, str):
#             return None  # "none"
#
#         mapped_lower = mapped.lower()
#         if mapped_lower in normalized_choices:
#             canonical_key = normalized_choices[mapped_lower]
#             return f'biolink:{canonical_key.replace(" ", "_")}'
#
#         # If it's a value match (reverse-lookup)
#         reverse = find_key_from_value(mapped, original_choices)
#         if reverse:
#             return reverse
#
#         if allow_raw:
#             logger.warning(f"Returning fallback raw mapped predicate: '{mapped}'")
#             return mapped
#
#         logger.warning(f"Mapped predicate '{mapped}' not found in choices.")
#         return None  # "none"
#
#     if response_text is None:
#         return None
#
#     choices_keys = choices.keys()
#     # Normalize and strip known wrapping artifacts
#     cleaned_text = re.sub(r'```(?:json)?\n?', '', response_text.strip()).strip("` \n")
#
#     # Case-insensitive lookup for choice keys
#     normalized_choices = {k.lower(): k for k in choices_keys}
#
#     # === 1. Extract valid JSON/dict-style format ===
#     json_patterns = [
#         r'\{[^{}]*["\']mapped_predicate["\']\s*:\s*[^{}]*?\}',  # JSON/dict format
#     ]
#
#     for pattern in json_patterns:
#         match = re.search(pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
#         if match:
#             json_candidate = match.group().strip()
#             # {mapped_predicate: null}
#             if 'null' in json_candidate.lower():
#                 return None#"none"
#
#             try:
#                 json_candidate = json_candidate.replace("‘", "'").replace("’", "'").replace('“', '"').replace('”', '"')
#                 parsed = json.loads(json_candidate)
#             except json.JSONDecodeError:
#                 try:
#                     parsed = ast.literal_eval(json_candidate)
#                 except Exception as e:
#                     logger.warning(f"Fallback literal_eval failed: {e}")
#                     continue
#
#             mapped = parsed.get("mapped_predicate")
#             if not mapped:
#                 return None#"none"
#
#             return _format_if_valid(mapped, normalized_choices, choices)
#
#     # === 2. Fallback: Loose pattern match e.g., mapped_predicate: "treats" ===
#     loose_match = re.search(
#         r'["\']?mapped[_ ]predicate["\']?\s*:\s*["\']([^"\'}\n\r]+)["\']?',
#         cleaned_text,
#         re.IGNORECASE
#     )
#     if loose_match:
#         mapped = loose_match.group(1).strip()
#         return _format_if_valid(mapped, normalized_choices, choices)
#
#     # === 3. Final fallback: Try natural-language phrasing ===
#     nl_matches = re.findall(
#         r'(?:mapped[_ ]predicate[^a-zA-Z0-9]*)?[`\'"]([a-zA-Z0-9_ \-]+)[`\'"]',
#         cleaned_text,
#         flags=re.IGNORECASE
#     )
#     for match in reversed(nl_matches):
#         mapped = match.strip()
#         result = _format_if_valid(mapped, normalized_choices, choices)
#         if result:
#             return result
#
#     # === 4. Last-resort: match full text to a choice ===
#     mapped = cleaned_text.strip()
#     return _format_if_valid(mapped, normalized_choices, choices, allow_raw=True)





def retrieve_qualified_mappings(reverse=False, output_file=None):
    """
    Fetches and parses the predicate mapping YAML file from the Biolink Model repository.
    Returns:
        dict: A dictionary where keys are predicates and values are lists of their corresponding mappings.
    """

    yaml_url = "https://raw.githubusercontent.com/biolink/biolink-model/master/predicate_mapping.yaml"

    unwanted_matches = ["releasing_agent", "positive_modulator", "partial_agonist", "channel_blocker",
                        "antisense_inhibitor", "negative_allosteric_modulator", "negative_modulator",
                        "inverse_agonist", "gating_inhibitor"]

    response = requests.get(yaml_url)
    response.raise_for_status()

    predicate_data = yaml.safe_load(response.text)
    mapping_dict = defaultdict(list)

    for mapping in predicate_data.get("predicate mappings", []):
        predicate = mapping.get("qualified predicate", mapping.get("predicate"))
        if not predicate:
            continue
        aspect_qualifier = mapping.get("object aspect qualifier", "")
        direction_qualifier = mapping.get("object direction qualifier", "")
        matches = mapping.get("exact matches", [])
        filtered_matches = [match.split(":")[1] for match in matches if
                            ":" in match and not match.split(":")[1].isdigit() and "_" in match and
                            match.split(":")[1] not in unwanted_matches]
        if reverse:
            for match in filtered_matches:
                mapping_dict.update({f"biolink:{match}": {
                    "predicate": f"biolink:{predicate}",
                    "object_aspect_qualifier": aspect_qualifier.replace(" ", "_"),
                    "object_direction_qualifier": direction_qualifier.replace(" ", "_")}})
        else:
            mapping_dict[predicate].extend([" ".join(match.split("_")) for match in filtered_matches])

    if output_file is not None:
        with open(output_file, "w") as file:
            file.write(json.dumps(mapping_dict, indent=2))

    return dict(mapping_dict)

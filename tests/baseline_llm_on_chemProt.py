import requests
import json
import os
from dotenv import load_dotenv
from tqdm import tqdm
load_dotenv()

LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:11434/api/generate")
CHAT_MODEL = os.getenv("CHAT_MODEL", "alibayram/medgemma:latest") 
TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", 0.5))
headers = {"Content-Type": "application/json"}
USE_LOCAL = os.getenv("USE_LOCAL", "true").lower() == "true"

def call_medgemma( prompt ):
    data = {"model": CHAT_MODEL, "prompt": prompt, "stream":False}
    response = requests.post(LLM_API_URL, json=data)
    response.raise_for_status()
    text = response.json()
    return text.get("response", "").strip()


def classify_relationships(entries):
    results = {"results": []}

    for entry in tqdm(entries, desc="Classifying relationships"):
        prompt = PROMPT_TEMPLATE.format(
            classes=", ".join(PREDICATE_CLASSES),
            subject=entry["subject"],
            object=entry["object"],
            relationship_text=entry["relationship"],
            abstract=entry["abstract"],
        )

        try:
            text = call_medgemma(prompt)
            start = text.find("{")
            end = text.rfind("}") + 1
            json_part = text[start:end] if start != -1 and end != -1 else '{}'
            try:
                parsed = json.loads(json_part)
            except:
                parsed = {"predicate": "NONE", "negated": False}

            results["results"].append({
                "subject": entry["subject"],
                "object": entry["object"],
                "relationship": entry["relationship"],
                "top_choice": {
                    "predicate": parsed.get("predicate", "NONE"),
                    "negated": parsed.get("negated", False),
                    "selector": CHAT_MODEL
                }
            })

        except Exception as e:
            print(f"Error processing entry {entry}: {e}")
            results["results"].append({
                "subject": entry["subject"],
                "object": entry["object"],
                "relationship": entry["relationship"],
                "top_choice": {
                    "predicate": "NONE",
                    "negated": False,
                    "selector": CHAT_MODEL
                }
            })

    return results


if __name__ == "__main__":
    PREDICATE_CLASSES = [
        "upregulator",
        "downregulator",
        "agonist",
        "antagonist",
        "substrate",
        "product",
        "regulator",
        "modulator"
    ]

    PROMPT_TEMPLATE = """
    You are a biomedical relation classification model.
    Given the following information, choose the predicate that best describes the relationship
    between the subject and object, based on the context in the abstract.

    If no predicate meaningfully applies, return "NONE".

    Also indicate if the chosen predicate should be considered a *negated* variant
    (e.g., “does not activate”, “not associated with”, etc.).

    Return only JSON in this format:
    {{
      "predicate": "<one of {classes} or NONE>",
      "negated": <true or false>
    }}

    ---
    Subject: {subject}
    Object: {object}
    Relationship text: {relationship_text}
    Abstract: {abstract}
    """.strip()
    with open("newest_chemprot_test_file.json", "r") as f:
        data = json.load(f)

    results = classify_relationships(data)

    with open("outputs1/base_llm_on_chemprot_results_medgemma:27b.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nClassification completed. Results saved to 'predicate_results.json'.")

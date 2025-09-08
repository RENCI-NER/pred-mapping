import argparse
import json
from src.utils import load_from_json
from bmt import Toolkit


def clean_mappings(mappings_file, negations_file, no_has=True):
    with open(mappings_file, "r") as m:
        mappings = json.load(m)

    with open(negations_file, "r") as n:
        negations = json.load(n)

    for key in mappings:
        regular = mappings[key]
        negation = negations[f"{key} NEG"]
        if isinstance(regular, list):
            assert len(regular) == len(negation), f"{key}, Reg: {len(regular)}, Neg: {len(negation)}"
            i = 0
            while regular and i < len(regular):
                if negation[i] == "NOT ENOUGH INFORMATION" or "negation" in negation[i] or negation[i] == "" or not negation[i]:
                    regular.pop(i)
                    negation.pop(i)
                elif no_has and regular[i] == "has":
                    regular.pop(i)
                    negation.pop(i)
                else:
                    i += 1
            mappings[key] = regular
            negations[f"{key} NEG"] = negation
        elif negation == "NOT ENOUGH INFORMATION" or "negation" in negation or negation == "":
            mappings[key] = []
            negations[f"{key} NEG"] = []

    mappings_out = mappings_file.replace(".json", "_cleaned.json")
    negations_out = negations_file.replace(".json", "_cleaned.json")
    with open(mappings_out, "w") as mout:
        mout.write(json.dumps(mappings, indent=2))

    with open(negations_out, "w") as nout:
        nout.write(json.dumps(negations, indent=2))


def merge_mappings(mappings_file, negations_file, output_file):
    mappings = load_from_json(mappings_file)
    negations = load_from_json(negations_file)

    mappings.update(negations)

    with open(output_file, 'w') as f:
        f.write(json.dumps(mappings, indent=2))


def cull_mapped_predicates(mapped_predicate_file, qualified_mapping_file):
    """ Uses file with embedding vector mapping """
    predicates = mapped_predicate_file if isinstance(mapped_predicate_file, list) else load_from_json(mapped_predicate_file)

    qualified_mappings = qualified_mapping_file if isinstance(qualified_mapping_file, dict) else load_from_json(qualified_mapping_file)

    remove_domains = [
        "agent",
        "publication",
        "information content entity"
    ]
    avoid = set()
    t = Toolkit()
    keep_predicates = []
    for entry in predicates:
        raw_predicate = entry["predicate"]
        predicate = raw_predicate.replace("biolink:", "").replace("_NEG", "")
        element = t.get_element(predicate)

        if not element:  # If it's not a real predicate, rather it was inferred from qualified_predicate yaml file
            predicate = qualified_mappings.get(raw_predicate, qualified_mappings.get(raw_predicate.replace("_NEG", ""))).get("predicate").replace("biolink:", "")
            if not predicate:
                print("what happ")
            element = t.get_element(predicate)

        keep = True

        try:
            if element.domain in remove_domains or element.deprecated is not None:
                keep = False
        except AttributeError:
            print(predicate)

        try:
            while keep and element.is_a is not None:
                element = t.get_element(element.is_a)
                if element.name == "related to at concept level" and element.name!="has_chemical_role":
                    keep = False
                    avoid.add(predicate)
        except AttributeError as e:
            print(f"e :{e}, \n**entry : {entry}")

        if keep:
            keep_predicates.append(entry)

    print(f"Loaded {len(predicates)} Culled to {len(keep_predicates)} predicates.")

    return keep_predicates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ontology = "biolink"
    parser.add_argument("-m", "--mappings", default=f"{ontology}_mappings.json", help="Mappings file")
    parser.add_argument("-n", "--negations", default=f"negated_{ontology}_mappings.json", help="Negation mappings file")
    parser.add_argument("-a", "--all_mappings", default=f"all_{ontology}_mappings.json", help="Output mappings file")
    args = parser.parse_args()

    mappings_file = args.mappings
    negations_file = args.negations
    clean_mappings(mappings_file, negations_file)

    cleaned_mappings_file = mappings_file.replace(".json", "_cleaned.json")
    cleaned_negations_file = negations_file.replace(".json", "_cleaned.json")
    all_mappings_file = args.all_mappings
    merge_mappings(cleaned_mappings_file, cleaned_negations_file, all_mappings_file)

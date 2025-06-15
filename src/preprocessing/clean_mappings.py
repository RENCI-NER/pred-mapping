import argparse
import json
from bmt import Toolkit
from collect_predicate_text import TextCollector


def clean_mappings( mappings_file, negations_file, no_has=True ):
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
                if negation[i] == "NOT ENOUGH INFORMATION" or regular[i] == "" or not regular[i]:
                    regular.pop(i)
                    negation.pop(i)
                elif no_has and regular[i] == "has":
                    regular.pop(i)
                    negation.pop(i)
                else:
                    i += 1
            mappings[key] = regular
            negations[f"{key} NEG"] = negation
        elif negation == "NOT ENOUGH INFORMATION":
            mappings[key] = []
            negations[f"{key} NEG"] = []

    mappings_out = mappings_file.replace(".json", "_cleaned.json")
    negations_out = negations_file.replace(".json", "_cleaned.json")
    with open(mappings_out, "w") as mout:
        mout.write(json.dumps(mappings, indent=2))

    with open(negations_out, "w") as nout:
        nout.write(json.dumps(negations, indent=2))


def merge_mappings( mappings_file, negations_file, output_file ):
    with open(mappings_file) as f:
        mappings = json.load(f)

    with open(negations_file) as f:
        negations = json.load(f)

    mappings.update(negations)

    with open(output_file, 'w') as f:
        f.write(json.dumps(mappings, indent=2))


def cull_mapped_predicates( mapped_predicate_file ):
    """ Uses file with embedding vector mapping """
    with open(mapped_predicate_file, "r") as f:
        predicates = json.load(f)
        print(f"Loaded {len(predicates)} predicates.")

    remove_domains = [
        "agent",
        "publication",
        "information content entity"
    ]

    t = Toolkit()
    tc = TextCollector()
    qualified_mappings = tc.retrieve_qualified_mappings(reverse=True)
    keep_predicates = []
    for entry in predicates:
        raw_predicate = entry["predicate"]
        predicate = raw_predicate.replace("biolink:", "").replace("_NEG", "")
        element = t.get_element(predicate)

        if not element:  # If it's not a real predicate, rather it was inferred from qualified_predicate yaml file
            predicate = qualified_mappings.get(raw_predicate, qualified_mappings.get(raw_predicate.replace("_NEG", "")))
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
                if element.name == "related to at concept level":
                    keep = False
        except AttributeError as e:
            print(f"e :{e}, \n**entry : {entry}")

        if keep:
            keep_predicates.append(entry)

    print(f"Culled to {len(keep_predicates)} predicates.")

    return keep_predicates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mappings", default="biolink_mappings.json", help="Mappings file")
    parser.add_argument("-n", "--negations", default="llama_negated_biolink_mappings.json", help="Negation mappings file")
    parser.add_argument("-a", "--all_mappings", default="llama_all_biolink_mappings.json", help="Output mappings file")
    args = parser.parse_args()

    mappings_file = args.mappings
    negations_file = args.negations
    clean_mappings(mappings_file, negations_file)

    cleaned_mappings_file = mappings_file.replace(".json", "_cleaned.json")
    cleaned_negations_file = negations_file.replace(".json", "_cleaned.json")
    all_mappings_file = args.all_mappings
    merge_mappings(cleaned_mappings_file, cleaned_negations_file, all_mappings_file)

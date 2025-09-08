import json
import argparse


def generate_mappings_and_dictionary(input_ontology, input_mappings_file, output_mappings_file, dict_file):
    # Define ontology prefix
    if input_ontology.lower() == "chemprot":
        prefix = ""
    elif input_ontology.lower() == "biolink":
        prefix = "biolink:"
    else:
        raise ValueError(f"Unknown ontology: {input_ontology}")

    # Load JSON mappings
    with open(input_mappings_file, "r", encoding="utf-8") as f:
        lines = json.load(f)

    dictionary_entries = set()

    # Write expanded predicate-synonym pairs
    with open(output_mappings_file, "w", encoding="utf-8") as f_out:
        for predicate, synonyms in lines.items():
            concept_id = f"{prefix}{predicate.replace(' ', '_')}"
            for synonym in synonyms:
                synonym = synonym.strip()
                if predicate != synonym and len(synonym) > 1:
                    pair = f"{concept_id} || {predicate} || {synonym}"
                    f_out.write(f"{pair}\n")

                    # Add entries to dictionary if valid
                    if len(predicate) >= 2 and len(synonym) >= 2 and predicate.isprintable() and synonym.isprintable():
                        dictionary_entries.update([f"{concept_id}||{predicate}", f"{concept_id}||{synonym}"])

    # Save dictionary
    with open(dict_file, "w", encoding="utf-8") as f_dict:
        f_dict.write("\n".join(sorted(dictionary_entries)))

    print(f"Mappings written to: {output_mappings_file}")
    print(f"Dictionary written to: {dict_file} ({len(dictionary_entries)} entries)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ontology = "biolink"  # or "chemprot"
    parser.add_argument("-m", "--mappings", default=f"all_{ontology}_mappings.json", help="Input JSON mappings file")
    parser.add_argument("-a", "--all_mappings", default=f"all_{ontology}_mappings.txt", help="Output file for full predicate-synonym pairs")
    parser.add_argument("-o", "--output_dict", default=f"{ontology}_dictionary.txt", help="Output dictionary file")
    args = parser.parse_args()

    generate_mappings_and_dictionary(ontology, args.mappings, args.all_mappings, args.output_dict)

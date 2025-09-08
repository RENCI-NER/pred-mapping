import argparse
import json
import yaml
import requests
from collections import defaultdict
from typing import Optional, Dict, Any
from bmt import Toolkit


class TextCollector:
    def __init__( self ):
        self.skos = {}
        self.ro = {}
        self.uberon = {}
        self.fma = {}
        self.bspo = {}
        self.chebi = {}
        self.mondo = {}
        self.prefix_to_properties = {
            "skos": self.skos,
            "BFO": self.ro,
            "RO": self.ro,
            "UBERON": self.uberon,
            "UBERON_CORE": self.uberon,
            "FMA": self.fma,
            "BSPO": self.bspo,
            "CHEBI": self.chebi,
            "MONDO": self.mondo,
        }
        self.prefix_to_source = {
            "skos": "skos",
            "BFO": "RO",
            "RO": "RO",
            "UBERON": "UBERON",
            "UBERON_CORE": "UBERON",
            "FMA": "FMA",
            "BSPO": "BSPO",
            "CHEBI": "CHEBI",
            "MONDO": "MONDO",
        }
        self.bad_counts = defaultdict(int)
        self.missing_counts = defaultdict(int)
        self.all_responses = {}

    @staticmethod
    def parse_ols_properties( responses, onto_properties ):
        property_fields = ["description", "synonyms"]
        annotation_fields = ["definition", "description", "editor preferred term", "alternative label"]
        for response in responses:
            print(len(response["_embedded"]["properties"]))
            for property in response["_embedded"]["properties"]:
                iri = property["iri"]
                text = []
                if "label" in property:
                    text.append(property["label"])
                for field in property_fields:
                    if field in property:
                        text += property[field]
                if "annotation" in property:
                    for field in annotation_fields:
                        if field in property["annotation"]:
                            text += property["annotation"][field]

                # Remove empty strings
                text = [entry for entry in text if entry]

                if text:
                    onto_properties[iri] = text

    @staticmethod
    def expand_curie( curie ):
        expansions = {
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "BFO": "http://purl.obolibrary.org/obo/BFO_",
            "RO": "http://purl.obolibrary.org/obo/RO_",
            "FMA": "http://purl.obolibrary.org/obo/FMA#",
            "UBERON": "http://purl.obolibrary.org/obo/uberon_",
            "UBERON_CORE": "http://purl.obolibrary.org/obo/uberon/core#",
            "BSPO": "http://purl.obolibrary.org/obo/BSPO_",
            "CHEBI": "http://purl.obolibrary.org/obo/CHEBI_",
            "MONDO": "http://purl.obolibrary.org/obo/MONDO_",
        }
        prefix, suffix = curie.split(":")
        if prefix in expansions:
            return expansions[prefix] + suffix

        print("Bad prefix", prefix)
        return None

    def collect_ontology_text( self, curie ):
        prefix = curie.split(":")[0]
        try:
            source = self.prefix_to_source[prefix]
        except KeyError:
            self.bad_counts[prefix] += 1
            return []

        onto_properties = self.prefix_to_properties[source]
        if len(onto_properties) == 0:
            self.refresh_ontology_properties(source, onto_properties)

        iri = self.expand_curie(curie)
        if iri not in onto_properties:
            print("Missing", curie, iri)
            self.missing_counts[prefix] += 1
            x = " ".join(curie.split(":")[1].split("_"))

            # If the string is an integer, we don't want it, but if it's text, we do
            if x.isdigit():
                return []

            return [x]

        return onto_properties.get(iri, [])

    def refresh_ontology_properties( self, prefix, onto_properties ):
        responses = []
        page = 0
        print(prefix)
        while True:
            url = f"https://www.ebi.ac.uk/ols4/api/ontologies/{prefix.lower()}/properties?size=500"
            if page > 0:
                url += f"&page={page}"
            # print(url)
            resp = requests.get(url)
            resp.raise_for_status()
            response = resp.json()
            responses.append(response)
            print(response["page"])
            page += 1
            if response["page"]["totalPages"] == page:
                print("No more")
                break
        self.all_responses[prefix] = responses
        self.parse_ols_properties(responses, onto_properties)

    def dump_responses( self ):
        with open("../../litcoin_testing/responses.json", "w") as f:
            f.write(json.dumps(self.all_responses, indent=2))

    def collect_text( self, curie ):
        pref = curie.split(":")[0]
        if pref in ["UMLS", "SEMMEDDB", "RXNORM", "SNOMED", "SNOMEDCT", "NCIT", "LOINC", "REPODB"]:
            return [' '.join(curie[len(pref) + 1:].split("_"))]

        return self.collect_ontology_text(curie)

    @staticmethod
    def format_predicate_mapping( mapping_dict ):
        biolink_mappings = defaultdict(set)
        for predicate, text_dict in mapping_dict.items():
            for entry in text_dict:
                if entry == "text":
                    biolink_mappings[predicate].update(
                        [text.replace("\n", " ") for text in text_dict[entry] if len(text.replace("\n", " ")) > 1])
                else:
                    for text_list in text_dict[entry].values():
                        biolink_mappings[predicate].update(
                            [text.replace("\n", " ") for text in text_list if len(text.replace("\n", " ")) > 1])

        biolink_mappings = {key: list(val) for key, val in biolink_mappings.items()}

        return biolink_mappings

    def retrieve_qualified_mappings( self, reverse: bool = False, q_output_file: Optional[str] = None ) -> Dict[
        str, Any]:
        """
        Fetches and parses the predicate mapping YAML file from the Biolink Model repository.

        Args:
            reverse (bool): If True, return reverse mapping format.
            q_output_file (Optional[str]): If set, write the results to a JSON file.

        Returns:
            dict: Mapping of predicates to qualifiers, or reverse mapping which is qualifier to predicate mappings.
        """
        yaml_url = "https://raw.githubusercontent.com/biolink/biolink-model/master/predicate_mapping.yaml"
        unwanted_matches = [
            "releasing_agent", "partial_agonist", "channel_blocker",
            "antisense_inhibitor", "negative_allosteric_modulator",
            "inverse_agonist", "gating_inhibitor", "AUGMENTS", "opener", "blocker", "ki", "ic50", "vaccine", "INHIBITS"
        ]
        response = requests.get(yaml_url)
        response.raise_for_status()
        predicate_data = yaml.safe_load(response.text)

        reverse_mappings = defaultdict(dict)
        entries = {}

        matches_key = {"exact matches", "narrow matches", "close matches"}

        for mapping in predicate_data.get("predicate mappings", []):
            predicate = mapping.get("predicate", mapping.get("qualified predicate"))
            if not predicate:
                continue
            qualified_predicate = mapping.get("qualified predicate", "")
            direction_qualifier = mapping.get("object direction qualifier", "")
            aspect_qualifier = mapping.get("object aspect qualifier", "")
            mapped_predicate = mapping.get("mapped predicate", "")
            current_mapping = matches_key.intersection(mapping)
            if not current_mapping and len(mapped_predicate.split(" ")) > 1:
                filtered_matches = [mapped_predicate]
                if qualified_predicate:
                    if aspect_qualifier and direction_qualifier:
                        extra_text = f"{qualified_predicate} {direction_qualifier} {aspect_qualifier}".strip()
                    else:
                        extra_text = f"{mapped_predicate}".strip()
                    text_list = [extra_text, mapped_predicate]
                else:
                    text_list = [f"{mapped_predicate}".strip()]
                entries[mapped_predicate] = {"text": text_list}
            else:
                filtered_matches = []
                for mapping_type in current_mapping:
                    entries[mapped_predicate] = {}
                    _matches = mapping[mapping_type]
                    mapping_dict = self.collect_mapping_data(_matches)
                    if mapping_dict:
                        entries[mapped_predicate][mapping_type] = mapping_dict
                    elif mapping_type == "exact matches":
                        for match in _matches:
                            good_prefix = match.split(":")[0] == "CTD"
                            match = match.split(":")[1]
                            if good_prefix and match not in unwanted_matches:
                                filtered_matches.append(match)
                                qualifier = match.replace("_", " ")
                                if qualified_predicate:
                                    if aspect_qualifier and direction_qualifier:
                                        extra_text = f"{qualified_predicate} {direction_qualifier} {aspect_qualifier}".strip()
                                    else:
                                        extra_text = f"{qualified_predicate} {qualifier}".strip()
                                    text_list = [extra_text, qualifier]
                                else:
                                    text_list = [f"{mapped_predicate}".strip()]
                                text_list = [t for t in text_list if t]
                                if entries.get(qualifier, {}):
                                    entries[qualifier]["text"].extend(text_list)
                                else:
                                    entries[qualifier] = {"text": text_list}
            if reverse:
                for match in filtered_matches:
                    reverse_mappings[f"biolink:{match.replace(" ", "_")}"] = {
                        "predicate": f"biolink:{predicate}",
                        "qualified_predicate": qualified_predicate,
                        "object_aspect_qualifier": aspect_qualifier.replace(" ", "_"),
                        "object_direction_qualifier": direction_qualifier.replace(" ", "_")
                    }

        if q_output_file is not None:
            with open(q_output_file, "w") as file:
                json.dump(reverse_mappings, file, indent=2)
        return entries

    def collect_mapping_data( self, predicate_mapping_type ):
        """Helper function to collect mapping data for a given predicate and mapping type."""
        mapping_dict = {}
        for curie in predicate_mapping_type:
            prefix = curie.split(":")[0]
            if prefix in self.prefix_to_source:
                collected = self.collect_text(curie)
                if collected:
                    mapping_dict[curie] = collected
        return mapping_dict

    def run(self, output_file=None, qualified_mappings_file=None):
        t = Toolkit()
        remove_domains = {"agent", "publication", "information content entity"}
        # Also saves the qualified mapping file to the directory
        qualified_mappings_dict = self.retrieve_qualified_mappings(reverse=True, q_output_file=qualified_mappings_file)
        predicates = t.get_descendants("biolink:related_to", formatted=False)
        entries = {}
        no_inverse = []
        inverses = [t.get_element(p).inverse for p in predicates if t.get_element(p).inverse]
        for p in predicates:
            predicate = t.get_element(p)
            if predicate.deprecated:
                continue
            if predicate.domain in remove_domains:
                continue

            if not t.has_inverse(p) and not predicate.symmetric and p not in inverses:
                no_inverse.append(p)

            text = [p]
            if predicate.description:
                text.append(predicate.description)

            entries[p] = {"text": text}

            for mapping_type in ["exact_mappings", "narrow_mappings", "close_mappings"]:
                if predicate[mapping_type]:
                    mapping_dict = self.collect_mapping_data(predicate[mapping_type])
                    if mapping_dict:
                        entries[p][mapping_type] = mapping_dict
            # To cater for aspect/direction qualifiers like increases expression which are not the original predicates
            entries.update(qualified_mappings_dict)

        if output_file is not None:
            with open(output_file, "w") as file:
                file.write(json.dumps(self.format_predicate_mapping(entries), indent=2))

        bads = [(c, bad) for bad, c in self.bad_counts.items()]
        bads.sort(reverse=True)
        print(f"bad || count")
        for count, bad in bads:
            print(f"{bad}: {count}")

        missing = [(c, miss) for miss, c in self.missing_counts.items()]
        missing.sort(reverse=True)
        print(f"missing || count")
        for count, miss in missing:
            print(f"{miss}: {count}")

        print(f"No inverse: {no_inverse}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mappings", default="biolink_mappings.json", help="Mappings file")
    parser.add_argument("-q", "--qualified_mappings", default="qualified_predicate_mappings.json",
                        help=" Qualified mappings file")
    args = parser.parse_args()
    mappings = args.mappings
    qualified_mappings = args.qualified_mappings
    tc = TextCollector()
    tc.run(output_file=mappings, qualified_mappings_file=qualified_mappings)

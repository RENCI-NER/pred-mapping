import argparse
import json
import yaml
import requests
from collections import defaultdict
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

            #If the string is an integer, we don't want it, but if it's text, we do
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
            response = requests.get(url).json()
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
                    biolink_mappings[predicate].update([text.replace("\n", " ") for text in text_dict[entry]])
                else:
                    for text_list in text_dict[entry].values():
                        biolink_mappings[predicate].update([text.replace("\n", " ") for text in text_list])

        biolink_mappings = {key: list(val) for key, val in biolink_mappings.items()}

        return biolink_mappings

    def retrieve_qualified_mappings( self, reverse=False ):
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
            matches = mapping.get("exact matches", [])
            filtered_matches = [match.split(":")[1] for match in matches if
                                ":" in match and not match.split(":")[1].isdigit() and "_" in match and match.split(":")[1] not in unwanted_matches]
            if reverse:
                mapping_dict.update({f"biolink:{match}": predicate for match in filtered_matches})
            else:
                mapping_dict[predicate].extend([" ".join(match.split("_")) for match in filtered_matches])

        return dict(mapping_dict)

    def collect_mapping_data( self, predicate_mapping_type ):
        """Helper function to collect mapping data for a given predicate and mapping type."""
        mapping_dict = {}
        for curie in predicate_mapping_type:
            prefix = curie.split(":")[0]
            if prefix in self.prefix_to_source:
                mapping_dict[curie] = self.collect_text(curie)
        return mapping_dict

    def run( self, output_file=None ):
        t = Toolkit()
        qualified_mappings = self.retrieve_qualified_mappings()
        predicates = t.get_descendants("biolink:related_to", formatted=False)
        entries = {}
        no_inverse = []
        inverses = [t.get_element(p).inverse for p in predicates if t.get_element(p).inverse]
        for p in predicates:
            predicate = t.get_element(p)
            if not t.has_inverse(p) and not predicate.symmetric and p not in inverses:
                no_inverse.append(p)

            if predicate.deprecated:
                continue

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

            for qualified_predicate in qualified_mappings.get(p, []):
                entries[qualified_predicate] = {"text": text + [qualified_predicate]}
                for mapping_type in ["exact_mappings", "narrow_mappings", "close_mappings"]:
                    if mapping_type in entries[p]:
                        entries[qualified_predicate][mapping_type] = entries[p][mapping_type]

        if output_file is not None:
            with open(output_file, "w") as file:
                file.write(json.dumps(self.format_predicate_mapping(entries), indent=2))

        bads = [(c, bad) for bad, c in self.bad_counts.items()]
        bads.sort(reverse=True)
        for count, bad in bads:
            print(f"{bad}: {count}")

        missing = [(c, miss) for miss, c in self.missing_counts.items()]
        missing.sort(reverse=True)
        for count, miss in missing:
            print(f"{miss}: {count}")

        print(f"No inverse: {no_inverse}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mappings", default="biolink_mappings.json", help="Mappings file")
    args = parser.parse_args()
    mappings = args.mappings
    tc = TextCollector()
    tc.run(output_file=mappings)

# Healpaca Relationship -> Biolink Predicate Mapping

## General workflow

### Biolink Predicate Descriptions
- Collect predicate text and descriptors of each predicate and any ontology terms that map to them 
 ```bash
    collect_predicate_text.py [-m mappings_file]
  ```
 Scrapes related ontologies and saves a JSON file to `mappings_file` with Biolink predicates as keys and the list of text descriptors as values.
- Generate negations version of each description 
  ```bash
    get_negations.py [-m mappings_file -n negations_file]
  ```
 Takes in the mapping file, sends each descriptor to OpenAI to produce negated versions. Saves the results to `negations_filename`.
- Merge and clean all mappings 
    ```bash
      clean_mappings.py [-m mappings_file -n negations_file -a all_mappings_file]
    ```
 Takes in mapping file and negations file, removes any LLM "not enough information" responses or empty strings, and merges into `all_mappings_file`. **If you try to merge a newly generated mapping file with the existing negations file in the Drive, this will break.** The `TextCollector` was updated and returns fewer bad descriptors, but the negations weren't regenerated to reduce spending. The old mappings file is in the Drive, so this step can be tested using that, or negations can be regenerated.
- Embed the cleaned predicates/descriptors and saved for API use 
    ```bash
      embed_biolink_mappings.py [-m mappings_file -e embeddings_file --lowercase]
    ```
 Takes in a mapping file (typically all mappings), sends them to OpenAI for embedding, then saves as `embedding_file`. **Currently this does not have batch submission.**



### Healpaca Relationships Extraction
[detailed in](../../README.md)

## Pipeline


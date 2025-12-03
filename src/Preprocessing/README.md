# Preprocessing Pipeline: 

This pipeline prepares ontology data for the predicate mapping service. It supports multiple ontologies including Biolink and ChemProt.

## Workflow

1. Navigate to the directory:
    ```bash
      cd pred-mapping/src/Preprocessing
   ```
   Then run the following module one after the other

2. Collect predicate text and descriptions: This module scrapes biolink ontology and saves JSON files to 
   - `mappings_file`: predicates as keys with lists of text descriptors as values,
   - `qualified_predicate_mappings.json` mappings of predicates to qualifiers for qualified predicates
   - `biolink_short_description.json` predicates as keys with short text descriptors as values 

**Note:** This module only works for the Biolink ontology, as ChemProt's semantic meanings are adapted from [PMC10215465](https://pmc.ncbi.nlm.nih.gov/articles/PMC10215465/table/bioengineering-10-00586-t001/) and additional definitions manually scraped using Google search.

```bash
collect_predicate_text.py [-m mappings_file -q qualified_mappings -d short_description] 
```

2. Generate negations: Takes the `mappings_file` and sends each descriptor to the LLM to produce natural negated versions. Saves results to `negations_file`.

```bash
get_negations.py [-m mappings_file -n negations_file]
```

3. Merge and clean mappings: Merges mapping and negations files, removes LLM "not enough information" responses or empty strings, and saves to `all_mappings_file`.

```bash
clean_mappings.py [-m mappings_file -n negations_file -a all_mappings_file] 
```

4. Embed predicate descriptors for downstream similarity search: Takes the `all_mappings_file` and generates embeddings (saves output JSON to `all_biolink_mapped_vectors.json`) using the configured embedding model. Default embedding dimension is [`nomic-embed-text` (`768` dim)](https://ollama.com/library/nomic-embed-text).

```bash
embed_biolink_mappings.py [-m mappings_file -e embeddings_file --lowercase] 
```

---

## Output Structure

After preprocessing, the directory structure should contain:
- If Biolink Model was used:
  - `biolink_data/biolink_short_description.json`
  - `biolink_data/all_biolink_mapped_vectors.json`
  - `biolink_data/qualified_predicate_mappings.json`

- if ChemProt was used:
  - `chemprot_data/chemprot_short_description.json`
  - `chemprot_data/all_chemprot_mapped_vectors.json`
  - `chemprot_data/qualified_predicate_mappings.json` (empty for ChemProt)

In general, 


```
   pred-mapping/
   ├── {ontology}_data/
   │   ├── {ontology}_short_description.json
   │   ├── all_{ontology}_mapped_vectors.json
   │   └── qualified_predicate_mappings.json
   └── src/
       ├── Preprocessing/                    
       ├── llm_client.py
       ├── ontology_config.py
       ├── predicate_database.py
       ├── predicate_lookup.py 
       ├── server.py    
       └── utils.py
```

Where `{ontology}` is `biolink` or `chemprot`.

## Important Notes
- **Batch Processing:** Current implementation processes embeddings in batches of 25 input relationships
- **Version Compatibility:** Ensure mapping and negation files are generated with compatible versions
- **Local LLM:** Uses Ollama by default for local processing

## Next Steps

After preprocessing, use the generated files with the main prediction service. For detailed next steps, see [README.md](../README.md).
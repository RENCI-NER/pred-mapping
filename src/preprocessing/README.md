# Preprocessing Pipeline: Input Relationship → Ontology Predicate Mapping

This preprocessing pipeline collects **[Biolink](https://github.com/biolink/biolink-model) schema predicate** for the predicate mapping service.  
It generates cleaned predicate mappings, negations, and embeddings for eventual similarity-based predicate matching.  

---

## General Workflow

### 1. Collect Ontology Predicate Descriptions
```bash
collect_predicate_text.py [-m mappings_file -q qualified_mappings] 
```

### 2. Generate Negations: Using an LLM to produce natural negated versions of each descriptor and saves results to negations_file..
```bash
get_negations.py [-m mappings_file -n negations_file] 
```

### 3. Merge and Clean Mappings: Removes invalid responses or empty strings, merges mapping and negation files then outputs all_mappings_file.
```bash
clean_mappings.py [-m mappings_file -n negations_file -a all_mappings_file]
```

### 4. Embed Predicates for Similarity Search: Generates embeddings for all Biolink predicates with the default embedding dimension: 768.
```bash
embed_biolink_mappings.py [-m mappings_file -e embeddings_file --lowercase]
```

## Environment Setup

Before running the pipeline, configure the environment:

### LLM configuration (used for negation generation and reranking)
```
export LLM_API_URL=http://localhost:11434/api/generate
export CHAT_MODEL=alibayram/medgemma:latest
export MODEL_TEMPERATURE=0.5
```

### Embedding configuration (used to embedd the ontology predicates and free-text input relationships)
```
export EMBEDDING_URL=http://localhost:11434/api/embeddings
export EMBEDDING_MODEL=nomic-embed-text
```

### Use local models
export USE_LOCAL=true

### Output Structure 

After preprocessing, the directory structure should look like this:

```text
data/
├── short_description.json            # Some predicate descriptions
├── all_biolink_mapped_vectors.json   # Embeddings for predicates
├── qualified_predicate_mappings.json # Qualifier mappings
```

### Important Notes

 - Embedding Dimension: Default is 768 (nomic-embed-text)
 - Batch Size: Embeddings are processed in batches of 25 relationships
 - Version Compatibility: Ensure mapping and negation files are generated with the same ontology version
 - Local LLM: Uses Ollama  by default for local processing
   - [![Download Ollama](https://img.shields.io/badge/Download-Ollama-blue?style=for-the-badge)](https://ollama.com/download)
 - Cost Optimization: Negations are generated once and reused to reduce LLM API calls









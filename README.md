# Biolink Predicate Mapping Pipeline

A two-part pipeline for mapping biomedical relationships (subject, object, context) to standardized Biolink predicates using a combination of embedding similarity and language model reasoning.

## Overview

The system originally from [LitCoin](https://github.com/ncats/LitCoin/blob/main/predicates/README.md) is split into two stages:

1. **Preprocessing Stage** (run infrequently):
   - Collect predicate text and short descriptions `collect_predicate_text.py [-m mappings_file -d short_description]`
   - Generate negations `get_negations.py [-m mappings_file -n negations_file]`
   - Merge and clean all mappings `clean_mappings.py [-m mappings_file -n negations_file -a all_mappings_file]`
   - Embed the cleaned predicates and saved for API use `embed_biolink_mappings.py [-m mappings_file -e embeddings_file --lowercase]`

2. **FastAPI Inference Service**:
   - Loads precomputed embeddings and descriptions
   - Accepts subject-object-relationship-context HEALpaca inputs
   - Returns the inputs each with nearest-neighbors matching predicates(s) and the top-matching Biolink predicate

---

## Preprocessing Pipeline

[LitCoin](https://github.com/ncats/LitCoin/blob/main/predicates/README.md)

## FastAPI Inference 
From the terminal: 
```bash
uvicorn main:app --reload 
```

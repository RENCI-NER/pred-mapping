# Multi-Ontology Predicate Mapping Pipeline

A two-stage pipeline for mapping biomedical relationships (subject, object, context) to standardized predicates using embedding similarity, SapBERT predictions, and language model reasoning. Supports multiple ontologies including Biolink and ChemProt.

## Overview

The system consists of two stages:

1. **Preprocessing Stage** (run infrequently):
   - Collect predicate text and descriptions
   - Generate negations and clean mappings
   - Embed predicates for similarity search
   - Optional: Train SapBERT models for enhanced accuracy

2. **FastAPI Inference Service**:
   - Loads precomputed embeddings and descriptions
   - Accepts subject-object-relationship-context inputs
   - Uses vector similarity + optional SapBERT predictions
   - Returns top-matching predicates with LLM reranking

## Supported Ontologies

- **Biolink Model**: High-level datamodel of biological entities and associations
  - Source: [biolink-model](https://github.com/biolink/biolink-model)
  - Schema: [biolink-model.yaml](https://github.com/biolink/biolink-model/blob/master/biolink-model.yaml)

- **ChemProt**: Chemical-protein interaction corpus for relation extraction
  - Source: [BioCreative VI](hhttps://huggingface.co/datasets/bigbio/chemprot)
  - Paper: [PMC5721660](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5721660/)

---

## Setup and Installation

### Local Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the server:
   ```bash
   uvicorn src.server:APP --reload
   ```

### Docker Setup (MacBook)

1. Build the image:
   ```bash
   docker buildx build --platform linux/amd64,linux/arm64 -t predmapping:v1 --push .
   ```

2. Create cache directory:
   ```bash
   mkdir -p .cache
   chmod 777 .cache
   ```

3. Run the container:
   ```bash
   docker run --rm \
     --platform linux/amd64 \
     -p 6380:6380 \
     predmapping:v1
   ```

### Environment Configuration

Set the default ontology:
```bash
export ONTOLOGY=biolink  # or chemprot
```

---

## API Usage

### Swagger UI
Access the interactive API documentation at:
```
http://localhost:8000/docs
```

### Configuration Endpoints

**List available ontologies:**
```bash
GET /ontologies
```

### Query Endpoint

**Endpoint:** `POST /query/`

**Parameters:**
- `ontology`: Ontology to use (biolink/chemprot)
- `similarity_based_retrieval_method`: Search method (sklearn_knn/scipy_cosine)
- `use_sapbert`: Enable SapBERT predictions (true/false)

**Input Example:**
```json
[
  {
    "subject": "Haloperidol",
    "object": "Prolactin", 
    "relationship": "increases levels of",
    "abstract": "The effects of a 6-hour infusion with haloperidol on serum prolactin and luteinizing hormone (LH) levels was studied in a group of male subjects. Five hours after starting the infusions, a study of the pituitary responses to LH-releasing hormone (LH-RH) was carried out. Control patients received infusions of 0.9% NaCl solution. During the course of haloperidol infusions, significant hyperprolactinemia was found, together with an abolished pituitary response to LH-RH, as compared with responses of control subjects."
  }
]
```

**Response Example:**
```json
{
  "results": [
    {
      "subject": "Haloperidol",
      "object": "Prolactin",
      "relationship": "increases levels of",
      "top_choice": {
        "predicate": "increased amount of",
        "object_aspect_qualifier": "",
        "object_direction_qualifier": "",
        "negated": false,
        "selector": "medgemma:7b"
      },
      "Top_n_candidates": {
        "0": {
          "mapped_predicate": "increased amount of",
          "score": 0.84652
        },
        "1": {
          "mapped_predicate": "has increased amount",
          "score": 0.82094
        }
      },
      "Top_n_retrieval_method": "sklearn_knn"
    }
  ],
  "ontology": "biolink"
}
```

### Usage Examples

**Use specific ontology:**
```bash
curl -X POST "http://localhost:8000/query/?ontology=chemprot" \
  -H "Content-Type: application/json" \
  -d '[{"subject": "Drug", "object": "Protein", "relationship": "inhibits", "abstract": "..."}]'
```

**Enable SapBERT:**
```bash
curl -X POST "http://localhost:8000/query/?use_sapbert=true" \
  -H "Content-Type: application/json" \
  -d '[...]'
```

**Use different similarity method:**
```bash
curl -X POST "http://localhost:8000/query/?similarity_based_retrieval_method=scipy_cosine" \
  -H "Content-Type: application/json" \
  -d '[...]'
```

---

## Architecture

### Components

1. **Ontology Configuration** (`src/ontology_config.py`): Manages multiple ontology settings
2. **Predicate Lookup** (`src/predicate_lookup.py`): Core similarity search and SapBERT integration
3. **FastAPI Server** (`src/server.py`): REST API endpoints
4. **LLM Client** (`src/llm_client.py`): Handles local/remote language model calls

### Data Structure

```
project/
├── biolink_data/
│   ├── biolink_short_description.json
│   ├── all_biolink_mapped_vectors.json
│   └── qualified_predicate_mappings.json
├── chemprot_data/
│   ├── chemprot_short_description.json
│   ├── all_chemprot_mapped_vectors.json
│   └── qualified_predicate_mappings.json
└── src/
    ├── Biolink_SapBert/                    # Optional SapBERT models
    │   ├── data/
    │   │   └── embedding_mappings.npy
    │   └── model/
    └── Chemprot_SapBert/
        ├── data/
        │   └── embedding_mappings.npy
        └── model/
```
### Features

- **Multi-ontology support**: Switch between Biolink and ChemProt
- **SapBERT integration**: Enhanced accuracy with specialized biomedical embeddings
- **Local LLM support**: Works with Ollama for local inference
- **Flexible similarity search**: Multiple retrieval methods
- **Structured output**: Pydantic validation for reliable JSON parsing

---

## Local LLM Setup (Optional)

For local inference with Ollama:

1. Install Ollama:
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. Pull models:
   ```bash
   ollama pull alibayram/medgemma:27b
   ollama pull nomic-embed-text:latest
   ```

3. Configure environment:
   ```bash
   export USE_LOCAL=true
   export CHAT_MODEL=alibayram/medgemma:27b
   export EMBEDDING_MODEL=nomic-embed-text:latest
   ```
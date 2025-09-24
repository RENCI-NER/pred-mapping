# Predicate Mapping Pipeline

A two-stage pipeline for mapping free-text biomedical relationships in (subject, object, relationships, context) to standardized Biolink predicates using a combination of embedding similarity and language model reasoning.

1. **Preprocessing Stage** (runs infrequently):

    Details: [Here](src/preprocessing/README.md)

2. **Inference Pipeline**:
   - Loads precomputed embeddings for biolink predicate; short description of the predicates and hierarchies in some of the predicates
   - Accepts subject-object-relationship-context as inputs
   - Returns the inputs each with nearest-neighbors matching predicates(s) and the top-matching Biolink predicate to replace the free-text relationship

---

## Inference Pipeline
  1. Clone this repo:
     ```bash
        https://github.com/RENCI-NER/pred-mapping.git
     ```
  2. Set the `.env` file ie the model variables eg:
        ```bash
            LLM_API_URL=http://localhost:11434/api/generate
            CHAT_MODEL=alibayram/medgemma:latest
            MODEL_TEMPERATURE=0.5
            EMBEDDING_URL=http://localhost:11434/api/embeddings
            EMBEDDING_MODEL=nomic-embed-text
        ```
        Any model ollama-based deployed as a service
        The expected embedding dimension is `768`
     - [![Download Ollama](https://img.shields.io/badge/Download-Ollama-blue?style=for-the-badge)](https://ollama.com/download)
  3. Run the Pipeline
### Option 1: Starting the Server Locally:
  1. Navigate to the project directory:
     ``` bash
         cd pred-mapping/
     ```
  2. Install dependencies:
     ```bash
        pip install -r requirements.txt
     ```
  3. From the terminal, start the server by running: 
     ```bash
        uvicorn src.server:APP --reload
     ```

### Option 2: Dockerize the Inference Pipeline:
 1. Build the image:
       ```bash
       docker buildx build --platform linux/amd64,linux/arm64 -t pred-mapping:latest --push .
       ```
 2. Once the build is complete, run:
   ```bash
   docker run --rm \
     --platform linux/amd64 \
     -p 6380:6380 \
     pred-mapping:latest
   ```
---

## API Usage

### Swagger UI
Access the interactive API documentation at:
```
http://localhost:8000/docs
```

#### Configuration Endpoints

**  Biolink Schema Details:**
```bash
GET /biolink
```

#### Query Endpoint

**Endpoint:** `POST /query/`

**Parameters:**
- `retrieval_method`: Top-k Candidate Search method (sklearn_knn/scipy_cosine)

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

### Curl Examples

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/query/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '[
  {
    "abstract": "The present study was designed to investigate the cardioprotective effects of betaine on acute myocardial ischemia induced experimentally in rats focusing on regulation of signal transducer and activator of transcription 3 (STAT3) and apoptotic pathways as the potential mechanism underlying the drug effect. Male Sprague Dawley rats were treated with betaine (100, 200, and 400 mg/kg) orally for 40 days. Acute myocardial ischemic injury was induced in rats by subcutaneous injection of isoproterenol (85 mg/kg), for two consecutive days. Serum cardiac marker enzyme, histopathological variables and expression of protein levels were analyzed. Oral administration of betaine (200 and 400 mg/kg) significantly reduced the level of cardiac marker enzyme in the serum and prevented left ventricular remodeling. Western blot analysis showed that isoproterenol-induced phosphorylation of STAT3 was maintained or further enhanced by betaine treatment in myocardium. Furthermore, betaine (200 and 400 mg/kg) treatment increased the ventricular expression of Bcl-2 and reduced the level of Bax, therefore causing a significant increase in the ratio of Bcl-2/Bax. The protective role of betaine on myocardial damage was further confirmed by histopathological examination. In summary, our results showed that betaine pretreatment attenuated isoproterenol-induced acute myocardial ischemia via the regulation of STAT3 and apoptotic pathways.",
    "subject": "Betaine",
    "object": "Bcl-2",
    "relationship": "increases expression of"
  }
]'
```

**Use different similarity method to retrieve the top-k candidates:**
```bash
curl -X POST "http://localhost:8000/query/?retrieval_method=scipy_cosine" \
  -H "Content-Type: application/json" \
  -d '[...]'
```

---

## Architecture

### Components

1. **Schema Configuration** (`src/ontology_config.py`): Manages the ontology settings including the input files
2. **Predicate Lookup** (`src/predicate_lookup.py`): Core similarity search 
3. **FastAPI Server** (`src/server.py`): REST API endpoints
4. **LLM Client** (`src/llm_client.py`): Handles local/remote language model calls

### Inference Pipeline Code Structure

```
project/
├── data/
│   ├── short_description.json
│   ├── all_biolink_mapped_vectors.json
│   └── qualified_predicate_mappings.json
└── src/
    ├── biolink_predicate_lookup.py                 
    ├── config.py
    ├── llm_client.py
    ├── predicate_database.py
    ├── server.py
    └── utils.py
```
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
   
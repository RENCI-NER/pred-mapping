# Biolink Predicate Mapping Pipeline

A two-part pipeline for mapping biomedical relationships (subject, object, context) to standardized Biolink predicates using a combination of embedding similarity and language model reasoning.

1. **Preprocessing Stage** (run infrequently):

    Details: [Here](src/preprocessing/README.md)

2. **Inference Pipeline**:
   - Loads precomputed embeddings and descriptions
   - Accepts subject-object-relationship-context HEALpaca inputs
   - Returns the inputs each with nearest-neighbors matching predicates(s) and the top-matching Biolink predicate

---

## Inference Pipeline

A. Starting the server:

- Running the Pipeline from terminal, 

  1. Clone this repo:
     ```bash
        https://github.com/RENCI-NER/pred-mapping.git
     ```
  2. Navigate to the project directory:
     ``` bash
         cd pred-mapping/
     ```
  3. Open the terminal and run:
     ```bash
        pip install -r requirements.txt
     ```
  4. Create `.env` file and set the model variables eg:
        ```bash
            LLM_API_URL=https://healpaca.apps.renci.org/api/generate
            CHAT_MODEL=HEALpaca-2.0
            MODEL_TEMPERATURE=0.5
            EMBEDDING_URL=https://healpaca.apps.renci.org/api/embeddings
            EMBEDDING_MODEL=nomic-embed-text
        ```
        These models are also available as a service: 
        ```bash
            LLM_API_URL=https://ollama.apps.renci.org/api/generate
            CHAT_MODEL=llama3.1:latest
        ```
  5. From the terminal, start the server by running: 
     ```bash
        uvicorn src.server:APP --reload
     ```

- Dockerizing the Pipeline on MacBook:
  1. To create the docker image, run:
     ```bash
        docker buildx build --platform linux/amd64,linux/arm64 -t <image-name>:<tag> --push .
     ```
  2. Once the build is complete, run: 
     ```bash
        docker run --rm \
          --platform linux/amd64 \
          -p 6380:6380 \
          -e HOME=/tmp \
          -e JINA_HOME=/tmp/.jina \
          -v $(pwd)/.cache:/tmp/.cache \
          <image-name>:<tag>
     ```
     Note: 
        1. Replace the <image-name>:<tag> with your desired name/tag eg: predmapping:v1
        2. Make sure .cache exists and is writable:
                ```
                    mkdir -p .cache
                ```
                ```
                   chmod 777 .cache
                ```

B. From the swagger UI: 

- Send a POST request to ```/query/``` with a list of input triples
    ```bash
       [
         {
            "subject": "Haloperidol",
            "object": "Prolactin",
            "relationship": "increases levels of",
            "abstract": "The effects of a 6-hour infusion with haloperidol on serum prolactin and luteinizing hormone (LH) levels was studied in a group of male subjects. Five hours after starting the infusions, a study of the pituitary responses to LH-releasing hormone (LH-RH) was carried out. Control patients received infusions of 0.9% NaCl solution. During the course of haloperidol infusions, significant hyperprolactinemia was found, together with an abolished pituitary response to LH-RH, as compared with responses of control subjects."
        }
      ]
    ```

- Example response: 
    ```bash
        {
            "results": [
                 {
                    "subject": "Haloperidol",
                    "object": "Prolactin",
                    "relationship": "increases levels of",
                    "top_choice": {
                        "predicate": "biolink:increased_amount_of",
                        "object_aspect_qualifier": "",
                        "object_direction_qualifier": "",
                        "negated": false,
                        "selector": "vectorDB"
                    },
                    "Top_n_candidates": {
                        "0": {
                          "mapped_predicate": "increased amount of",
                          "score": 0.84652
                        },
                        "1": {
                          "mapped_predicate": "has increased amount",
                          "score": 0.82094
                        },
                        "2": {
                          "mapped_predicate": "has decreased amount",
                          "score": 0.81367
                        },
                        "3": {
                          "mapped_predicate": "decreased amount in",
                          "score": 0.81367
                        },
                        "4": {
                          "mapped_predicate": "related to",
                          "score": 0.77009
                        },
                        "5": {
                          "mapped_predicate": "increases abundance of",
                          "score": 0.75817
                        },
                        "6": {
                          "mapped_predicate": "decreases abundance of",
                          "score": 0.75817
                        },
                        "7": {
                          "mapped_predicate": "increases activity of",
                          "score": 0.75254
                        },
                        "8": {
                          "mapped_predicate": "decreases activity of",
                          "score": 0.75254
                        },
                        "9": {
                          "mapped_predicate": "increases secretion of",
                          "score": 0.73697
                        }
                    },
                    "Top_n_retrieval_method": "vectorDb"
                 }
            ]
        }
    ```

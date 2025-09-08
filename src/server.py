# src/server.py
from enum import Enum
import logging
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Extra, Field
from typing import List, Dict, Optional
from src import predicate_lookup as pl
from src.utils import load_from_json
from src.ontology_config import get_current_config, set_ontology, list_ontologies, get_ontology_details

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

APP = FastAPI(title="Ontology Predicate Mapper")


@APP.get("/", include_in_schema=False)
def root():
    return RedirectResponse("docs")


APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InputTriples(BaseModel):
    abstract: str = Field(..., example=(
        "The effects of a 6-hour infusion with haloperidol on serum prolactin and luteinizing hormone (LH) levels was studied in a group of male subjects. "
        "Five hours after starting the infusions, a study of the pituitary responses to LH-releasing hormone (LH-RH) was carried out. "
        "Control patients received infusions of 0.9% NaCl solution. "
        "During the course of haloperidol infusions, significant hyperprolactinemia was found, together with an abolished pituitary response to LH-RH, as compared with responses of control subjects."
    ))
    subject: str = Field(..., example="Haloperidol")
    object: str = Field(..., example="Prolactin")
    relationship: str = Field(..., example="increases levels of")

    class Config:
        extra = Extra.forbid


class OntologyChoice(str, Enum):
    biolink = "biolink"
    chemprot = "chemprot"


class RetrievalMethod(str, Enum):
    knn = "sklearn_knn"
    scipy = "scipy_cosine"


class Candidate(BaseModel):
    mapped_predicate: str
    score: float


class PredicateChoice(BaseModel):
    predicate: str
    object_aspect_qualifier: Optional[str] = ""
    object_direction_qualifier: Optional[str] = ""
    negated: bool = False
    selector: str


class PredicateResult(BaseModel):
    subject: str
    object: str
    relationship: str
    top_choice: PredicateChoice
    Top_n_candidates: Dict[int, Candidate]
    Top_n_retrieval_method: str


class QueryResponse(BaseModel):
    results: List[PredicateResult]
    ontology: str


@APP.get("/ontologies",
         summary="List available ontologies",
         tags=["Configuration"])
async def get_ontologies():
    """List available ontologies with details"""
    return {
        "available": list_ontologies(),
        "current": get_current_config().name,
        "details": get_ontology_details()
    }


@APP.post("/query/",
          summary="Get a standard predicate for a subject-object pair",
          description="Uses a similarity search to determine the top-n biolink predicates for each triple then re-ranks to select the best",
          tags=["Relation Extraction"],
          response_model=QueryResponse)
async def query_predicate(
        triples: List[InputTriples],
        ontology: OntologyChoice = Query(
            default=OntologyChoice.biolink,
            description="Ontology to use for predicate mapping"
        ),
        similarity_based_retrieval_method: RetrievalMethod = Query(
            default=RetrievalMethod.knn,
            description="Similarity search method for candidate retrieval"
        ),
        use_sapbert: bool = Query(
            default=True,
            description="Enable SapBERT predictions. Combines vector database results with SapBERT embeddings to enhance predicate mapping."
        )
):
    try:
        # Switch ontology if specified
        original_ontology = get_current_config().name
        if ontology.value != original_ontology:
            set_ontology(ontology.value)

        config = get_current_config()
        logger.info(f"Processing {len(triples)} triples with {config.name}")

        input_data = [triple.model_dump() for triple in triples]

        if similarity_based_retrieval_method.value == "sklearn_knn":
            results = await run_query(input_data, is_knn=True, use_sapbert=use_sapbert)
        else:
            results = await run_query(input_data, use_sapbert=use_sapbert)

        current_ontology = get_current_config().name

        # Switch back if we changed
        if ontology.value != original_ontology:
            set_ontology(original_ontology)

        return {"results": results, "ontology": current_ontology}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Configuration error: {str(e)}")
    except Exception as e:
        logger.error(f"Error: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


async def run_query(triple_input: list, is_knn=False, use_sapbert=True):
    """Execute predicate mapping query using current ontology"""
    try:
        config = get_current_config()

        predicate_client = pl.PredicateClient()
        db = pl.PredicateDatabase(client=predicate_client, is_knn=is_knn)

        logger.info("Loading database")
        db.load_db_from_json(config.embedding_file)

        data = pl.parse_new_llm_response(triple_input)
        relationships = await pl.lookup_unique_predicates(data, db, use_sapbert=use_sapbert)

        logger.info("Loading descriptions and reranking")
        predicate_descriptions = load_from_json(config.description_file)
        qualified_predicate = {}
        relationships = pl.relationship_queries_to_batch(
            relationships, predicate_descriptions, db.is_vdb, db.is_knn
        )

        logger.info("LLM reranking")
        output_triples = await predicate_client.rerank_relationship_choices(
            relationships, qualified_predicate, db.is_vdb, db.is_knn
        )

        successful = sum(1 for r in output_triples if r.get('top_choice', {}).get('predicate') != ' ')
        failed = len(output_triples) - successful

        if failed > 0:
            logger.warning(f"{failed} failed predictions out of {len(output_triples)}")

        return output_triples

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise RuntimeError(f"Query processing failed: {str(e)}")
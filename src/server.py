from enum import Enum
import logging
import traceback
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Extra, Field
from typing import List, Dict, Optional
from src import biolink_predicate_lookup as blp
from src.utils import load_from_json
from src.config import get_current_config, list_ontologies, get_ontology_details
from src.Preprocessing.clean_mappings import cull_mapped_predicates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.ERROR)

APP = FastAPI(title="Biolink Predicate Mapper")


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


class HEALpacaInput(BaseModel):
    abstract: str = Field(..., example=(
        "The present study was designed to investigate the cardioprotective effects of betaine on acute myocardial ischemia induced experimentally in rats focusing on regulation of signal transducer and activator of transcription 3 (STAT3) and apoptotic pathways as the potential mechanism underlying the drug effect. "
        "Male Sprague Dawley rats were treated with betaine (100, 200, and 400 mg/kg) orally for 40 days. Acute myocardial ischemic injury was induced in rats by subcutaneous injection of isoproterenol (85 mg/kg), for two consecutive days. Serum cardiac marker enzyme, histopathological variables and expression of protein levels were analyzed. "
        "Oral administration of betaine (200 and 400 mg/kg) significantly reduced the level of cardiac marker enzyme in the serum and prevented left ventricular remodeling. Western blot analysis showed that isoproterenol-induced phosphorylation of STAT3 was maintained or further enhanced by betaine treatment in myocardium. "
        "Furthermore, betaine (200 and 400 mg/kg) treatment increased the ventricular expression of Bcl-2 and reduced the level of Bax, therefore causing a significant increase in the ratio of Bcl-2/Bax. "
        "The protective role of betaine on myocardial damage was further confirmed by histopathological examination. In summary, our results showed that betaine pretreatment attenuated isoproterenol-induced acute myocardial ischemia via the regulation of STAT3 and apoptotic pathways."
    ))
    subject: str = Field(..., example="Betaine")
    object: str = Field(..., example="Bcl-2")
    relationship: str = Field(..., example="increases expression of")

    class Config:
        extra = Extra.forbid


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


class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None
    partial_results: Optional[List[PredicateResult]] = None


@APP.get("/biolink",
         summary="Biolink Schema Details",
         tags=["Configuration"])
async def get_ontologies():
    """ontologies details"""
    return {
        "details": get_ontology_details()
    }


@APP.post("/query/",
          summary="Get a standard predicate for a subject-object pair",
          description="Uses a similarity search to determine the top-n biolink predicates for each triple then re-ranks to select the best",
          tags=["Relation Extraction"],
          response_model=QueryResponse
          )
async def query_predicate(
        triples: List[HEALpacaInput],
        retrieval_method: RetrievalMethod = Query(
            default=RetrievalMethod.knn
        )
):
    try:
        logger.info(f"Processing {len(triples)} triples with method {retrieval_method.value}")
        input_data = [triple.model_dump() for triple in triples]
        if retrieval_method.value == "sklearn_knn":
            results = await run_query(input_data, is_knn=True)
        else:
            results = await run_query(input_data)

        logger.info(f"Successfully processed {len(results)} results")
        return {"results": results}
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        raise HTTPException(
            status_code=503,
            detail="Service configuration error: required data files not found"
        )
    except TimeoutError as e:
        logger.error(f"Request timeout: {e}")
        raise HTTPException(
            status_code=504,
            detail="Request timeout: external service took too long to respond"
        )
    except ConnectionError as e:
        logger.error(f"External service connection failed: {e}")
        raise HTTPException(
            status_code=502,
            detail="External service temporarily unavailable"
        )
    except ValueError as e:
        logger.error(f"Invalid input data: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {str(e)}"
        )
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        raise HTTPException(
            status_code=502,
            detail="External service error occurred"
        )
    except Exception as e:
        logger.error(f"Unexpected error in query_predicate: {type(e).__name__}: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while processing request"
        )


async def run_query(triple_input: list, is_knn=False):
    """
        Executes predicate mapping query
    """
    try:
        config = get_current_config()

        predicate_client = blp.PredicateClient()
        db = blp.PredicateDatabase(client=predicate_client, is_knn=is_knn)

        logger.info("Loading and populating database")
        qualified_predicate = load_from_json(config.qualified_predicate_file)
        db.populate_db(cull_mapped_predicates(config.embedding_file, qualified_predicate))

        data = blp.parse_new_llm_response(triple_input)
        logger.info(f"Vector searching for {len(triple_input)} relationships")
        relationships = await blp.lookup_unique_predicates(data, db)

        logger.info("Loading predicate descriptions and reranking candidates")
        predicate_descriptions = load_from_json(config.description_file)
        relationships = blp.relationship_queries_to_batch(relationships, predicate_descriptions, db.is_knn)

        logger.info("Performing LLM reranking of predicate choices")
        output_triples = await predicate_client.rerank_relationship_choices(relationships, qualified_predicate, db.is_knn)
        successful = sum(1 for r in output_triples if r.get('top_choice', {}).get('predicate') != ' ')
        failed = len(output_triples) - successful
        if failed > 0:
            logger.warning(
                f"Processing completed with {failed} failed predictions out of {len(output_triples)} total")
        return output_triples

    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e}")
        raise FileNotFoundError(f"Configuration file missing: {e}")

    except Exception as e:
        logger.error(f"Error in run_query: {type(e).__name__}: {str(e)}")
        raise RuntimeError(f"Query processing failed: {str(e)}")

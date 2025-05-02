import json
from enum import Enum
from pathlib import Path
import logging
import traceback
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Extra, Field
from typing import List, Dict, Optional
from src import biolink_predicate_lookup as blp

APP = FastAPI()


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
    nn = "nearest_neighbor"
    sim = "cosine_similarities"
    vectordb = "vectordb"


class Candidate(BaseModel):
    mapped_predicate: str
    score: float


class PredicateChoice(BaseModel):
    predicate: str
    object_aspect_qualifier: Optional[str] = ""
    object_direction_qualifier: Optional[str] = ""
    selector: str


class PredicateResult(BaseModel):
    subject: str
    object: str
    relationship: str
    abstract: str
    top_choice: PredicateChoice
    Top_n_candidates: Dict[int, Candidate]
    Top_n_retrieval_method: str  # e.g., "nearest_neighbors"


class QueryResponse(BaseModel):
    results: List[PredicateResult]


BASE_DIR = Path(__file__)
BASE_DIR = Path(__file__).resolve().parent
DESCRIPTION_FILE = BASE_DIR.parent / "data" / "short_description.json"
EMBEDDING_FILE = BASE_DIR.parent / "data" / "all_biolink_mapped_vectors.json"
QUALIFIED_PREDICATE_FILE = BASE_DIR.parent / "data" / "qualified_predicate_mapping.json"


# "RENCI Relationship Extraction Pipeline"

@APP.post("/query/",
          summary="Get a standard predicate for a subject-object pair",
          description="Uses a similarity search to determine the top-n biolink predicates for each triple then re-ranks to select the best",
          tags=["Relation Extraction"],
          response_model=QueryResponse
          )
async def query_predicate(
        triples: List[HEALpacaInput],
        retrieval_method: RetrievalMethod = Query(
            default=RetrievalMethod.sim,
            include_in_schema=False
        )
):
    try:
        input_data = [triple.model_dump() for triple in triples]
        if retrieval_method.value == "vectordb":
            results = await run_query(input_data, QUALIFIED_PREDICATE_FILE, DESCRIPTION_FILE, EMBEDDING_FILE,
                                      is_vdb=True, is_nn=False)
        elif retrieval_method.value == "nearest_neighbor":
            results = await run_query(input_data, QUALIFIED_PREDICATE_FILE, DESCRIPTION_FILE, EMBEDDING_FILE,
                                      is_vdb=False, is_nn=True)
        else:
            results = await run_query(input_data, QUALIFIED_PREDICATE_FILE, DESCRIPTION_FILE, EMBEDDING_FILE)
        return {"results": results}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


async def run_query(triple_input: list, qualifiedPredicate_file: str, description_file: str, embedding_file: str,
                     is_vdb=False, is_nn=False):
    llm = blp.PredicateClient()
    with open(embedding_file, "r") as f:
        predicate_embedding = json.load(f)
    logging.info(f"Initializing the DB with {len(predicate_embedding)} predicate embeddings.... ")
    db = blp.PredicateDatabase(client=llm, is_vdb=is_vdb, is_nn=is_nn)
    db.populate_db(predicate_embedding)

    data = blp.parse_new_llm_response(triple_input)
    logging.info(f"Vector Searching {len(triple_input)} Data.... ")
    relationships = await blp.lookup_unique_predicates(data, db)

    logging.info(f"Reranking and Selecting top predicate choice .... ")
    with open(description_file, "r") as f:
        predicate_descriptions = json.load(f)
    with open(qualifiedPredicate_file, "r") as f:
        qualified_predicate = json.load(f)
    relationships = blp.relationship_queries_to_batch(relationships, predicate_descriptions, db.is_vdb, db.is_nn)
    output_triples = await llm.check_relationship(relationships, qualified_predicate, db.is_vdb, db.is_nn)
    return output_triples

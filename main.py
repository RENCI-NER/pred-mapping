import json
from enum import Enum
from pathlib import Path
import logging
import traceback
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Extra, Field
from typing import List
from src import biolink_predicate_lookup as blp

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HEALpacaInput(BaseModel):
    abstract: str = Field(..., example=(
        "Despite increasing reports on nonionic contrast media-induced nephropathy (CIN) in hospitalized adult patients during cardiac procedures, the studies in pediatrics are limited, with even less focus on possible predisposing factors and preventive measures for patients undergoing cardiac angiography. "
        "This prospective study determined the incidence of CIN for two nonionic contrast media (CM), iopromide and iohexol, among 80 patients younger than 18 years and compared the rates for this complication in relation to the type and dosage of CM and the presence of cyanosis. "
        "The 80 patients in the study consecutively received either iopromide (group A, n = 40) or iohexol (group B, n = 40). Serum sodium (Na), potassium (K), and creatinine (Cr) were measured 24 h before angiography as baseline values, then measured again at 12-, 24-, and 48-h intervals after CM use. Urine samples for Na and Cr also were checked at the same intervals. "
        "Risk of renal failure, Injury to the kidney, Failure of kidney function, Loss of kidney function, and End-stage renal damage (RIFLE criteria) were used to define CIN and its incidence in the study population. Accordingly, among the 15 CIN patients (18.75%), 7.5% of the patients in group A had increased risk and 3.75% had renal injury, whereas 5% of group B had increased risk and 2.5% had renal injury. "
        "Whereas 33.3% of the patients with CIN were among those who received the proper dosage of CM, the percentage increased to 66.6% among those who received larger doses, with a significant difference in the incidence of CIN related to the different dosages of CM (p = 0.014). "
        "Among the 15 patients with CIN, 6 had cyanotic congenital heart diseases, but the incidence did not differ significantly from that for the noncyanotic patients (p = 0.243). Although clinically silent, CIN is not rare in pediatrics. "
        "The incidence depends on dosage but not on the type of consumed nonionic CM, nor on the presence of cyanosis, and although CIN usually is reversible, more concern is needed for the prevention of such a complication in children."
    ))
    subject: str = Field(..., example="Asenapine")
    object: str = Field(..., example="Schizophrenia")
    relationship: str = Field(..., example="treats")

    class Config:
        extra = Extra.forbid


class RetrievalMethod(str, Enum):
    nn = "nearest_neighbor"
    sim = "cosine_similarities"
    vectordb = "vectordb"

BASE_DIR = Path(__file__).resolve().parent
DESCRIPTION_FILE = BASE_DIR / "data" / "short_description.json"
EMBEDDING_FILE = BASE_DIR / "data" / "all_biolink_mapped_vectors.json"


@app.post("/query/")
async def query_predicate(
        triples: List[HEALpacaInput],
        retrieval_method: RetrievalMethod
    ):
    try:
        input_data = [triple.dict() for triple in triples]

        if retrieval_method.value == "vectordb":
            results = run_query(input_data, DESCRIPTION_FILE, EMBEDDING_FILE, is_vdb=True, is_nn=False)
        elif retrieval_method.value == "nearest_neighbor":
            results = run_query(input_data, DESCRIPTION_FILE, EMBEDDING_FILE, is_vdb=False, is_nn=True)
        else:
            results = run_query(input_data, DESCRIPTION_FILE, EMBEDDING_FILE)

        return {"results": results}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def run_query(triple_input: list, description_file: str, embedding_file: str, is_vdb=False, is_nn=False):
    llm = blp.PredicateClient()
    with open(embedding_file, "r") as f:
        predicate_embedding = json.load(f)
    logging.info(f"Initializing the DB with {len(predicate_embedding)} predicate embeddings.... ")
    db = blp.PredicateDatabase(client=llm, is_vdb=is_vdb, is_nn=is_nn)
    db.populate_db(predicate_embedding)

    data = blp.parse_new_llm_response(triple_input)
    logging.info(f"Vector Searching {len(triple_input)} Data.... ")
    relationships = blp.lookup_unique_predicates(data, db)

    logging.info(f"Reranking and Selecting top predicate choice .... ")
    with open(description_file, "r") as f:
        predicate_descriptions = json.load(f)
    relationships = blp.relationship_queries_to_batch(relationships, predicate_descriptions, db.is_vdb, db.is_nn)
    output_triples = llm.check_relationship(relationships, db.is_vdb, db.is_nn)
    return output_triples


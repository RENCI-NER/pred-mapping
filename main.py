import os
import json
import logging
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Extra, Field
from typing import List
import biolink_predicate_lookup as blp

logging.basicConfig(level=logging.INFO)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HEALpacaInput(BaseModel):
    subject: str = Field(..., example="mutation")
    object: str = Field(..., example="ALD gene")
    abstract: str = Field(..., example=(
        "Fragments of the adrenoleukodystrophy (ALD) cDNA from a patient with adolescent ALD "
        "were amplified by polymerase chain reaction and subcloned. Bidirectional sequencing of the "
        "entire coding ALD gene disclosed a cytosine to guanine transversion at nucleotide 1451 in exon five, "
        "resulting in substitution of proline 484 by arginine. Five of nine siblings of the patient, comprising "
        "two cerebral ALD, one adrenomyeloneuropathy, one Addison only as well as the symptomatic mother "
        "(all accumulating very long chain fatty acids) carried this mutation, which was not found in the "
        "unaffected persons, in five unrelated ALD patients, and in twenty controls. We propose that this missense "
        "mutation generated the disease per se as well as the metabolic defect; the different phenotypes, however, "
        "must have originated by means of additional pathogenetic factors."
    ))
    relationship: str = Field(..., example="found in")

    class Config:
        extra = Extra.forbid



# Constants for precomputed files
DESCRIPTION_FILE = os.path.join("data", "short_description.json")
VECTOR_DB_FILE = os.path.join("data", "all_biolink_mapped_vectors.json")


@app.post("/query/")
async def query_predicate(triples: List[HEALpacaInput]):
    try:
        input_data = [triple.dict() for triple in triples]
        results = run_query(input_data, DESCRIPTION_FILE, VECTOR_DB_FILE)
        return {"results": results}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def run_query(triple_input: list, description_file: str, vector_db_file: str):
    llm = blp.PredicateClient()
    with open(vector_db_file, "r") as f:
        predicate_embedding = json.load(f)
    logging.info(f"Initializing the DB with {len(predicate_embedding)} predicate embeddings.... ")
    db = blp.PredicateDatabase(client=llm)
    db.populate_db(predicate_embedding)

    data = blp.parse_new_llm_response(triple_input)
    logging.info(f"Vector Searching {len(triple_input)} Data.... ")
    relationships = blp.lookup_unique_predicates(data, db)

    logging.info(f"Reranking and Selecting top predicate choice .... ")
    with open(description_file, "r") as f:
        predicate_descriptions = json.load(f)
    relationships = blp.relationship_queries_to_batch(relationships, predicate_descriptions)
    output_triples = llm.check_relationship(relationships)
    return output_triples

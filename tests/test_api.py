# export PYTHONPATH="$PYTHONPATH:$PWD"

import os
import pytest
import json
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.server import APP, RetrievalMethod
client = TestClient(APP)


@pytest.fixture(scope="session")
def is_ci_env():
    return os.environ.get("CI", "false").lower() == "true"


def test_query_endpoint(is_ci_env):
    test_payload = [
        {
            "abstract": (
                "Despite increasing reports on nonionic contrast media-induced nephropathy (CIN) in hospitalized adult patients during cardiac procedures, the studies in pediatrics are limited, with even less focus on possible predisposing factors and preventive measures for patients undergoing cardiac angiography. "
                "This prospective study determined the incidence of CIN for two nonionic contrast media (CM), iopromide and iohexol, among 80 patients younger than 18 years and compared the rates for this complication in relation to the type and dosage of CM and the presence of cyanosis. "
                "The 80 patients in the study consecutively received either iopromide (group A, n = 40) or iohexol (group B, n = 40). Serum sodium (Na), potassium (K), and creatinine (Cr) were measured 24 h before angiography as baseline values, then measured again at 12-, 24-, and 48-h intervals after CM use. Urine samples for Na and Cr also were checked at the same intervals. "
                "Risk of renal failure, Injury to the kidney, Failure of kidney function, Loss of kidney function, and End-stage renal damage (RIFLE criteria) were used to define CIN and its incidence in the study population. Accordingly, among the 15 CIN patients (18.75%), 7.5% of the patients in group A had increased risk and 3.75% had renal injury, whereas 5% of group B had increased risk and 2.5% had renal injury. "
                "Whereas 33.3% of the patients with CIN were among those who received the proper dosage of CM, the percentage increased to 66.6% among those who received larger doses, with a significant difference in the incidence of CIN related to the different dosages of CM (p = 0.014). "
                "Among the 15 patients with CIN, 6 had cyanotic congenital heart diseases, but the incidence did not differ significantly from that for the noncyanotic patients (p = 0.243). Although clinically silent, CIN is not rare in pediatrics. "
                "The incidence depends on dosage but not on the type of consumed nonionic CM, nor on the presence of cyanosis, and although CIN usually is reversible, more concern is needed for the prevention of such a complication in children."
            ),
            "subject": "Asenapine",
            "object": "Schizophrenia",
            "relationship": "treats"
        }
    ]
    if is_ci_env:
        with patch("src.biolink_predicate_lookup.PredicateClient.get_chat_completion") as mock_chat, \
                patch("src.biolink_predicate_lookup.PredicateClient.get_embedding") as mock_embed:

            mock_embed.return_value = [0.1] * 768
            mock_chat.return_value = '{"mapped_predicate": "biolink:treats"}'

            response = client.post("/query/", json=test_payload, params={"retrieval_method": RetrievalMethod.sim.value})
    else:
        DIR = os.path.dirname(os.path.abspath(__file__))
        with open(f"{DIR}/sample_input.json") as f:
            test_payload = json.load(f)
        response = client.post("/query/", json=test_payload, params={"retrieval_method": RetrievalMethod.vectordb.value})

    assert response.status_code == 200
    data = response.json()
    with open(f"{DIR}/sample_output.json", "w") as f:
        json.dump(data, f, indent=4)

    assert "results" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) == len(test_payload)
    assert "top_choice" in data["results"][0]



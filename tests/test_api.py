# export PYTHONPATH="$PYTHONPATH:$PWD"
import os
import json
import pytest
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
            "subject": "Ifenprodil",
            "relationship": "inhibits",
            "object": "N-methyl-D-aspartate receptors",
            "abstract": ("Effect of ifenprodil on GluN1/GluN2B N-methyl-D-aspartate receptor gating."
                         "Ifenprodil is an allosteric inhibitor of GluN1/GluN2B N-methyl-D-aspartate receptors. "
                         "Despite its widespread use as a prototype for drug development and a subtype-selective tool for physiologic experiments, its precise effect on GluN1/GluN2B gating is yet to be fully understood. "
                         "Interestingly, recent crystallographic evidence identified that ifenprodil, unlike zinc, binds at the interface of the GluN1/GluN2B amino terminal domain dimer by an induced-fit mechanism. "
                         "To delineate the effect of this unique binding on GluN1/GluN2B receptor gating, we recorded steady-state currents from cell-attached and outside-out patches. "
                         "At pH 7.9 in cell-attached patches, ifenprodil increased the occupancy of the long-lived shut conformations, thereby reducing the open probability of the receptor with no change in the mean open time. "
                         "In addition, ifenprodil selectively affected the area of shut time constants, but not the time constants themselves. Kinetic analyses suggested that ifenprodil prevents the transition of the receptor to an open state and increases its dwell time in an intrinsically occurring closed conformation or desensitized state. "
                         "We found distinct differences in the action of ifenprodil at GluN1/GluN2B in comparison with previous studies on the effect of zinc on GluN1/GluN2A gating, which may arise due to their unique binding sites. Our data also uncover the potential pH-dependent action of ifenprodil on gating. "
                         "At a low pH (pH 7.4), but not pH 7.9, ifenprodil reduces the mean open time of GluN1/GluN2B receptors, which may be responsible for its usefulness as a context-dependent inhibitor in conditions like ischemia and stroke, when the pH of the extracellular milieu becomes acidic.")
        }
    ]
    if is_ci_env:
        with patch("src.biolink_predicate_lookup.PredicateClient.get_chat_completion") as mock_chat, \
                patch("src.biolink_predicate_lookup.PredicateClient.get_embedding") as mock_embed:

            mock_embed.return_value = [0.1] * 768
            mock_chat.return_value = '{"mapped_predicate": "biolink:treats"}'

            response = client.post("/query/", json=test_payload,
                                   params={"retrieval_method": RetrievalMethod.scipy.value})
    else:
        # DIR = os.path.dirname(os.path.abspath(__file__))
        # with open(f"{DIR}/all.json") as f:
        #     test_payload = json.load(f)
        # test_payload = [test for test in test_payload if "induces" in test["relationship"]][:3]
        response = client.post("/query/", json=test_payload, params={"retrieval_method": RetrievalMethod.knn.value})

    assert response.status_code == 200
    data = response.json()

    # with open(f"{DIR}/induces_results.json", "w") as f:
    #     json.dump(data, f, indent=4)

    assert "results" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) == len(test_payload)
    assert "top_choice" in data["results"][0]

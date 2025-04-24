# export PYTHONPATH="$PYTHONPATH:$PWD"

import os
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from main import app, RetrievalMethod
from src.biolink_predicate_lookup import extract_mapped_predicate, relationship_queries_to_batch

client = TestClient(app)


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

    test_payload1 = [
        {
            "subject": "Betaine",
            "object": "Bcl-2",
            "relationship": "increases expression of",
            "abstract": (
                "The present study was designed to investigate the cardioprotective effects of betaine on acute myocardial ischemia induced experimentally in rats focusing on regulation of signal transducer and activator of transcription 3 (STAT3) and apoptotic pathways as the potential mechanism underlying the drug effect. "
                 "Male Sprague Dawley rats were treated with betaine (100, 200, and 400 mg/kg) orally for 40 days. Acute myocardial ischemic injury was induced in rats by subcutaneous injection of isoproterenol (85 mg/kg), for two consecutive days. Serum cardiac marker enzyme, histopathological variables and expression of protein levels were analyzed. "
                 "Oral administration of betaine (200 and 400 mg/kg) significantly reduced the level of cardiac marker enzyme in the serum and prevented left ventricular remodeling. Western blot analysis showed that isoproterenol-induced phosphorylation of STAT3 was maintained or further enhanced by betaine treatment in myocardium. "
                 "Furthermore, betaine (200 and 400 mg/kg) treatment increased the ventricular expression of Bcl-2 and reduced the level of Bax, therefore causing a significant increase in the ratio of Bcl-2/Bax. "
                 "The protective role of betaine on myocardial damage was further confirmed by histopathological examination. In summary, our results showed that betaine pretreatment attenuated isoproterenol-induced acute myocardial ischemia via the regulation of STAT3 and apoptotic pathways."
            )
        },

        {
            "subject": "Cocaine",
            "object": "Cocaine use disorder",
            "relationship": "potential target for therapeutics",
            "abstract": (
                "Drug-related attentional bias may have significant implications for the treatment of cocaine use disorder (CocUD). However, the neurobiology of attentional bias is not completely understood. "
                "This study employed dynamic causal modeling (DCM) to conduct an analysis of effective (directional) connectivity involved in drug-related attentional bias in treatment-seeking CocUD subjects. "
                "The DCM analysis was conducted based on functional magnetic resonance imaging (fMRI) data acquired from fifteen CocUD subjects while performing a cocaine-word Stroop task, during which blocks of Cocaine Words (CW) and Neutral Words (NW) alternated. "
                "There was no significant attentional bias at group level. Although no significant brain activation was found, the DCM analysis found that, relative to the NW, the CW caused a significant increase in the strength of the right (R) anterior cingulate cortex (ACC) to R hippocampus effective connectivity. "
                "Greater increase of this connectivity was associated with greater CW reaction time (relative to NW reaction time). The increased strength of R ACC to R hippocampus connectivity may reflect ACC activation of hippocampal memories related to drug use, which was triggered by the drug cues. "
                "This circuit could be a potential target for therapeutics in CocUD patients. No significant change was found in the other modeled connectivities."
            )
        }

    ]

    if is_ci_env:
        with patch("src.biolink_predicate_lookup.PredicateClient.get_chat_completion") as mock_chat, \
                patch("src.biolink_predicate_lookup.PredicateClient.get_embedding") as mock_embed:

            mock_embed.return_value = [0.1] * 768
            mock_chat.return_value = '{"mapped_predicate": "biolink:treats"}'

            response = client.post("/query/", json=test_payload, params={"retrieval_method": RetrievalMethod.sim.value})
    else:
        response = client.post("/query/", json=test_payload1, params={"retrieval_method": RetrievalMethod.nn.value})

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) == len(test_payload1)
    assert "top_choice" in data["results"][0]


def test_extract_valid_json_mapping():
    response = '{"mapped_predicate": "treats"}'
    choices = {"treats": "used to treat", "prevents": "used to prevent"}
    result = extract_mapped_predicate(response, choices)
    assert result == "biolink:treats"


def test_extract_loose_format():
    response = "mapped_predicate: 'prevents'"
    choices = {"treats": "used to treat", "prevents": "used to prevent"}
    result = extract_mapped_predicate(response, choices)
    assert result == "biolink:prevents"


def test_extract_no_match_returns_none():
    response = '{"mapped_predicate": "invalid_pred"}'
    choices = {"treats": "used to treat"}
    result = extract_mapped_predicate(response, choices)
    assert result is None


def test_relationship_queries_to_batch():
    query_results = [{
        "subject": "Aspirin",
        "object": "Headache",
        "relationship": "treats",
        "abstract": "Aspirin is used to treat headaches.",
        "Top_n_candidates": {
            "treats": 0.9,
            "relieves": 0.7
        }
    }]
    descriptions = {"treats": "used to treat", "relieves": "alleviates symptom"}
    batch = relationship_queries_to_batch(query_results, descriptions, is_vdb=True, is_nn=False)

    assert batch[0]["Top_n_retrieval_method"] == "vectorDb"
    assert "predicate_choices" in batch[0]
    assert isinstance(batch[0]["Top_n_candidates"], dict)
    assert batch[0]["predicate_choices"]["treats"] == "used to treat"

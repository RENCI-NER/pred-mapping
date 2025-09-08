def fetch_chemprot_data(self, output_file=None):
    import pandas as pd
    # Dataset splits
    splits = {
        'sample': 'chemprot_full_source/sample-00000-of-00001.parquet',
        'train': 'chemprot_full_source/train-00000-of-00001.parquet',
        'test': 'chemprot_full_source/test-00000-of-00001.parquet',
        'validation': 'chemprot_full_source/validation-00000-of-00001.parquet'
    }

    # Load validation set
    df_validation = pd.read_parquet("hf://datasets/bigbio/chemprot/" + splits["validation"])

    # Mapping CPR codes to categories
    cpr2label = {
        "CPR:0": "Other",
        "CPR:1": "Part_of",
        "CPR:2": "Regulator",
        "CPR:3": "Upregulator",
        "CPR:4": "Downregulator",
        "CPR:5": "Agonist",
        "CPR:6": "Antagonist",
        "CPR:7": "Modulator",
        "CPR:8": "Cofactor",
        "CPR:9": "Substrate",
        "CPR:10": "Not_relation"
    }

    triples = []

    # Group by abstract (PMID)
    for pmid, group in df_validation.groupby("pmid"):
        text = group["text"].iloc[0]

        # Build entity dictionary (id -> text)
        entities = {}
        for ids, texts in zip(group["entities.id"], group["entities.text"]):
            for e_id, e_text in zip(ids, texts):
                entities[e_id] = e_text

        # Iterate over relations
        for rel_types, arg1_list, arg2_list in zip(
                group["relations.type"], group["relations.arg1"], group["relations.arg2"]
        ):
            for rel_code, subj_id, obj_id in zip(rel_types, arg1_list, arg2_list):
                relation = cpr2label.get(rel_code, rel_code)  # fallback: keep raw code

                if subj_id not in entities or obj_id not in entities:
                    continue

                subject = entities[subj_id]
                object_ = entities[obj_id]

                # Skip trivial 1-char entities
                if len(subject) <= 1 or len(object_) <= 1:
                    continue

                triples.append({
                    "subject": subject,
                    "object": object_,
                    "relationCategory": relation,
                    "abstract": text
                })

    print(f"Extracted {len(triples)} triples")

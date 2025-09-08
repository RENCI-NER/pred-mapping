from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os
import logging
from scipy.spatial.distance import cdist

LOGGER = logging.getLogger(__name__)


def sapbert_predict(model_folder, all_names, use_gpu=False):
    tokenizer = AutoTokenizer.from_pretrained(model_folder)
    if use_gpu:
        model = AutoModel.from_pretrained(model_folder).cuda(0)
    else:
        model = AutoModel.from_pretrained(model_folder)

    bs = 128
    all_reps = []
    for i in tqdm(np.arange(0, len(all_names), bs)):
        toks = tokenizer.batch_encode_plus(all_names[i:i + bs],
                                           padding="max_length",
                                           max_length=25,
                                           truncation=True,
                                           return_tensors="pt")
        if use_gpu:
            toks_cuda = {}
            for k, v in toks.items():
                toks_cuda[k] = v.cuda(0)
            output = model(**toks_cuda)
        else:
            output = model(**toks)
        cls_rep = output[0][:, 0, :]
        all_reps.append(cls_rep.cpu().detach().numpy())
        LOGGER.info(f'in sapbert_predict, batches done: {i + 1}')
    all_reps_emb = np.concatenate(all_reps, axis=0)
    return all_reps_emb


def sapbert_score(query_emb, all_rels_emb, all_rel_ids, all_rel_names, metric="similarity"):
    dist = cdist(query_emb.T, all_rels_emb, "cosine")
    if metric == "similarity":
        dist = 1 - dist  # convert to similarity
        nn_indices = np.argsort(-dist[0])[:10]
    else:
        nn_indices = np.argsort(-dist)[:10]
    topk_results = [
        {
            "label": all_rel_ids[i],
            "mention": all_rel_names[i],
            metric: float(dist[0][i])
        }
        for i in nn_indices
    ]

    return topk_results


def sapbert_score_batch(query_embs, all_pred_embs, all_pred, all_pred_texts, top_k=10):
    distances = cdist(query_embs, all_pred_embs, "cosine")
    similarities = 1 - distances
    top_indices = np.argsort(-similarities, axis=1)[:, :top_k]
    return [
        [
            {
                "text": all_pred_texts[i],
                "mapped_predicate": all_pred[i],
                "score": round(float(similarities[q_idx][i]), 3)
            }
            for i in top_k_indices
        ]
        for q_idx, top_k_indices in enumerate(top_indices)
    ]



def load_labels(path):
    with open(path, 'r') as f:
        return f.read().splitlines()


def get_labels(biolink_pred_path=None, output_dict_path=None):
    if os.path.exists(output_dict_path):
        dictionary_entries = load_labels(output_dict_path)
        biolink_rels = [rel.split("||") for rel in dictionary_entries]
        all_rels = [r[1] for r in biolink_rels]
        all_rels_id = [r[0] for r in biolink_rels]
        return all_rels, all_rels_id

    input_file = biolink_pred_path
    dictionary_entries = set()

    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            parts = [p.strip() for p in line.strip().split("||")]
            if len(parts) != 3:
                print(f"[Line {i}] Skipping malformed line: {line.strip()}")
                continue

            CUI, head, tail = parts

            if len(head) < 2 or len(tail) < 2:
                print(f"[Line {i}] Skipping short entry: head='{head}', tail='{tail}'")
                continue
            if not head.isprintable() or not tail.isprintable():
                print(f"[Line {i}] Skipping non-printable: head='{head}', tail='{tail}'")
                continue

            dictionary_entries.update([f"{CUI}||{head}", f"{CUI}||{tail}"])

    with open(output_dict_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(sorted(dictionary_entries)))

    print(f"Created {output_dict_path} with {len(dictionary_entries)} entries.")
    print(f"Source file: {input_file}")

    return dictionary_entries



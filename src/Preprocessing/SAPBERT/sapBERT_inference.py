import argparse
import numpy as np
import time
from utils import sapbert_predict, get_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--MODEL_FOLDER', type=str, default="../../fresh_output_all_chemprot_mappings/re_model/",
                        help='SapBERT model trained from biolink predicate data')
    parser.add_argument('--OUTPUT_FILE', type=str, default="../../fresh_output_all_chemprot_mappings/inference_outputs/chemprot_mapping_embedding.npy",
                        help='SapBERT model inference output file for biolink input file')

    args = parser.parse_args()
    MODEL_FOLDER = args.MODEL_FOLDER
    OUTPUT_FILE = args.OUTPUT_FILE

    BIOLINK_RELS = get_labels()
    biolink_rels = [rel.split("||") for rel in BIOLINK_RELS]
    all_rels = [r[1] for r in biolink_rels]
    all_rels_id = [r[0] for r in biolink_rels]

    print("Embedding dictionary terms with SapBERT...")
    start = time.time()
    all_rels_emb = sapbert_predict(MODEL_FOLDER, all_rels, use_gpu=False)
    end = time.time()
    print(f"Time consumed in SapBERT prediction: {end - start:.2f} seconds")

    np.save(OUTPUT_FILE, all_rels_emb)
    print(f"Embeddings saved to {OUTPUT_FILE}")
### 2. SapBERT Training (Optional)

For enhanced specificity, train SapBERT models specific to ontology:

**Setup SapBERT:**
```bash
# Clone the SapBERT repository
git clone https://github.com/cambridgeltl/SapBERT.git
cd SapBERT

# Install dependencies
pip install -r requirements.txt
```

**Prepare training data for ChemProt:**
```bash
# Create training data from mappings file. This outputs all_chemprot_mappings.txt 
python sapBERTprepare_training_data.py \
  -m src/Preprocessing/all_mappings_file.json \
  -o training_data/chemprot/ \
  --ontology chemprot
```

**Train SapBERT model for ChemProt:**
```bash
python train.py \
  --model_dir "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
  --train_dir "../training_data/chemprot/all_chemprot_mappings.txt" \
  --output_dir "../../Chemprot_SapBert/model/" \
  --no_cuda \
  --epoch 10 \
  --train_batch_size 256 \
  --learning_rate 2e-5 \
  --max_length 25 \
  --checkpoint_step 999999 \
  --parallel \
  --amp \
  --pairwise \
  --random_seed 33 \
  --loss ms_loss \
  --use_miner \
  --type_of_triplets "all" \
  --miner_margin 0.2 \
  --agg_mode "cls"
```

**Generate SapBERT embeddings for inference:**
```bash
python sapbert_inference.py \
  --MODEL_FOLDER "Chemprot_SapBert/model/" \
  --OUTPUT_FILE "Chemprot_SapBert/data/embedding_mappings.npy"
```
Generates embeddings for all ontology predicates using the trained SapBERT model. These embeddings are used during API inference for enhanced predicate matching.

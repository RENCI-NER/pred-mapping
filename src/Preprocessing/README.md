# Preprocessing Pipeline: Input Relationship → Ontology Predicate Mapping

This preprocessing pipeline prepares ontology data for the predicate mapping service. It supports multiple ontologies including Biolink and ChemProt.

## General Workflow

### 1. Ontology Predicate Descriptions

**Collect predicate text and descriptors:**
```bash
collect_predicate_text.py [-m mappings_file -q qualified_mappings] [--ontology ONTOLOGY]
```
Scrapes the specified ontology and saves a JSON file to `mappings_file` with predicates as keys and text descriptors as values. Also collects mappings of predicates to qualifiers.

**Note:** This only applies to Biolink as ChemProt's semantic meanings are adapted from [PMC10215465](https://pmc.ncbi.nlm.nih.gov/articles/PMC10215465/table/bioengineering-10-00586-t001/) and additional definitions from Google search.

**Generate negations:**
```bash
get_negations.py [-m mappings_file -n negations_file] [--ontology ONTOLOGY]
```
Takes the mapping file and sends each descriptor to the LLM to produce negated versions. Saves results to `negations_file`.

**Merge and clean mappings:**
```bash
clean_mappings.py [-m mappings_file -n negations_file -a all_mappings_file] [--ontology ONTOLOGY]
```
Merges mapping and negations files, removes LLM "not enough information" responses or empty strings, and saves to `all_mappings_file`.

**Note:** If merging a newly generated mapping file with existing negations, ensure compatibility between versions.

**Embed predicates for similarity search:**
```bash
embed_biolink_mappings.py [-m mappings_file -e embeddings_file --lowercase] [--ontology ONTOLOGY]
```
Takes the mapping file and generates embeddings using the configured embedding model. Default embedding dimension is `768`.

### 2. SapBERT Training (Optional)

For enhanced accuracy, train SapBERT models specific to ontology:

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

## Ontology-Specific Configurations

### Biolink Model
- **Source:** [biolink-model](https://github.com/biolink/biolink-model)
- **Output files:**
  - `biolink_data/biolink_short_description.json`
  - `biolink_data/all_biolink_mapped_vectors.json`
  - `biolink_data/qualified_predicate_mappings.json`

### ChemProt
- **Source:** [BioCreative VI corpus](https://biocreative.bioinformatics.udel.edu/news/corpora/chemprot-corpus-biocreative-vi/)
- **Output files:**
  - `chemprot_data/chemprot_short_description.json`
  - `chemprot_data/all_chemprot_mapped_vectors.json`
  - `chemprot_data/qualified_predicate_mappings.json` (empty for ChemProt)

## Environment Setup

Configure environment for preprocessing:

```bash
# Set ontology
export ONTOLOGY=biolink  # or chemprot

# LLM configuration for negation generation
export LLM_API_URL=http://localhost:11434/api/generate
export CHAT_MODEL=alibayram/medgemma:latest
export MODEL_TEMPERATURE=0.5

# Embedding configuration
export EMBEDDING_URL=http://localhost:11434/api/embeddings
export EMBEDDING_MODEL=nomic-embed-text

# Use local models
export USE_LOCAL=true
```

## Output Structure

After preprocessing, the directory structure should contain:

```
{ontology}_data/
├── {ontology}_short_description.json      # Final cleaned mappings
├── all_{ontology}_mapped_vectors.json     # Embeddings for API
└── qualified_predicate_mappings.json      # Qualifier mappings

{Ontology}_SapBert/                         # Optional SapBERT models
├── data/
│   └── embedding_mappings.npy             # SapBERT embeddings
└── model/                                  # Trained model files
```

Where `{ontology}` is `biolink` or `chemprot`.

## Important Notes

- **Embedding Dimension:** Default is 768 for `nomic-embed-text`
- **Batch Processing:** Current implementation processes embeddings 25 input relationships per batch
- **Version Compatibility:** Ensure mapping and negation files are generated with compatible versions
- **Local LLM:** Uses Ollama by default for local processing
- **Cost Optimization:** Negations are generated once and reused to reduce LLM API costs

## Next Steps

After preprocessing, use the generated files with the main prediction service. For detailed API usage, see the main [README.md](../README.md).
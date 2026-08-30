# United0 Clinical NLP Pipeline

## Overview

This repository/notebook implements an end-to-end Vietnamese clinical natural language processing pipeline for extracting and normalizing medical concepts from free-form clinical text. The system is organized into five sequential phases:

1. **Clinical Named Entity Recognition (NER)** — detect medical concepts and preserve exact character offsets.
2. **Assertion Classification** — determine whether detected symptoms, diagnoses, and medications are negated, related to family history, or historical.
3. **ICD-10 Linking** — map Vietnamese diagnosis mentions to ICD-10 codes using multilingual dense retrieval.
4. **RxNorm Linking** — map medication mentions to RxNorm RxCUIs using lexical and dense retrieval.
5. **End-to-End Inference** — combine the four trained components and generate competition-style JSON outputs for all input `.txt` files.

The notebook is designed for **Google Colab + Google Drive** and uses the folder layout produced by the four training stages.

---

## System Architecture

```text
Raw Vietnamese Clinical Text
           |
           v
+------------------------------+
| Phase 1: Clinical NER        |
| ViHealthBERT                 |
| Token Classification         |
+------------------------------+
           |
           | detected entities + [start, end) offsets
           v
+------------------------------+
| Phase 2: Assertion Model     |
| ViHealthBERT                 |
| Multi-label Classification   |
+------------------------------+
           |
           | isNegated / isFamily / isHistorical
           v
        Entity Type
      /             \
     /               \
CHẨN_ĐOÁN           THUỐC
    |                  |
    v                  v
+---------------+   +-----------------------------+
| Phase 3       |   | Phase 4                     |
| ICD-10        |   | RxNorm                      |
| Multilingual  |   | Lexical + Multilingual E5  |
| E5 Retriever  |   | Hybrid Linker              |
+---------------+   +-----------------------------+
      \                 /
       \               /
        v             v
+-----------------------------------+
| Phase 5: End-to-End Inference     |
| Merge entities, assertions,       |
| ICD candidates, RxNorm candidates |
+-----------------------------------+
                 |
                 v
        output/1.json ... 100.json
                 |
                 v
              output.zip
```

---

# Phase 1 — Vietnamese Clinical NER

## Purpose

Phase 1 identifies five clinical entity types from raw Vietnamese medical text:

- `TRIỆU_CHỨNG` — symptom
- `TÊN_XÉT_NGHIỆM` — laboratory/test name
- `KẾT_QUẢ_XÉT_NGHIỆM` — laboratory/test result
- `CHẨN_ĐOÁN` — diagnosis
- `THUỐC` — medication

## Backbone

**Model:** `demdecuong/vihealthbert-base-syllable`

**Task head:** token classification with BIO labels.

```text
Raw text
   |
   v
Unicode surface pieces
   |
   v
Exact character offsets
   |
   v
ViHealthBERT tokenizer
   |
   v
ViHealthBERT encoder
   |
   v
Token classification head
   |
   v
BIO sequence
   |
   v
Entity spans + exact raw-text positions
```

The implementation deliberately preserves character-level offsets. Raw text is first split into Unicode surface pieces with exact character positions. BIO labels are attached to these pieces before subword tokenization, and piece labels are propagated to model subwords.

### Label structure

```text
O
B-TRIEU_CHUNG
I-TRIEU_CHUNG
B-TEN_XET_NGHIEM
I-TEN_XET_NGHIEM
B-KET_QUA_XET_NGHIEM
I-KET_QUA_XET_NGHIEM
B-CHAN_DOAN
I-CHAN_DOAN
B-THUOC
I-THUOC
```

### Position convention

All positions follow the Python slicing convention:

```text
[start, end)
```

Therefore:

```python
text[start:end] == entity_text
```

### Model selection

The best Phase-1 checkpoint is selected using exact-span macro entity F1 on `ner_valid_concept.jsonl`.

### Drive output

```text
/content/drive/MyDrive/output/
├── run_config.json
├── training_history.json
├── best/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   ├── bpe.codes
│   └── training_meta.json
└── last/
```

---

# Phase 2 — Clinical Assertion Classification

## Purpose

Phase 2 operates on entities already detected by Phase 1. It is applied only to:

- `TRIỆU_CHỨNG`
- `CHẨN_ĐOÁN`
- `THUỐC`

For each entity it predicts a multi-label assertion set:

- `isNegated`
- `isFamily`
- `isHistorical`

An entity may receive zero, one, or multiple assertion labels.

## Architecture

**Backbone:** ViHealthBERT initialized from the best Phase-1 checkpoint.

**Head:** sequence classification with three independent sigmoid outputs.

**Loss:** `BCEWithLogitsLoss`.

```text
Clinical context
      |
      v
Entity position from Phase 1
      |
      v
Entity-centered context builder
      |
      +-- <TYPE_TRIEU_CHUNG>
      +-- <TYPE_CHAN_DOAN>
      +-- <TYPE_THUOC>
      +-- <ENT> ... </ENT>
      |
      v
ViHealthBERT encoder
      |
      v
3-logit classification head
      |
      v
sigmoid
      |
      +--> isNegated
      +--> isFamily
      +--> isHistorical
```

The entity is always preserved in the model input. Left and right context are cropped only when necessary to fit the model context length.

## Thresholds

Each assertion label has an independently tuned threshold on the concept validation set. The checkpoint stores the chosen thresholds so Phase 5 can reproduce the selected validation behavior.

## Model selection

The best checkpoint is selected by tuned mean sample-level Jaccard on the concept validation set.

## Drive output

```text
/content/drive/MyDrive/output_phase2_assertion/
├── run_config.json
├── training_history.json
├── best_thresholds.json
├── best/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer files...
│   ├── thresholds.json
│   └── training_meta.json
└── last/
```

---

# Phase 3 — ICD-10 Cross-Lingual Retriever

## Purpose

Phase 3 maps a Vietnamese `CHẨN_ĐOÁN` mention to ICD-10 code candidates.

Example:

```text
"viêm phổi không xác định tác nhân"
          |
          v
Top ICD-10 candidates
J18.9, J18.8, J15.9, ...
```

## Backbone

**Model:** `intfloat/multilingual-e5-base`

The ICD knowledge base is primarily English while the clinical mention is Vietnamese, so a multilingual bi-encoder is used.

### E5 input convention

```text
query: <Vietnamese diagnosis mention>
passage: <ICD English document>
```

## Retrieval architecture

```text
Vietnamese diagnosis mention
          |
          v
"query: ..."
          |
          v
Multilingual E5
          |
          v
L2-normalized query embedding
          |
          | cosine similarity / inner product
          v
Precomputed ICD embeddings
          |
          v
Top-K ICD-10 codes
```

The ICD catalog document may contain:

- ICD code metadata
- English title
- English search text
- block title
- chapter title

The code itself is kept as metadata while semantic text is encoded for retrieval.

## Training strategy

Training uses positive diagnosis-code pairs and explicit hard negatives. Repeated `(query, positive_code)` pairs are deduplicated before training.

The system evaluates the full ICD catalog rather than only the provided hard negatives.

## Model selection

Primary metric:

```text
Recall@20
```

Tie breaker:

```text
MRR
```

## Stored retrieval index

```text
best/
├── model.safetensors
├── tokenizer...
├── retrieval_config.json
├── icd_embeddings.npy
├── icd_metadata.jsonl
├── validation_top20.jsonl
├── training_meta.json
└── icd.faiss              # optional
```

`icd_embeddings.npy` is the precomputed catalog embedding matrix used during Phase-5 inference.

---

# Phase 4 — RxNorm Drug Linker

## Purpose

Phase 4 maps a detected `THUỐC` mention to RxNorm RxCUI candidates.

Example:

```text
"amlodipine 10 mg po daily"
            |
            v
RxCUI 308135
```

## Architecture

Phase 4 is a **hybrid lexical + dense retrieval model**.

```text
Drug mention
    |
    +---------------------------+
    |                           |
    v                           v
Lexical branch              Dense branch
    |                           |
Normalization               "query: ..."
Exact matching                  |
Character TF-IDF                v
3-5 char n-grams             Multilingual E5
    |                           |
    v                           v
Lexical Top-N               Dense Top-N
    \                           /
     \                         /
      +------ candidate union -+
                 |
                 v
  alpha * dense_score
+ (1-alpha) * lexical_score
                 |
                 v
           Top RxCUI candidates
```

### Lexical branch

The lexical branch uses:

- Unicode normalization
- common medication-unit normalization
- route/frequency cleanup
- exact normalized matching
- character TF-IDF
- character n-grams `(3, 5)`

### Dense branch

**Model:** `intfloat/multilingual-e5-base`

```text
query: <drug mention>
passage: <canonical RxNorm concept text>
```

### Hybrid score

```text
hybrid_score = alpha * dense_score + (1 - alpha) * lexical_score
```

`alpha` is tuned on `rxnorm_link_valid_concept.jsonl`.

### Training loss

```text
Loss = 0.70 * explicit-hard-negative CE
     + 0.30 * masked in-batch CE
```

Same-RxCUI examples are masked as in-batch negatives to avoid false-negative training signals.

### Organizer overrides

Known organizer mappings are stored separately in:

```text
organizer_overrides.json
```

These overrides are not used to inflate model-selection metrics. During inference they can provide deterministic mappings for known organizer examples.

### Example override mappings

```text
amlodipine 10 mg po daily        -> 308135
aspirin 81 mg po daily           -> 243670
metoprolol succinate xl 50 mg    -> 866436
acetaminophen 325-650 mg         -> 313782
chlorpheniramine 0.4 MG/ML       -> 360047
capsaicin 0.38 MG/ML             -> 1660761
```

## Model selection

Primary metric:

```text
Hybrid Recall@20
```

Tie breaker:

```text
Hybrid MRR
```

## Drive output

```text
/content/drive/MyDrive/output_phase4_rxnorm/
├── run_config.json
├── lexical_metrics.json
├── lexical_valid_top100.jsonl
├── zero_shot_dense_metrics.json
├── training_history.json
├── organizer_overrides.json
├── best/
│   ├── model/tokenizer files
│   ├── rxnorm_concept_embeddings.npy
│   ├── rxnorm_concept_metadata.jsonl
│   ├── hybrid_config.json
│   ├── validation_top20.jsonl
│   ├── training_meta.json
│   └── rxnorm.faiss              # optional
└── last/
```

If the Phase-4 dense bundle is unavailable, Phase 5 can operate in lexical fallback mode.

---

# Phase 5 — End-to-End Inference

## Purpose

Phase 5 loads the outputs of Phases 1–4 and applies the complete system to raw `.txt` files.

The Colab version is configured to work directly with Google Drive paths.

## Default model locations

```text
/content/drive/MyDrive/
├── data/
│   ├── rxnorm.csv
│   ├── icd10.csv
│   └── ...
├── output/
│   └── best/                         # Phase 1 NER
├── output_phase2_assertion/
│   └── best/                         # Phase 2 assertion model
├── output_phase3_icd_retriever/
│   └── best/                         # Phase 3 ICD retriever
└── output_phase4_rxnorm/
    ├── organizer_overrides.json
    └── best/                         # Phase 4 hybrid bundle, when available
```

## Long-document NER

The Phase-1 backbone has a limited context length. Phase 5 therefore uses **sliding-window NER with overlap** so long clinical records are not truncated after the first model window.

```text
Long clinical record
       |
       v
Window 1 ---------
       Window 2 ---------
              Window 3 ---------
                     ...
       |
       v
NER predictions per window
       |
       v
convert to global character offsets
       |
       v
merge overlapping / duplicate spans
```

This is important for clinical notes that are substantially longer than 256 model tokens.

---

# End-to-End Data Flow

```text
1.txt
 |
 v
Phase 1 NER
 |
 +--> TRIỆU_CHỨNG
 +--> TÊN_XÉT_NGHIỆM
 +--> KẾT_QUẢ_XÉT_NGHIỆM
 +--> CHẨN_ĐOÁN
 +--> THUỐC
 |
 v
Phase 2 assertions for symptom/diagnosis/drug
 |
 +--> isNegated
 +--> isFamily
 +--> isHistorical
 |
 +-------------------------------+
 |                               |
 v                               v
CHẨN_ĐOÁN                      THUỐC
 |                               |
 v                               v
Phase 3 ICD-10                Phase 4 RxNorm
 |                               |
 v                               v
ICD candidates                RxCUI candidates
 |                               |
 +---------------+---------------+
                 |
                 v
        Competition JSON entity
                 |
                 v
             1.json
```

---

# Input Structure

Phase 5 accepts either a directory of `.txt` files or a ZIP file containing them.

Typical test input:

```text
input/
├── 1.txt
├── 2.txt
├── 3.txt
├── ...
└── 100.txt
```

A `.txt` file contains raw free-form Vietnamese clinical text.

---

# Output Structure

The final output is:

```text
output/
├── 1.json
├── 2.json
├── 3.json
├── ...
└── 100.json
```

The folder can also be packaged as:

```text
output.zip
```

Each JSON file contains a list of detected medical entities.

## Base entity

```json
{
  "text": "đau ngực",
  "type": "TRIỆU_CHỨNG",
  "position": [45, 53],
  "assertions": []
}
```

## Diagnosis entity

```json
{
  "text": "viêm phổi",
  "type": "CHẨN_ĐOÁN",
  "position": [80, 90],
  "assertions": ["isHistorical"],
  "candidates": ["J18.9"]
}
```

## Medication entity

```json
{
  "text": "amlodipine 10 mg po daily",
  "type": "THUỐC",
  "position": [120, 145],
  "assertions": ["isHistorical"],
  "candidates": ["308135"]
}
```

### Field rules

| Field | Entity types | Description |
|---|---|---|
| `text` | all | exact text span found in the source document |
| `type` | all | one of the five clinical entity labels |
| `position` | all | `[start, end)` raw-character offset |
| `assertions` | symptom, diagnosis, medication | contextual assertion labels |
| `candidates` | diagnosis, medication | ICD-10 or RxNorm identifiers |

---

# Training Data Layout

The notebook expects its data under:

```text
/content/drive/MyDrive/data/
```

Important files include:

```text
data/
├── ner_train.jsonl
├── ner_valid_template.jsonl
├── ner_valid_concept.jsonl
├── assertion_train_balanced.jsonl
├── assertion_valid_template.jsonl
├── assertion_valid_concept.jsonl
├── icd_link_train.jsonl
├── icd_link_valid_concept.jsonl
├── icd10.csv
├── rxnorm_link_train.jsonl
├── rxnorm_link_valid_concept.jsonl
├── rxnorm.csv
└── organizer_gold_linking_seeds.jsonl
```

---

# Model Summary

| Phase | Task | Backbone / Method | Main output |
|---|---|---|---|
| Phase 1 | Clinical NER | ViHealthBERT + BIO token classification | entity spans and types |
| Phase 2 | Assertion classification | ViHealthBERT + 3-output sigmoid head | contextual assertion labels |
| Phase 3 | ICD-10 linking | multilingual-E5 bi-encoder | ICD-10 candidates |
| Phase 4 | RxNorm linking | lexical TF-IDF + multilingual-E5 hybrid | RxCUI candidates |
| Phase 5 | Integration | sliding-window NER + all trained components | final JSON files |

---

# Dependencies

The notebook uses the following main Python packages:

```text
numpy
pandas
torch
transformers
scikit-learn
```

Optional acceleration/indexing:

```text
faiss
```

Google Colab is recommended because the default paths assume a mounted Google Drive at:

```text
/content/drive/MyDrive
```

---

# Colab Usage

Mount Google Drive first:

```python
from google.colab import drive
drive.mount('/content/drive')
```

The training cells can then be run phase by phase in order:

```text
Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 -> Phase 5
```

Phase 2 reuses the best Phase-1 encoder. Phase 5 expects the trained output folders shown above.

For direct Colab execution, the Phase-5 cell uses Colab-safe argument parsing and default Drive paths, so it can be executed without manually providing command-line arguments when the default input path exists.

---

# Design Principles

The implementation follows several reproducibility and evaluation safeguards:

- raw character offsets are preserved through the NER pipeline;
- validation concepts can be held out from training;
- ICD and RxNorm retrieval are evaluated against full catalogs;
- Phase-3 and Phase-4 checkpoints are chosen using held-out retrieval metrics;
- Phase-2 assertion thresholds are tuned independently;
- organizer RxNorm mappings are isolated from model-selection metrics;
- long documents use overlapping sliding-window NER at inference time;
- each phase saves configuration and metadata alongside its checkpoint.

---

# Project Structure at a Glance

```text
United0 / Colab workflow
|
+-- Phase 1: NER
|   +-- ViHealthBERT
|   +-- BIO decoding
|   `-- /MyDrive/output/best
|
+-- Phase 2: Assertions
|   +-- ViHealthBERT
|   +-- entity-centered encoding
|   +-- tuned thresholds
|   `-- /MyDrive/output_phase2_assertion/best
|
+-- Phase 3: ICD-10
|   +-- multilingual-E5
|   +-- catalog embeddings
|   `-- /MyDrive/output_phase3_icd_retriever/best
|
+-- Phase 4: RxNorm
|   +-- exact normalized lookup
|   +-- character TF-IDF
|   +-- multilingual-E5
|   +-- hybrid fusion
|   `-- /MyDrive/output_phase4_rxnorm/best
|
`-- Phase 5: End-to-End
    +-- sliding-window NER
    +-- assertion prediction
    +-- ICD linking
    +-- RxNorm linking
    `-- output/*.json -> output.zip
```

---

## Notes

This README documents the architecture and behavior implemented in `Untitled0.ipynb`. If model paths, training configurations, label schemas, or output formats are changed in the notebook, this README should be updated accordingly.

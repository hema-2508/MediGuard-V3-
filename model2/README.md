# MediGuard Model-2: Medicine Extraction

Production-ready NLP pipeline that extracts structured medicine information from prescription or OCR text using a fine-tuned **BioBERT** token-classification model.

## Output schema

```json
{
  "medicines": [
    {
      "name": "Metformin",
      "strength": "500 mg",
      "dosage": "1 tab",
      "frequency": "twice daily",
      "duration": "30 days",
      "route": "PO",
      "confidence": 0.92
    }
  ]
}
```

Each medicine record includes a `confidence` score aggregated from the extracted entity spans (mean token-level softmax probability per field).

## Project layout

```
model2/
  config.py       # paths, hyperparameters, entity schema
  dataset.py      # JSON NER loading, BIO label maps, PyTorch Dataset
  model.py        # BioBERT token-classification wrapper
  trainer.py      # AdamW + warmup + early stopping + checkpointing
  evaluate.py     # entity-level precision/recall/F1 (seqeval)
  inference.py    # single + batch extraction with confidence scores
  api.py          # FastAPI REST service
  utils.py        # logging, seeding, spaCy preprocessing, BIO helpers
  main.py         # CLI entry point
  data/prescription_ner/
    train.json
    val.json
    test.json
```

## Install

From the project root:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Model-2 adds these dependencies on top of Model-1:

- `transformers` — BioBERT backbone and tokenizers
- `spacy` — prescription text normalization and word tokenization
- `seqeval` — entity-level NER metrics
- `fastapi` + `uvicorn` — production inference API

## Data format

Training data lives under `model2/data/prescription_ner/` as JSON lists. Each example supports either:

**Token-level BIO annotations (preferred):**

```json
{
  "text": "Tab. Metformin 500 mg 1 tab BD x 30 days PO",
  "tokens": ["Tab.", "Metformin", "500", "mg", "1", "tab", "BD", "x", "30", "days", "PO"],
  "labels": ["O", "B-MEDICINE", "B-STRENGTH", "I-STRENGTH", "B-DOSAGE", "I-DOSAGE", "B-FREQUENCY", "O", "B-DURATION", "I-DURATION", "B-ROUTE"]
}
```

**Character-span annotations:**

```json
{
  "text": "Tab. Metformin 500 mg",
  "entities": [
    {"start": 5, "end": 14, "label": "MEDICINE"},
    {"start": 15, "end": 21, "label": "STRENGTH"}
  ]
}
```

Entity types: `MEDICINE`, `STRENGTH`, `DOSAGE`, `FREQUENCY`, `DURATION`, `ROUTE`.

## Usage

All commands are run from the `model2/` directory:

```bash
cd model2
```

### Train

Fine-tunes `dmis-lab/biobert-base-cased-v1.1` on the prescription NER dataset. The best checkpoint (by validation F1) is saved to `checkpoints/best_model/`:

```bash
python main.py train --epochs 30
```

### Evaluate

Runs entity-level evaluation on the held-out test set:

```bash
python main.py evaluate
```

### Predict

Single prescription:

```bash
python main.py predict --text "Tab. Metformin 500 mg 1 tab twice daily after food for 30 days PO"
```

Batch from JSON file (`["text1", "text2"]` or `{"texts": [...]}`):

```bash
python main.py predict --input-file samples.json
```

### Serve API

```bash
python main.py serve --host 0.0.0.0 --port 8001
```

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health check |
| POST | `/extract` | Single text extraction |
| POST | `/extract/batch` | Batch extraction (max 64 texts) |

Example request:

```bash
curl -X POST http://localhost:8001/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Tab. Omeprazole 20 mg 1 cap OD x 14 days PO"}'
```

### Full pipeline

```bash
python main.py all --epochs 30
```

## Design notes

- **Backbone**: BioBERT (`dmis-lab/biobert-base-cased-v1.1`) is a biomedical pretrained BERT model used as the token-classification encoder.
- **Preprocessing**: spaCy tokenizes prescription text; common OCR abbreviations (`BID` → `BD`, `TID` → `TDS`) are normalized before inference.
- **Confidence**: per-token softmax max-probability is averaged over each predicted entity span; the medicine-level score is the mean across populated fields.
- **Long text**: prescriptions exceeding `MAX_SEQ_LENGTH` use sliding-window inference with configurable stride overlap.
- **Checkpoints**: HuggingFace-format directories containing model weights, tokenizer, label map, and training metadata.
- **Early stopping**: monitors validation entity F1 with patience 5.

## Integration with MediGuard

Model-2 is the first stage in the MediGuard orchestration pipeline:

```
Prescription/OCR → Model-2 (Medicine Extraction) → Model-3 (Brand Mapping) → Model-1 (DDI) → ...
```

Import the extractor in downstream services:

```python
from model2.inference import MedicineExtractor

extractor = MedicineExtractor()
result = extractor.extract("Tab. Metformin 500 mg BD x 30 days PO")
```

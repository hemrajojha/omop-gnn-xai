# OMOP GNN XAI — Ontology-Grounded Explainable AI for Clinical Decision Support

> Code and results for the paper:  
> **"Ontology-Grounded Explainable AI to Support GNN-Based Clinical Decision Making"**  

---

## Overview

This repository provides the full implementation of a three-component XAI framework for graph neural network (GNN) predictions on electronic health records, reproducible end-to-end from raw MIMIC-IV data:

1. **Evidence Generation** — An E-TRANS GNN (HGTConv + Transformer) trained on OMOP CDM patient graphs from MIMIC-IV, with gradient-based attribution to identify top contributing clinical concepts per domain (condition, procedure, drug).

2. **Evidence Governance** — A Neo4j knowledge graph (KG) populated with OMOP ontology triples (SNOMED CT, RxNorm, LOINC via Athena). A four-rule semantic contract validates each attributed concept against the KG before narrative generation.

3. **Verbalisation** — An LLM-based narrative explainer (Claude Sonnet 4.6) generates ontology-constrained clinical explanations for four stakeholder personas: clinician, patient, auditor, and ML engineer.

The GNN achieves **AUROC 0.909** across 275 SNOMED CT diagnosis labels on the MIMIC-IV test set (15,024 patients).

---

## Repository Structure

```
omop-gnn-xai/
├── etl/                          # MIMIC-IV → OMOP CDM 5.4.1 ETL pipeline
│   ├── ddl/                      # Schema creation (raw, vocab, OMOP CDM tables)
│   ├── staging/                  # Raw CSV loading and intermediate staging tables
│   ├── etl/                      # CDM table population (person, visit, condition,
│   │                             #   procedure, drug, measurement, era tables)
│   ├── export/                   # Export clinical events, KG triples, concept metadata
│   ├── unload/                   # Final unload to OMOP CDM
│   ├── custom/                   # Custom concept mappings
│   ├── run/run_pipeline.sh       # End-to-end pipeline runner
│   ├── docker-compose.yml        # PostgreSQL container setup
│   └── .env.example              # DB connection template
├── src/
│   ├── gnn/
│   │   ├── model.py              # E-TRANS GNN (HGTConv + Transformer)
│   │   └── train.py              # Training loop and evaluation
│   ├── preprocess/
│   │   └── build_patient_graphs.py   # OMOP CDM → PyG heterogeneous graphs
│   └── kg/
│       └── train_rotate.py       # RotatE KG embedding training
├── neo4j_xai/
│   ├── omop_gnn_neo_loader.py    # KG loader + semantic contract validator
│   ├── narrative_explainer.py    # LLM narrative generation (four personas)
│   ├── evaluate_narratives.py    # Evaluation metrics (SF, HRC, P@k, DM)
│   ├── render_results.py         # HTML report renderer
│   ├── ontology_schema.cypher    # Neo4j schema and constraints
│   └── ontology_viz.cypher       # KG visualisation queries
├── results/
│   └── sample_narratives_synthetic/  # 1 synthetic patient × 4 personas (format example)
├── figures/
│   ├── omop_gnn_xai_framework.drawio   # Three-component framework diagram
│   └── omop_gnn_kg_instance.drawio     # KG instance diagram
├── .env.example                  # Anthropic + Neo4j credentials template
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.11 (tested with 3.11.15; scikit-learn 1.8 needs 3.11+)
- PostgreSQL 14+ (or Docker)
- Neo4j 5.20.0+
- An [Anthropic API key](https://console.anthropic.com/)
- MIMIC-IV access via [PhysioNet](https://physionet.org/content/mimiciv/) (credentialed)
- Athena vocabulary download: SNOMED CT, RxNorm, LOINC ([athena.ohdsi.org](https://athena.ohdsi.org/))

Install Python dependencies (tested with PyTorch 2.5.1, CUDA 12.4, PyTorch Geometric 2.7.0, PyKEEN 1.11.1). Install PyTorch for your platform first, then the rest:

```bash
# GPU (CUDA 12.4)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
# or CPU only
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

---

## Setup

### 1. Environment variables

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
```

### 2. OMOP CDM ETL pipeline

> Patient-level data cannot be shared due to the MIMIC-IV Data Use Agreement.  
> See [Data Access](#data-access) for how to obtain MIMIC-IV.

The `etl/` directory contains the full SQL pipeline mapping MIMIC-IV to OMOP CDM 5.4.1:

| Stage | Scripts | Description |
|-------|---------|-------------|
| DDL | `ddl/01–05` | Create raw, vocabulary, and OMOP CDM schemas |
| Staging | `staging/00–04` | Load raw CSVs and intermediate staging tables |
| ETL | `etl/01–21` | Populate CDM tables (person, visit, condition, procedure, drug, measurement, era) |
| Export | `export/01–03` | Export clinical events, KG triples, and concept metadata |
| Unload | `unload/01` | Final unload to OMOP CDM |

**Start PostgreSQL:**
```bash
cd etl
cp .env.example .env   # fill in DB credentials
docker-compose up -d
```

**Run the full pipeline:**
```bash
bash etl/run/run_pipeline.sh
```

### 3. Download Athena vocabularies

Download SNOMED CT, RxNorm, and LOINC from [athena.ohdsi.org](https://athena.ohdsi.org/) and place the CSV files in a local directory. Update the path in `etl/staging/01_load_vocab.sql`.

### 4. Build patient graphs

```bash
python src/preprocess/build_patient_graphs.py --mimic4_root /path/to/mimic4
```

### 5. Train RotatE KG embeddings

```bash
python src/kg/train_rotate.py
```

### 6. Train the GNN

```bash
python src/gnn/train.py --mimic4_root /path/to/mimic4 --epochs 50
```

### 7. Load KG and run semantic contract validation

```bash
# Apply Neo4j schema
cypher-shell -u neo4j -p <password> -f neo4j_xai/ontology_schema.cypher

# Load attribution results and validate against semantic contract
python neo4j_xai/omop_gnn_neo_loader.py
```

---

## Running the Narrative Explainer

Generate explanations for a single patient across all four personas:

```bash
python neo4j_xai/narrative_explainer.py \
    --patient_idx 42 \
    --all_personas \
    --out_dir results/
```

Generate for all patients in the KG:

```bash
python neo4j_xai/narrative_explainer.py --all --all_personas --out_dir results/
```

Render an interactive HTML report:

```bash
python neo4j_xai/render_results.py
# Opens results/report.html in browser
```

---

## Sample Output

`results/sample_narratives_synthetic/` contains narratives for one **synthetic** patient across all four personas, showing the output format. Each JSON includes:

- Predicted diagnosis label and confidence score
- Conformance seal status (semantic contract pass/fail)
- Generated narrative text
- Number of generation attempts and post-generation validation status

Narratives, evaluation results and per-patient outputs for real MIMIC-IV patients are not published, because the MIMIC-IV data use agreement does not allow sharing patient-level data. Running `neo4j_xai/narrative_explainer.py` and `evaluate_narratives.py` with credentialed access regenerates them locally (`results/sample_narratives/` and `results/evaluation.json`).

---

## Semantic Contract

Each attributed concept is validated against four rules before narrative generation:

| Rule | Description |
|------|-------------|
| `RULE_ATHENA_001` | Concept must exist in the Athena vocabulary |
| `RULE_SCORE_001` | Attribution score must exceed threshold (0.001) |
| `RULE_HIERARCHY_001` | Concept must have a traceable parent in the OMOP hierarchy |
| `RULE_DOMAIN_001` | Concept domain must match its clinical category |

Explanations with no violations receive a **conformance seal**.

---


## Data Access

Patient-level data (MIMIC-IV) cannot be shared under the PhysioNet Data Use Agreement. To reproduce results from raw data:

1. Complete the required training at [CITI Program](https://about.citiprogram.org/)
2. Apply for credentialed access at [physionet.org/content/mimiciv](https://physionet.org/content/mimiciv/)
3. Follow the ETL instructions in [Setup](#2-omop-cdm-etl-pipeline) above

---

## License

Code: MIT License  
Data: Subject to [PhysioNet Credentialed Health Data License](https://physionet.org/content/mimiciv/view-license/) and MIMIC-IV Data Use Agreement

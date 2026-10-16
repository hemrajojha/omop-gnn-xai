"""
omop_gnn_neo_loader.py
======================
Builds a Neo4j knowledge graph from OMOP GNN attribution results.
Adapted from the HOGE neo_loader.py schema pattern for clinical OMOP concepts.

Workflow:
  1. On HPC   : batch_explain.py     -> attribution_p{idx}.json
  2. Locally  : query_concepts.py    -> attribution_p{idx}_enriched.json
  3. Locally  : omop_gnn_neo_loader  -> Neo4j KG  (this script)

Neo4j Schema
------------
Nodes:
  (:Patient)             patient_idx, subject_id
  (:GnnExplanation)      explanation_id, model_name, label_snomed, label_name,
                         probability, logit, n_concepts, conformance_seal
  (:OmopConcept)         concept_id, concept_name, domain, concept_class,
                         concept_code, vocabulary_id
  (:ConceptAttribution)  id, score, rank, domain, passes_contract
  (:ClinicalRule)        rule_id, name, description, check_type

Relationships:
  (Patient)            -[:HAS_EXPLANATION]->  (GnnExplanation)
  (GnnExplanation)     -[:PREDICTED_LABEL]->  (OmopConcept)
  (GnnExplanation)     -[:HAS_ATTRIBUTION]->  (ConceptAttribution)
  (ConceptAttribution) -[:FOR_CONCEPT]->      (OmopConcept)
  (OmopConcept)        -[:IS_A {levels_up}]-> (OmopConcept)   # SNOMED hierarchy
  (ConceptAttribution) -[:PASSES_RULE]->      (ClinicalRule)
  (ConceptAttribution) -[:FAILS_RULE]->       (ClinicalRule)
  (GnnExplanation)     -[:VIOLATES_RULE]->    (ClinicalRule)

Usage:
  python analysis/omop_gnn_neo_loader.py \\
      --json_dir  analysis/explain/batch_enriched \\
      --patient_ids  42 100 500 1000 2000 3000 4000 5000

  # Or point to a specific set of enriched JSON files:
  python analysis/omop_gnn_neo_loader.py \\
      --json_files attribution_p42_enriched.json attribution_p100_enriched.json
"""

import argparse
import json
import sys
from pathlib import Path

from neo4j import GraphDatabase

# ============================================================
# CONFIG
# ============================================================

NEO4J_URI      = "neo4j://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "neo4j123"

MODEL_NAME           = "E-TRANS (HGTConv + Transformer, E11)"
SCORE_THRESHOLD      = 0.001   # minimum attribution score to cite in narrative
ANCESTOR_DEPTH_LIMIT = 3       # max IS_A hops loaded into KG


# ============================================================
# NEO4J DRIVER + HELPER
# ============================================================

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def run_query(q, params=None):
    with driver.session() as session:
        result = session.run(q, params or {})
        return result.data()


def run_query_silent(q, params=None):
    with driver.session() as session:
        session.run(q, params or {})


# ============================================================
# 1. SCHEMA CONSTRAINTS
# ============================================================

def create_constraints():
    print("Creating constraints...")
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Patient)            REQUIRE p.patient_idx    IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (e:GnnExplanation)     REQUIRE e.explanation_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:OmopConcept)        REQUIRE c.concept_id     IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (a:ConceptAttribution) REQUIRE a.id             IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (r:ClinicalRule)       REQUIRE r.rule_id        IS UNIQUE",
    ]
    for c in constraints:
        run_query_silent(c)
    print("  Constraints created")


# ============================================================
# 2. OMOP CONCEPT NODES
# Loaded from enriched JSON 'concepts' dict (already queried from OMOP CDM).
# ============================================================

def load_omop_concepts(concepts: dict, label_snomed: str, label_name: str):
    concept_query = """
    MERGE (c:OmopConcept {concept_id: $concept_id})
    SET c.concept_name  = $concept_name,
        c.domain        = $domain,
        c.concept_class = $concept_class,
        c.concept_code  = $concept_code
    """
    for cid_str, info in concepts.items():
        run_query_silent(concept_query, {
            "concept_id":    int(cid_str),
            "concept_name":  info.get("name", f"concept_{cid_str}"),
            "domain":        info.get("domain", "Unknown"),
            "concept_class": info.get("class",  "Unknown"),
            "concept_code":  info.get("code",   ""),
        })

    # Also ensure the predicted label concept exists
    if label_snomed and label_name:
        run_query_silent(concept_query, {
            "concept_id":    int(label_snomed),
            "concept_name":  label_name,
            "domain":        "Condition",
            "concept_class": "Clinical Finding",
            "concept_code":  label_snomed,
        })


# ============================================================
# 3. SNOMED IS_A HIERARCHY EDGES
# Loaded from enriched JSON ancestors lists.
# ============================================================

def load_snomed_hierarchy(concepts: dict):
    ancestor_concept_query = """
    MERGE (c:OmopConcept {concept_id: $concept_id})
    ON CREATE SET c.concept_name  = $concept_name,
                  c.domain        = 'Condition',
                  c.concept_class = $concept_class,
                  c.concept_code  = ''
    """
    is_a_query = """
    MATCH (child:OmopConcept  {concept_id: $child_id})
    MATCH (parent:OmopConcept {concept_id: $parent_id})
    MERGE (child)-[r:IS_A {levels_up: $levels_up}]->(parent)
    """
    # For ancestors, we only have name/class — no concept_id.
    # Use a separate AncestorConcept label to avoid the concept_id UNIQUE constraint.
    ancestor_by_name_query = """
    MERGE (a:AncestorConcept {concept_name: $concept_name})
    ON CREATE SET a.domain        = 'Condition',
                  a.concept_class = $concept_class,
                  a.concept_code  = ''
    WITH a
    MATCH (child:OmopConcept {concept_id: $child_id})
    MERGE (child)-[:IS_A {levels_up: $levels_up}]->(a)
    """
    for cid_str, info in concepts.items():
        for anc in info.get("ancestors", []):
            if anc.get("levels_up", 99) > ANCESTOR_DEPTH_LIMIT:
                continue
            run_query_silent(ancestor_by_name_query, {
                "concept_name": anc["name"],
                "concept_class": anc.get("class", "Unknown"),
                "child_id":     int(cid_str),
                "levels_up":    anc["levels_up"],
            })


# ============================================================
# 4. CLINICAL RULES  (Semantic Contract)
# Analogous to HOGE PolicyRules — four OMOP-grounded plausibility checks.
# ============================================================

def create_clinical_rules():
    print("Creating Clinical Rules (Semantic Contract)...")
    rules = [
        {
            "rule_id":    "RULE_ATHENA_001",
            "name":       "Concept Citation Validity",
            "check_type": "existence",
            "description": (
                "Every attributed concept must exist in the Athena vocabulary "
                "with a valid concept_name. Concepts absent from Athena cannot "
                "be cited in the narrative."
            ),
        },
        {
            "rule_id":    "RULE_SCORE_001",
            "name":       "Attribution Significance",
            "check_type": "threshold",
            "description": (
                f"Attribution score must exceed {SCORE_THRESHOLD} to be cited. "
                "Trivially low scores indicate negligible influence and must not "
                "appear as narrative drivers."
            ),
        },
        {
            "rule_id":    "RULE_HIERARCHY_001",
            "name":       "Hierarchy Traceability",
            "check_type": "hierarchy",
            "description": (
                "Attributed conditions must have at least one IS_A ancestor "
                f"within {ANCESTOR_DEPTH_LIMIT} levels in the SNOMED CT hierarchy, "
                "enabling provenance tracing in the narrative."
            ),
        },
        {
            "rule_id":    "RULE_DOMAIN_001",
            "name":       "Domain Consistency",
            "check_type": "domain",
            "description": (
                "The OMOP domain of an attributed concept must match its "
                "assigned clinical category (Condition / Procedure / Drug). "
                "Cross-domain mismatches indicate a vocabulary mapping error."
            ),
        },
    ]

    rule_query = """
    MERGE (r:ClinicalRule {rule_id: $rule_id})
    SET r.name        = $name,
        r.check_type  = $check_type,
        r.description = $description
    """
    for rule in rules:
        run_query_silent(rule_query, rule)
    print(f"  {len(rules)} Clinical Rules created")


# ============================================================
# 5. LOAD PATIENT + EXPLANATION + ATTRIBUTIONS
# ============================================================

def load_patient(data: dict):
    patient_idx  = data["patient_idx"]
    subject_id   = data.get("subject_id", -1)
    label_snomed = data["label_snomed"]
    label_name   = data.get("label_name", label_snomed)
    logit        = data["logit"]
    probability  = data["probability"]
    attributions = data["attributions"]   # {domain: [{concept_id, score}]}
    concepts     = data.get("concepts", {})

    explanation_id = f"EXP_P{patient_idx}"

    # ── Patient node ──────────────────────────────────────────────────────────
    run_query_silent("""
        MERGE (p:Patient {patient_idx: $patient_idx})
        SET p.subject_id = $subject_id
    """, {"patient_idx": patient_idx, "subject_id": subject_id})

    # ── GnnExplanation node ───────────────────────────────────────────────────
    n_concepts = sum(len(v) for v in attributions.values())
    run_query_silent("""
        MERGE (e:GnnExplanation {explanation_id: $explanation_id})
        SET e.model_name    = $model_name,
            e.label_snomed  = $label_snomed,
            e.label_name    = $label_name,
            e.probability   = $probability,
            e.logit         = $logit,
            e.n_concepts    = $n_concepts,
            e.conformance_seal = false
    """, {
        "explanation_id": explanation_id,
        "model_name":     MODEL_NAME,
        "label_snomed":   label_snomed,
        "label_name":     label_name,
        "probability":    probability,
        "logit":          logit,
        "n_concepts":     n_concepts,
    })

    # ── Patient → Explanation ─────────────────────────────────────────────────
    run_query_silent("""
        MATCH (p:Patient        {patient_idx:    $patient_idx})
        MATCH (e:GnnExplanation {explanation_id: $explanation_id})
        MERGE (p)-[:HAS_EXPLANATION]->(e)
    """, {"patient_idx": patient_idx, "explanation_id": explanation_id})

    # ── Explanation → Predicted Label concept ─────────────────────────────────
    run_query_silent("""
        MATCH (e:GnnExplanation {explanation_id: $explanation_id})
        MERGE  (lc:OmopConcept  {concept_id: $label_concept_id})
        ON CREATE SET lc.concept_name  = $label_name,
                      lc.domain        = 'Condition',
                      lc.concept_class = 'Clinical Finding',
                      lc.concept_code  = $label_snomed
        MERGE (e)-[:PREDICTED_LABEL]->(lc)
    """, {
        "explanation_id":   explanation_id,
        "label_concept_id": int(label_snomed),
        "label_name":       label_name,
        "label_snomed":     label_snomed,
    })

    # ── ConceptAttribution nodes ───────────────────────────────────────────────
    domain_rank = {"condition": 0, "procedure": 0, "drug": 0}
    for domain, entries in attributions.items():
        for entry in entries:
            cid   = entry["concept_id"]
            score = entry["score"]
            domain_rank[domain] += 1
            rank = domain_rank[domain]

            cid_str  = str(cid)
            info     = concepts.get(cid_str, {})
            cname    = info.get("name", f"concept_{cid}")
            cdomain  = info.get("domain", domain.capitalize())
            cclass   = info.get("class",  "Unknown")
            ccode    = info.get("code",   "")
            attr_id  = f"{explanation_id}_{domain.upper()}_{cid}"

            # ConceptAttribution node
            run_query_silent("""
                MERGE (a:ConceptAttribution {id: $attr_id})
                SET a.score          = $score,
                    a.rank           = $rank,
                    a.domain         = $domain,
                    a.passes_contract = false
            """, {
                "attr_id": attr_id,
                "score":   score,
                "rank":    rank,
                "domain":  domain,
            })

            # Ensure OmopConcept node exists
            run_query_silent("""
                MERGE (c:OmopConcept {concept_id: $concept_id})
                ON CREATE SET c.concept_name  = $concept_name,
                              c.domain        = $domain,
                              c.concept_class = $concept_class,
                              c.concept_code  = $concept_code
            """, {
                "concept_id":    cid,
                "concept_name":  cname,
                "domain":        cdomain,
                "concept_class": cclass,
                "concept_code":  ccode,
            })

            # GnnExplanation → ConceptAttribution → OmopConcept
            run_query_silent("""
                MATCH (e:GnnExplanation     {explanation_id: $explanation_id})
                MATCH (a:ConceptAttribution {id:             $attr_id})
                MATCH (c:OmopConcept        {concept_id:     $concept_id})
                MERGE (e)-[:HAS_ATTRIBUTION]->(a)
                MERGE (a)-[:FOR_CONCEPT]->(c)
            """, {
                "explanation_id": explanation_id,
                "attr_id":        attr_id,
                "concept_id":     cid,
            })

    return explanation_id


# ============================================================
# 6. SEMANTIC CONTRACT VALIDATION
# Runs four Cypher checks per patient; creates PASSES_RULE / FAILS_RULE edges.
# Sets conformance_seal = true if all attributions pass all rules.
# ============================================================

def validate_semantic_contract(explanation_id: str):

    # ── Rule 1: Concept Citation Validity ─────────────────────────────────────
    # Pass: concept_name is not a placeholder (i.e., was found in Athena)
    run_query_silent("""
        MATCH (e:GnnExplanation {explanation_id: $eid})-[:HAS_ATTRIBUTION]->(a)
              -[:FOR_CONCEPT]->(c:OmopConcept)
        MATCH (r:ClinicalRule {rule_id: 'RULE_ATHENA_001'})
        WHERE c.concept_name IS NOT NULL
          AND NOT c.concept_name STARTS WITH 'concept_'
        MERGE (a)-[:PASSES_RULE]->(r)
    """, {"eid": explanation_id})

    run_query_silent("""
        MATCH (e:GnnExplanation {explanation_id: $eid})-[:HAS_ATTRIBUTION]->(a)
              -[:FOR_CONCEPT]->(c:OmopConcept)
        MATCH (r:ClinicalRule {rule_id: 'RULE_ATHENA_001'})
        WHERE c.concept_name STARTS WITH 'concept_'
           OR c.concept_name IS NULL
        MERGE (a)-[:FAILS_RULE]->(r)
        MERGE (e)-[:VIOLATES_RULE]->(r)
    """, {"eid": explanation_id})

    # ── Rule 2: Attribution Significance ──────────────────────────────────────
    run_query_silent("""
        MATCH (e:GnnExplanation {explanation_id: $eid})-[:HAS_ATTRIBUTION]->(a)
        MATCH (r:ClinicalRule {rule_id: 'RULE_SCORE_001'})
        WHERE a.score >= $threshold
        MERGE (a)-[:PASSES_RULE]->(r)
    """, {"eid": explanation_id, "threshold": SCORE_THRESHOLD})

    run_query_silent("""
        MATCH (e:GnnExplanation {explanation_id: $eid})-[:HAS_ATTRIBUTION]->(a)
        MATCH (r:ClinicalRule {rule_id: 'RULE_SCORE_001'})
        WHERE a.score < $threshold
        MERGE (a)-[:FAILS_RULE]->(r)
        MERGE (e)-[:VIOLATES_RULE]->(r)
    """, {"eid": explanation_id, "threshold": SCORE_THRESHOLD})

    # ── Rule 3: Hierarchy Traceability ────────────────────────────────────────
    # Pass: attributed condition concept has at least one IS_A ancestor in KG
    run_query_silent("""
        MATCH (e:GnnExplanation {explanation_id: $eid})-[:HAS_ATTRIBUTION]->(a)
              -[:FOR_CONCEPT]->(c:OmopConcept)
        MATCH (r:ClinicalRule {rule_id: 'RULE_HIERARCHY_001'})
        WHERE a.domain = 'condition'
          AND EXISTS { MATCH (c)-[:IS_A]->(:AncestorConcept) }
        MERGE (a)-[:PASSES_RULE]->(r)
    """, {"eid": explanation_id})

    # Procedures and drugs always pass hierarchy check (not SNOMED condition hierarchy)
    run_query_silent("""
        MATCH (e:GnnExplanation {explanation_id: $eid})-[:HAS_ATTRIBUTION]->(a)
        MATCH (r:ClinicalRule {rule_id: 'RULE_HIERARCHY_001'})
        WHERE a.domain IN ['procedure', 'drug']
        MERGE (a)-[:PASSES_RULE]->(r)
    """, {"eid": explanation_id})

    run_query_silent("""
        MATCH (e:GnnExplanation {explanation_id: $eid})-[:HAS_ATTRIBUTION]->(a)
              -[:FOR_CONCEPT]->(c:OmopConcept)
        MATCH (r:ClinicalRule {rule_id: 'RULE_HIERARCHY_001'})
        WHERE a.domain = 'condition'
          AND NOT EXISTS { MATCH (c)-[:IS_A]->(:AncestorConcept) }
        MERGE (a)-[:FAILS_RULE]->(r)
        MERGE (e)-[:VIOLATES_RULE]->(r)
    """, {"eid": explanation_id})

    # ── Rule 4: Domain Consistency ────────────────────────────────────────────
    # Pass: attribution domain matches concept's OMOP domain_id
    domain_map = {
        "condition": "Condition",
        "procedure": "Procedure",
        "drug":      "Drug",
    }
    for attr_domain, omop_domain in domain_map.items():
        run_query_silent("""
            MATCH (e:GnnExplanation {explanation_id: $eid})-[:HAS_ATTRIBUTION]->(a)
                  -[:FOR_CONCEPT]->(c:OmopConcept)
            MATCH (r:ClinicalRule {rule_id: 'RULE_DOMAIN_001'})
            WHERE a.domain = $attr_domain
              AND c.domain  = $omop_domain
            MERGE (a)-[:PASSES_RULE]->(r)
        """, {"eid": explanation_id, "attr_domain": attr_domain, "omop_domain": omop_domain})

        run_query_silent("""
            MATCH (e:GnnExplanation {explanation_id: $eid})-[:HAS_ATTRIBUTION]->(a)
                  -[:FOR_CONCEPT]->(c:OmopConcept)
            MATCH (r:ClinicalRule {rule_id: 'RULE_DOMAIN_001'})
            WHERE a.domain = $attr_domain
              AND c.domain <> $omop_domain
            MERGE (a)-[:FAILS_RULE]->(r)
            MERGE (e)-[:VIOLATES_RULE]->(r)
        """, {"eid": explanation_id, "attr_domain": attr_domain, "omop_domain": omop_domain})

    # ── Update passes_contract flag on each ConceptAttribution ────────────────
    run_query_silent("""
        MATCH (e:GnnExplanation {explanation_id: $eid})-[:HAS_ATTRIBUTION]->(a)
        WHERE NOT EXISTS { MATCH (a)-[:FAILS_RULE]->() }
        SET a.passes_contract = true
    """, {"eid": explanation_id})

    # ── Set conformance_seal on GnnExplanation if no VIOLATES_RULE ────────────
    run_query_silent("""
        MATCH (e:GnnExplanation {explanation_id: $eid})
        WHERE NOT EXISTS { MATCH (e)-[:VIOLATES_RULE]->() }
        SET e.conformance_seal = true
    """, {"eid": explanation_id})

    # ── Report ─────────────────────────────────────────────────────────────────
    result = run_query("""
        MATCH (e:GnnExplanation {explanation_id: $eid})
        RETURN e.conformance_seal AS seal,
               e.label_name      AS label,
               e.probability     AS prob
    """, {"eid": explanation_id})
    if result:
        row  = result[0]
        seal = "PASS" if row["seal"] else "FAIL"
        print(f"    Contract: {seal} | {row['label']} | p={row['prob']:.4f}")


# ============================================================
# MAIN
# ============================================================

def collect_json_files(args) -> list[Path]:
    files = []
    if args.json_files:
        for f in args.json_files:
            p = Path(f)
            if p.exists():
                files.append(p)
            else:
                print(f"  [warn] File not found: {f}")
    elif args.json_dir and args.patient_ids:
        d = Path(args.json_dir)
        for pid in args.patient_ids:
            # Prefer enriched version
            enriched = d / f"attribution_p{pid}_enriched.json"
            plain    = d / f"attribution_p{pid}.json"
            if enriched.exists():
                files.append(enriched)
            elif plain.exists():
                files.append(plain)
                print(f"  [warn] No enriched JSON for patient {pid} — "
                      f"run query_concepts.py first for full concept metadata")
            else:
                print(f"  [warn] No JSON found for patient {pid} in {d}")
    elif args.json_dir:
        files = sorted(Path(args.json_dir).glob("*_enriched.json"))
        if not files:
            files = sorted(Path(args.json_dir).glob("attribution_p*.json"))
    return files


def main():
    p = argparse.ArgumentParser(description="Load OMOP GNN attribution results into Neo4j")
    p.add_argument("--json_dir",     default=None,
                   help="Directory containing attribution JSON files")
    p.add_argument("--json_files",   nargs="+", default=None,
                   help="Specific enriched JSON file paths")
    p.add_argument("--patient_ids",  nargs="+", type=int, default=None,
                   help="Patient indices to load from --json_dir")
    p.add_argument("--neo4j_uri",    default=NEO4J_URI)
    p.add_argument("--neo4j_user",   default=NEO4J_USER)
    p.add_argument("--neo4j_pass",   default=NEO4J_PASSWORD)
    args = p.parse_args()

    files = collect_json_files(args)
    if not files:
        print("[error] No JSON files found. Provide --json_files or --json_dir.")
        sys.exit(1)

    print(f"\nOMOP GNN → Neo4j Loader")
    print(f"  Target:   {args.neo4j_uri}")
    print(f"  Patients: {len(files)}")

    # ── Setup ──────────────────────────────────────────────────────────────────
    create_constraints()
    create_clinical_rules()

    # ── Per-patient loading ────────────────────────────────────────────────────
    print(f"\nLoading {len(files)} patient(s)...")
    loaded = errors = 0

    for json_path in files:
        try:
            with open(json_path) as f:
                data = json.load(f)

            pid = data["patient_idx"]
            print(f"\n  Patient {pid}  ({json_path.name})")

            concepts = data.get("concepts", {})

            # 1. Concept nodes
            load_omop_concepts(concepts, data["label_snomed"], data.get("label_name"))

            # 2. SNOMED IS_A hierarchy
            load_snomed_hierarchy(concepts)

            # 3. Patient + explanation + attributions
            explanation_id = load_patient(data)

            # 4. Semantic contract validation
            validate_semantic_contract(explanation_id)

            loaded += 1

        except Exception as exc:
            errors += 1
            print(f"  [ERROR] {json_path.name}: {exc}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    stats = run_query("""
        MATCH (p:Patient)            WITH count(p) AS patients
        MATCH (e:GnnExplanation)     WITH patients, count(e) AS explanations
        MATCH (c:OmopConcept)        WITH patients, explanations, count(c) AS concepts
        MATCH (a:ConceptAttribution) WITH patients, explanations, concepts, count(a) AS attributions
        RETURN patients, explanations, concepts, attributions
    """)
    if stats:
        s = stats[0]
        print(f"Neo4j KG Summary:")
        print(f"  Patients           : {s['patients']}")
        print(f"  GNN Explanations   : {s['explanations']}")
        print(f"  OMOP Concepts      : {s['concepts']}")
        print(f"  Concept Attributions: {s['attributions']}")

    sealed = run_query("""
        MATCH (e:GnnExplanation)
        RETURN e.conformance_seal AS seal, count(e) AS n
        ORDER BY seal DESC
    """)
    print(f"\n  Conformance seals:")
    for row in sealed:
        label = "PASS" if row["seal"] else "FAIL"
        print(f"    {label}: {row['n']} explanation(s)")

    print(f"\n  Loaded: {loaded} | Errors: {errors}")
    print(f"\nNeo4j KG successfully built!")
    driver.close()


if __name__ == "__main__":
    main()

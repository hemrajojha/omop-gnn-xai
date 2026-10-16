// ============================================================
// OMOP GNN XAI — Ontology Schema
// Equivalent to HOGE Fig. 2 — run once in Neo4j Browser
// before loading any patient data.
// ============================================================


// ── 1. Schema Constraints ─────────────────────────────────────────────────────

CREATE CONSTRAINT IF NOT EXISTS FOR (p:Patient)             REQUIRE p.patient_idx    IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (e:GnnExplanation)      REQUIRE e.explanation_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (c:OmopConcept)         REQUIRE c.concept_id     IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (a:ConceptAttribution)  REQUIRE a.id             IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (r:ClinicalRule)        REQUIRE r.rule_id        IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (m:PatientGNN)          REQUIRE m.model_id       IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (g:GradientAttributor)  REQUIRE g.method_id      IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (d:OmopDomain)          REQUIRE d.domain_name    IS UNIQUE;


// ── 2. Clinical Concept Domains  (analogous to rvo:IndependentVariable) ───────
//    Represent the three clinical input spaces the GNN reasons over.

MERGE (d_cond:OmopDomain {domain_name: 'Condition'})
SET   d_cond.vocabulary   = 'SNOMED CT',
      d_cond.omop_table   = 'condition_occurrence',
      d_cond.description  = 'Clinical condition / diagnosis concept';

MERGE (d_proc:OmopDomain {domain_name: 'Procedure'})
SET   d_proc.vocabulary   = 'SNOMED CT / CPT4 / ICD10PCS',
      d_proc.omop_table   = 'procedure_occurrence',
      d_proc.description  = 'Clinical procedure concept';

MERGE (d_drug:OmopDomain {domain_name: 'Drug'})
SET   d_drug.vocabulary   = 'RxNorm',
      d_drug.omop_table   = 'drug_exposure',
      d_drug.description  = 'Drug / medication concept';


// ── 3. PatientGNN Model node  (analogous to rvo:Model / Loan Predictor) ───────

MERGE (model:PatientGNN {model_id: 'E-TRANS'})
SET   model.architecture  = 'HGTConv + Transformer (dual-branch, alpha=0.8)',
      model.training_data = 'MIMIC-IV → OMOP CDM (SNOMED CT standardised)',
      model.auroc          = 0.909,
      model.n_labels       = 275,
      model.checkpoint    = 'logs/e11/checkpoints/best_model.pt',
      model.reference     = 'E-TRANS (extended TRANS, IJCAI 2024 base)';


// ── 4. Attribution Method node  (analogous to SHAP in HOGE) ──────────────────

MERGE (attr:GradientAttributor {method_id: 'GRADIENT_L1'})
SET   attr.name        = 'Gradient Attribution (L1 Norm)',
      attr.formula     = 'score_i = ||d logit / d emb_i||_1',
      attr.description = 'L1 norm of gradient of predicted logit w.r.t. concept node embedding',
      attr.reference   = 'Erion et al. 2021, Nature Machine Intelligence 3(7)';


// ── 5. Method → Model  (analogous to SHAP forExplainer → ModelExplanation) ───

MATCH (model:PatientGNN       {model_id:  'E-TRANS'})
MATCH (attr:GradientAttributor {method_id: 'GRADIENT_L1'})
MERGE (attr)-[:EXPLAINS_OUTPUT_OF]->(model);


// ── 6. Method → Domains  (what gets attributed) ───────────────────────────────

MATCH (attr:GradientAttributor {method_id: 'GRADIENT_L1'})
MATCH (d_cond:OmopDomain {domain_name: 'Condition'})
MATCH (d_proc:OmopDomain {domain_name: 'Procedure'})
MATCH (d_drug:OmopDomain {domain_name: 'Drug'})
MERGE (attr)-[:ATTRIBUTES_DOMAIN]->(d_cond)
MERGE (attr)-[:ATTRIBUTES_DOMAIN]->(d_proc)
MERGE (attr)-[:ATTRIBUTES_DOMAIN]->(d_drug);


// ── 7. Clinical Rules  (analogous to PolicyRule + RiskFactor in HOGE) ─────────

MERGE (r1:ClinicalRule {rule_id: 'RULE_ATHENA_001'})
SET   r1.name        = 'Concept Citation Validity',
      r1.check_type  = 'existence',
      r1.severity    = 'HARD',
      r1.description = 'Every cited concept must exist in the Athena vocabulary with a valid concept_name. Concepts absent from Athena cannot appear in any narrative.';

MERGE (r2:ClinicalRule {rule_id: 'RULE_SCORE_001'})
SET   r2.name        = 'Attribution Significance',
      r2.check_type  = 'threshold',
      r2.threshold   = 0.001,
      r2.severity    = 'SOFT',
      r2.description = 'Attribution score must exceed 0.001. Trivially low scores indicate negligible influence and must not appear as narrative drivers.';

MERGE (r3:ClinicalRule {rule_id: 'RULE_HIERARCHY_001'})
SET   r3.name        = 'Hierarchy Traceability',
      r3.check_type  = 'hierarchy',
      r3.max_levels  = 3,
      r3.severity    = 'SOFT',
      r3.description = 'Attributed condition must have at least one IS_A ancestor within 3 levels in the SNOMED CT hierarchy, enabling provenance tracing in the narrative.';

MERGE (r4:ClinicalRule {rule_id: 'RULE_DOMAIN_001'})
SET   r4.name        = 'Domain Consistency',
      r4.check_type  = 'domain',
      r4.severity    = 'HARD',
      r4.description = 'The OMOP domain of an attributed concept must match its assigned clinical category (Condition / Procedure / Drug). Mismatches indicate a vocabulary mapping error.';


// ── 8. Rules → Domains they govern  (analogous to PolicyRule IMPACTS RiskFactor)

MATCH (r1:ClinicalRule {rule_id: 'RULE_ATHENA_001'})
MATCH (r4:ClinicalRule {rule_id: 'RULE_DOMAIN_001'})
MATCH (d_cond:OmopDomain {domain_name: 'Condition'})
MATCH (d_proc:OmopDomain {domain_name: 'Procedure'})
MATCH (d_drug:OmopDomain {domain_name: 'Drug'})
MERGE (r1)-[:GOVERNS_DOMAIN]->(d_cond)
MERGE (r1)-[:GOVERNS_DOMAIN]->(d_proc)
MERGE (r1)-[:GOVERNS_DOMAIN]->(d_drug)
MERGE (r4)-[:GOVERNS_DOMAIN]->(d_cond)
MERGE (r4)-[:GOVERNS_DOMAIN]->(d_proc)
MERGE (r4)-[:GOVERNS_DOMAIN]->(d_drug);

MATCH (r2:ClinicalRule {rule_id: 'RULE_SCORE_001'})
MATCH (d_cond:OmopDomain {domain_name: 'Condition'})
MATCH (d_proc:OmopDomain {domain_name: 'Procedure'})
MATCH (d_drug:OmopDomain {domain_name: 'Drug'})
MERGE (r2)-[:GOVERNS_DOMAIN]->(d_cond)
MERGE (r2)-[:GOVERNS_DOMAIN]->(d_proc)
MERGE (r2)-[:GOVERNS_DOMAIN]->(d_drug);

MATCH (r3:ClinicalRule {rule_id: 'RULE_HIERARCHY_001'})
MATCH (d_cond:OmopDomain {domain_name: 'Condition'})
MERGE (r3)-[:GOVERNS_DOMAIN]->(d_cond);


// ── 9. Verify schema ──────────────────────────────────────────────────────────
// Run this after the above to confirm everything loaded:

// MATCH (n) RETURN labels(n) AS label, count(n) AS count ORDER BY count DESC;
// MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS count ORDER BY count DESC;

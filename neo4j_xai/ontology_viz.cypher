// ============================================================
// OMOP GNN XAI — Abstract Ontology Visualization
// Fig. 2 equivalent for OMOP GNN (run in Neo4j Browser)
//
// Creates schema-level nodes grounded on RVO model, extended
// with OMOP CDM classes. Visualize with:
//   MATCH (n:RVO_Class)-[r]->(m) RETURN n,r,m
//   MATCH (n:SchemaNode)-[r]->(m) RETURN n,r,m
// ============================================================


// ── PART A: RVO Base Classes (background ontology) ────────────────────────────
// Analogous to rvo:IndependentVariable, rvo:Measure, rvo:Model, rvo:Case in HOGE

MERGE (rvo_entity:RVO_Class   {name: 'omop:ClinicalEntity'})
SET   rvo_entity.description  = 'Base class for all OMOP CDM clinical concepts';

MERGE (rvo_measure:RVO_Class  {name: 'omop:AttributionMeasure'})
SET   rvo_measure.description = 'Quantitative gradient-based attribution score (rvo:Measure)';

MERGE (rvo_case:RVO_Class     {name: 'gnn:PatientCase'})
SET   rvo_case.description    = 'A patient prediction case (rvo:Case)';

MERGE (rvo_model:RVO_Class    {name: 'gnn:ClinicalPredictor'})
SET   rvo_model.description   = 'The GNN predictive model (rvo:Model)';


// ── PART B: OMOP GNN Schema Classes (extending RVO) ──────────────────────────

MERGE (s_patient:SchemaNode   {name: 'Patient'})
SET   s_patient.description   = 'MIMIC-IV patient record';

MERGE (s_exp:SchemaNode       {name: 'GnnExplanation'})
SET   s_exp.description       = 'GNN prediction output + attribution for one patient';

MERGE (s_concept:SchemaNode   {name: 'OmopConcept'})
SET   s_concept.description   = 'Athena-standardised clinical concept (SNOMED CT / RxNorm / LOINC)';

MERGE (s_attr:SchemaNode      {name: 'ConceptAttribution'})
SET   s_attr.description      = 'Gradient attribution score for one concept node',
      s_attr.value_type       = 'float:score';

MERGE (s_rule:SchemaNode      {name: 'ClinicalRule'})
SET   s_rule.description      = 'OMOP-grounded clinical plausibility constraint (Semantic Contract)';

MERGE (s_domain:SchemaNode    {name: 'OmopDomain'})
SET   s_domain.description    = 'Clinical input domain: Condition / Procedure / Drug';

MERGE (s_gnn:SchemaNode       {name: 'PatientGNN'})
SET   s_gnn.description       = 'E-TRANS model (HGTConv + Transformer, dual-branch)';

MERGE (s_grad:SchemaNode      {name: 'GradientAttributor'})
SET   s_grad.description      = 'Gradient Attribution (L1 Norm) — analogous to SHAP';


// ── PART C: rdf:type links — Schema classes → RVO base classes ────────────────

MATCH (s_patient:SchemaNode  {name: 'Patient'})
MATCH (rvo_case:RVO_Class    {name: 'gnn:PatientCase'})
MERGE (s_patient)-[:rdf_type]->(rvo_case);

MATCH (s_gnn:SchemaNode      {name: 'PatientGNN'})
MATCH (rvo_model:RVO_Class   {name: 'gnn:ClinicalPredictor'})
MERGE (s_gnn)-[:rdf_type]->(rvo_model);

MATCH (s_attr:SchemaNode     {name: 'ConceptAttribution'})
MATCH (rvo_measure:RVO_Class {name: 'omop:AttributionMeasure'})
MERGE (s_attr)-[:rdf_type]->(rvo_measure);

MATCH (s_concept:SchemaNode  {name: 'OmopConcept'})
MATCH (rvo_entity:RVO_Class  {name: 'omop:ClinicalEntity'})
MERGE (s_concept)-[:rdf_type]->(rvo_entity);

MATCH (s_domain:SchemaNode   {name: 'OmopDomain'})
MATCH (rvo_entity:RVO_Class  {name: 'omop:ClinicalEntity'})
MERGE (s_domain)-[:derivedFrom]->(rvo_entity);


// ── PART D: Ontology relationships between schema classes ─────────────────────

// Patient → GnnExplanation
MATCH (s_patient:SchemaNode {name: 'Patient'})
MATCH (s_exp:SchemaNode     {name: 'GnnExplanation'})
MERGE (s_patient)-[:HAS_EXPLANATION]->(s_exp);

// PatientGNN → GnnExplanation  (model viaModel case)
MATCH (s_gnn:SchemaNode {name: 'PatientGNN'})
MATCH (s_exp:SchemaNode {name: 'GnnExplanation'})
MERGE (s_gnn)-[:viaModel]->(s_exp);

// GradientAttributor → GnnExplanation  (forExplainer — same as SHAP in HOGE)
MATCH (s_grad:SchemaNode {name: 'GradientAttributor'})
MATCH (s_exp:SchemaNode  {name: 'GnnExplanation'})
MERGE (s_grad)-[:forExplainer]->(s_exp);

// GradientAttributor → PatientGNN
MATCH (s_grad:SchemaNode {name: 'GradientAttributor'})
MATCH (s_gnn:SchemaNode  {name: 'PatientGNN'})
MERGE (s_grad)-[:EXPLAINS_OUTPUT_OF]->(s_gnn);

// GnnExplanation → ConceptAttribution  (hasAttribution — analogous to hasContribution)
MATCH (s_exp:SchemaNode  {name: 'GnnExplanation'})
MATCH (s_attr:SchemaNode {name: 'ConceptAttribution'})
MERGE (s_exp)-[:hasAttribution]->(s_attr);

// GnnExplanation → OmopConcept  (predictedLabel)
MATCH (s_exp:SchemaNode     {name: 'GnnExplanation'})
MATCH (s_concept:SchemaNode {name: 'OmopConcept'})
MERGE (s_exp)-[:predictedLabel]->(s_concept);

// ConceptAttribution → OmopConcept  (forConcept — analogous to forFeature)
MATCH (s_attr:SchemaNode    {name: 'ConceptAttribution'})
MATCH (s_concept:SchemaNode {name: 'OmopConcept'})
MERGE (s_attr)-[:forConcept]->(s_concept);

// OmopConcept → OmopConcept  (IS_A — SNOMED hierarchy)
MATCH (s_concept:SchemaNode {name: 'OmopConcept'})
MERGE (s_concept)-[:IS_A {note: 'SNOMED CT hierarchy, levels_up 1..3'}]->(s_concept);

// OmopDomain → OmopConcept  (hasConceptType — domain contains concepts)
MATCH (s_domain:SchemaNode  {name: 'OmopDomain'})
MATCH (s_concept:SchemaNode {name: 'OmopConcept'})
MERGE (s_domain)-[:hasConceptType]->(s_concept);

// ClinicalRule → OmopDomain  (governsDomain — analogous to PolicyRule hasFeature)
MATCH (s_rule:SchemaNode   {name: 'ClinicalRule'})
MATCH (s_domain:SchemaNode {name: 'OmopDomain'})
MERGE (s_rule)-[:governsDomain]->(s_domain);

// GnnExplanation ↔ ClinicalRule  (violatesRule / linkedToRule)
MATCH (s_exp:SchemaNode  {name: 'GnnExplanation'})
MATCH (s_rule:SchemaNode {name: 'ClinicalRule'})
MERGE (s_exp)-[:VIOLATES_RULE]->(s_rule);

// ConceptAttribution ↔ ClinicalRule  (passesRule / failsRule)
MATCH (s_attr:SchemaNode {name: 'ConceptAttribution'})
MATCH (s_rule:SchemaNode {name: 'ClinicalRule'})
MERGE (s_attr)-[:PASSES_RULE]->(s_rule);


// ── PART E: Visualize queries (run separately in Browser) ─────────────────────

// Fig. 2 equivalent — full abstract ontology:
// MATCH (n)-[r]->(m)
// WHERE n:SchemaNode OR n:RVO_Class
// RETURN n, r, m

// Fig. 3 equivalent — concrete patient instance (after running neo_loader.py):
// MATCH p = (pt:Patient {patient_idx: 42})-[*1..3]-()
// RETURN p LIMIT 80

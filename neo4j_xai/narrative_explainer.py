"""
narrative_explainer.py
======================
Component C — Narrative Explainer.

Queries the Neo4j KG for a validated attribution bundle for a given patient,
then calls the Claude API to generate an ontology-constrained clinical narrative.

Three-layer semantic contract enforcement:
  1. Pre-generation  : prompt instructs LLM to reference only KG-validated concepts
  2. Post-generation : programmatic check that all cited concept names exist in KG
  3. Rejection/retry : on violation, regenerate with feedback (up to MAX_RETRIES)

Usage:
  python neo4j_xai/narrative_explainer.py \\
      --patient_idx  42 \\
      --persona      clinician \\
      --out_dir      neo4j_xai/results

  # All patients in KG, all personas:
  python neo4j_xai/narrative_explainer.py --all --out_dir neo4j_xai/results

Requirements:
  pip install anthropic neo4j
  Set ANTHROPIC_API_KEY in environment or .env file.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import anthropic
from neo4j import GraphDatabase

# ============================================================
# CONFIG
# ============================================================

NEO4J_URI      = "neo4j://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "neo4j"

CLAUDE_MODEL = "claude-sonnet-4-6"
MAX_RETRIES  = 3
MAX_TOKENS   = 1024

PERSONAS = ["clinician", "patient", "auditor", "ml_engineer"]

# ============================================================
# NEO4J + CLAUDE CLIENTS  (initialised after arg parsing)
# ============================================================

driver = None
claude = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


def neo4j_query(q, params=None):
    with driver.session() as session:
        return session.run(q, params or {}).data()


# ============================================================
# RETRIEVE VALIDATED EVIDENCE BUNDLE FROM NEO4J
# ============================================================

def get_evidence_bundle(patient_idx: int) -> dict | None:
    explanation = neo4j_query("""
        MATCH (p:Patient {patient_idx: $pid})-[:HAS_EXPLANATION]->(e:GnnExplanation)
        MATCH (e)-[:PREDICTED_LABEL]->(lc:OmopConcept)
        RETURN e.explanation_id   AS explanation_id,
               e.label_name       AS label_name,
               lc.concept_code    AS label_snomed_code,
               e.probability      AS probability,
               e.logit            AS logit,
               e.conformance_seal AS conformance_seal
    """, {"pid": patient_idx})

    if not explanation:
        return None
    exp = explanation[0]

    # Only validated (passing) attributions per domain
    attributions = neo4j_query("""
        MATCH (e:GnnExplanation {explanation_id: $eid})-[:HAS_ATTRIBUTION]->(a:ConceptAttribution)
              -[:FOR_CONCEPT]->(c:OmopConcept)
        WHERE a.passes_contract = true
        OPTIONAL MATCH (c)-[:IS_A {levels_up: 1}]->(parent:AncestorConcept)
        RETURN a.domain       AS domain,
               a.score        AS score,
               a.rank         AS rank,
               c.concept_id   AS concept_id,
               c.concept_name AS concept_name,
               c.concept_code AS concept_code,
               c.domain       AS omop_domain,
               collect(DISTINCT parent.concept_name) AS parent_classes
        ORDER BY a.domain, a.rank
    """, {"eid": exp["explanation_id"]})

    # Violations (for auditor persona)
    violations = neo4j_query("""
        MATCH (e:GnnExplanation {explanation_id: $eid})-[:VIOLATES_RULE]->(r:ClinicalRule)
        RETURN r.rule_id AS rule_id, r.name AS rule_name, r.description AS description
    """, {"eid": exp["explanation_id"]})

    # Group attributions by domain
    by_domain = {"condition": [], "procedure": [], "drug": []}
    for row in attributions:
        by_domain[row["domain"]].append(row)

    return {
        "patient_idx":     patient_idx,
        "explanation_id":  exp["explanation_id"],
        "label_name":      exp["label_name"],
        "label_snomed":    exp["label_snomed_code"],
        "probability":     exp["probability"],
        "conformance_seal": exp["conformance_seal"],
        "attributions":    by_domain,
        "violations":      violations,
    }


# ============================================================
# BUILD PROMPTS (per persona)
# ============================================================

def _format_attributions(bundle: dict) -> str:
    lines = []
    for domain in ("condition", "procedure", "drug"):
        entries = bundle["attributions"].get(domain, [])
        if not entries:
            continue
        lines.append(f"\n{domain.upper()} concepts (ranked by attribution score):")
        for e in entries:
            parents = ", ".join(e["parent_classes"]) if e["parent_classes"] else "—"
            lines.append(
                f"  {e['rank']}. {e['concept_name']} "
                f"(score={e['score']:.4f}, SNOMED: {e['concept_code']}) "
                f"| parent class: {parents}"
            )
    return "\n".join(lines)


GROUNDING_INSTRUCTION = (
    "STRICT GROUNDING RULE: You MUST reference ONLY the clinical concepts listed "
    "in the Evidence Bundle below. Do not invent, infer, or hallucinate any "
    "concept names, drug names, procedure names, or clinical relationships that "
    "are not explicitly present in the bundle. Every claim in your response must "
    "be traceable to a concept in the bundle."
)


def build_prompt(bundle: dict, persona: str) -> str:
    label      = bundle["label_name"]
    prob       = bundle["probability"]
    attrs_text = _format_attributions(bundle)
    seal       = "PASSED" if bundle["conformance_seal"] else "FAILED"

    evidence_block = f"""
=== EVIDENCE BUNDLE (Semantic Contract: {seal}) ===
Predicted diagnosis  : {label}
Prediction confidence: {prob:.1%}
Patient index        : {bundle['patient_idx']}

Validated attributed concepts:
{attrs_text}
=== END EVIDENCE BUNDLE ===
"""

    if persona == "clinician":
        return f"""{GROUNDING_INSTRUCTION}

You are a clinical decision support system generating an explanation for a clinician.

{evidence_block}

Generate a clinical explanation with these sections:
1. SUMMARY: One sentence stating the predicted diagnosis and confidence.
2. PRIMARY DRIVERS: Top 2-3 contributing conditions and what they suggest clinically. Use the exact concept names from the bundle at least once each.
3. SUPPORTING EVIDENCE: Key procedures and drugs from the bundle and their clinical relevance. Use the exact concept names from the bundle at least once each.
4. SNOMED CONTEXT: For the top condition, note its SNOMED parent class.
5. COUNTERFACTUAL: One sentence on what absence of the top-ranked condition would imply.
6. RECOMMENDATION: One sentence for the clinician.

Be concise (under 200 words). Use clinical language. Do not mention attribution scores."""

    elif persona == "patient":
        return f"""{GROUNDING_INSTRUCTION}

You are generating a plain-language explanation for a patient (no medical jargon, no scores).

{evidence_block}

Write 3-4 sentences explaining:
- What condition was flagged and why it matters
- What symptoms or tests contributed (use plain names)
- What the patient's care team will do next

Do not mention SNOMED codes, attribution scores, or model names. Maximum 100 words."""

    elif persona == "auditor":
        violations_text = ""
        if bundle["violations"]:
            violations_text = "\nContract violations detected:\n" + "\n".join(
                f"  - {v['rule_name']}: {v['description']}"
                for v in bundle["violations"]
            )
        else:
            violations_text = "\nNo contract violations detected."

        return f"""{GROUNDING_INSTRUCTION}

You are generating a clinical audit report for a compliance officer.

{evidence_block}
{violations_text}

Generate a structured audit report with:
1. CONFORMANCE STATUS: Pass/Fail and number of validated concepts.
2. EVIDENCE PROVENANCE: List each domain (condition/procedure/drug) with concept count and top concept name + SNOMED code.
3. CONTRACT CHECKS: State which of the four rules passed/failed (Athena citation, attribution significance, hierarchy traceability, domain consistency).
4. TRACEABILITY NOTE: One sentence on how each cited concept maps to its KG path.

Use formal language. Under 250 words."""

    elif persona == "ml_engineer":
        return f"""{GROUNDING_INSTRUCTION}

You are generating a technical summary for an ML engineer reviewing the GNN explanation.

{evidence_block}

Generate a technical report covering:
1. MODEL OUTPUT: Logit, probability, and predicted label SNOMED code.
2. TOP ATTRIBUTIONS: List concept_id, concept_name, domain, and score for top 3 per domain.
3. CONTRACT STATUS: Conformance seal result and any violations.
4. QUALITY FLAGS: Note if any concepts had low scores near threshold, or missing hierarchy paths.

Use precise technical language. Present as bullet points. Under 200 words."""

    else:
        raise ValueError(f"Unknown persona: {persona}")


# ============================================================
# POST-GENERATION VALIDATION
# ============================================================

def validate_narrative(narrative: str, bundle: dict) -> tuple[bool, list[str]]:
    """
    Check that every concept name in the narrative exists in the validated bundle.
    Returns (passed, list_of_violations).
    """
    valid_names = set()
    valid_names.add(bundle["label_name"].lower())
    for entries in bundle["attributions"].values():
        for e in entries:
            valid_names.add(e["concept_name"].lower())
            for parent in e["parent_classes"]:
                if parent:
                    valid_names.add(parent.lower())

    # Extract capitalised medical-looking phrases (2+ words, title case) from narrative
    # This is a lightweight heuristic check
    candidate_phrases = re.findall(r'[A-Z][a-z]+(?: [a-z]+){1,4}', narrative)
    violations = []
    for phrase in candidate_phrases:
        # Skip common non-clinical phrases
        if phrase.lower() in {"the model", "the patient", "the clinician",
                              "the system", "the evidence", "the prediction",
                              "the following", "the bundle", "the narrative"}:
            continue
        # Flag if phrase looks clinical but not in bundle
        if len(phrase) > 10 and phrase.lower() not in valid_names:
            # Only flag if it looks like a diagnosis/drug/procedure name
            if any(kw in phrase.lower() for kw in
                   ["syndrome", "disease", "disorder", "failure", "infection",
                    "cancer", "tumor", "osis", "itis", "emia", "therapy", "ectomy"]):
                violations.append(phrase)

    return len(violations) == 0, violations


# ============================================================
# GENERATE NARRATIVE (with retry)
# ============================================================

def generate_narrative(bundle: dict, persona: str) -> dict:
    prompt = build_prompt(bundle, persona)
    attempt = 0
    violation_feedback = ""

    while attempt < MAX_RETRIES:
        attempt += 1
        full_prompt = prompt
        if violation_feedback:
            full_prompt += f"\n\nPREVIOUS ATTEMPT VIOLATION: {violation_feedback}\nPlease regenerate strictly within the evidence bundle."

        message = claude.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": full_prompt}],
        )
        narrative = message.content[0].text
        passed, violations = validate_narrative(narrative, bundle)

        if passed:
            return {
                "persona":           persona,
                "patient_idx":       bundle["patient_idx"],
                "label_name":        bundle["label_name"],
                "probability":       bundle["probability"],
                "conformance_seal":  bundle["conformance_seal"],
                "narrative":         narrative,
                "attempts":          attempt,
                "post_gen_passed":   True,
                "generated_at":      datetime.now().isoformat(),
            }
        else:
            violation_feedback = f"Potentially unsupported terms detected: {', '.join(violations[:3])}"
            print(f"    [retry {attempt}] Post-gen check flagged: {violations[:3]}")

    # Return best effort after max retries
    return {
        "persona":          persona,
        "patient_idx":      bundle["patient_idx"],
        "label_name":       bundle["label_name"],
        "probability":      bundle["probability"],
        "conformance_seal": bundle["conformance_seal"],
        "narrative":        narrative,
        "attempts":         attempt,
        "post_gen_passed":  False,
        "generated_at":     datetime.now().isoformat(),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    global driver
    p = argparse.ArgumentParser()
    p.add_argument("--patient_idx", type=int, default=None)
    p.add_argument("--all",         action="store_true",
                   help="Generate for all patients in Neo4j KG")
    p.add_argument("--persona",     choices=PERSONAS, default="clinician")
    p.add_argument("--all_personas", action="store_true",
                   help="Generate all four persona narratives per patient")
    p.add_argument("--out_dir",     default="neo4j_xai/results")
    p.add_argument("--neo4j_uri",   default=NEO4J_URI)
    p.add_argument("--neo4j_user",  default=NEO4J_USER)
    p.add_argument("--neo4j_pass",  default=NEO4J_PASSWORD)
    args = p.parse_args()

    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_pass))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect patient indices
    if args.all:
        rows = neo4j_query("MATCH (p:Patient) RETURN p.patient_idx AS idx ORDER BY idx")
        patient_ids = [r["idx"] for r in rows]
    elif args.patient_idx is not None:
        patient_ids = [args.patient_idx]
    else:
        print("[error] Provide --patient_idx or --all")
        sys.exit(1)

    personas = PERSONAS if args.all_personas else [args.persona]

    print(f"\nNarrative Explainer")
    print(f"  Patients : {patient_ids}")
    print(f"  Personas : {personas}")
    print(f"  Model    : {CLAUDE_MODEL}")
    print(f"  Output   : {out_dir}\n")

    for pid in patient_ids:
        bundle = get_evidence_bundle(pid)
        if not bundle:
            print(f"  [warn] No KG data for patient {pid} — run neo_loader first")
            continue

        seal = "SEALED" if bundle["conformance_seal"] else "UNSEALED"
        print(f"  Patient {pid} | {bundle['label_name']} | p={bundle['probability']:.4f} | {seal}")

        for persona in personas:
            print(f"    Generating [{persona}]...", end=" ")
            result = generate_narrative(bundle, persona)
            status = "OK" if result["post_gen_passed"] else "WARN"
            print(f"{status} (attempt {result['attempts']})")

            # Save result
            out_file = out_dir / f"p{pid}_{persona}.json"
            with open(out_file, "w") as f:
                json.dump(result, f, indent=2)

    print(f"\nResults saved to: {out_dir}")
    driver.close()


if __name__ == "__main__":
    main()

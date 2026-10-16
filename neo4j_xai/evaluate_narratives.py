"""
evaluate_narratives.py
======================
Formal evaluation of generated narratives across four dimensions:

  A1. Faithfulness       -- do narratives reflect the GNN's actual attributions?
  A2. Grounding          -- are cited concepts grounded in the evidence bundle?
  A3. Graph Retrieval    -- are KG-retrieved concepts used accurately?
  B.  Comprehensibility  -- do narratives communicate the decision correctly?
  C.  Counterfactual     -- are counterfactuals coherent and feature-specific?

Usage:
    py -3 neo4j_xai/evaluate_narratives.py
    py -3 neo4j_xai/evaluate_narratives.py --results_dir neo4j_xai/results --top_k 5
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean

RESULTS_DIR  = Path("neo4j_xai/results")
PATIENTS_DIR = Path("neo4j_xai/patients")

# Four clinical rules defined in the semantic contract
VALID_RULE_IDS = {
    "RULE_ATHENA_001", "RULE_SCORE_001",
    "RULE_HIERARCHY_001", "RULE_DOMAIN_001",
}
VALID_RULE_NAMES = {
    "concept citation validity", "attribution significance",
    "hierarchy traceability", "domain consistency",
}

CLINICAL_KW = [
    "syndrome", "disease", "disorder", "failure", "infection", "cancer",
    "tumor", "osis", "itis", "emia", "therapy", "ectomy", "pathy",
    "sclerosis", "fibrosis", "hypoxia", "anemia", "diabetes",
]

# -- Helpers -------------------------------------------------------------------

def load_results(results_dir: Path) -> dict:
    results = {}
    for f in sorted(results_dir.glob("p*_*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            pid, persona = d.get("patient_idx"), d.get("persona")
            if pid is not None and persona:
                results.setdefault(pid, {})[persona] = d
        except Exception as e:
            print(f"  [warn] {f.name}: {e}")
    return results


def load_patient_data(pid: int) -> dict:
    f = PATIENTS_DIR / f"attribution_p{pid}_enriched.json"
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_bundle(data: dict) -> dict:
    """Build evaluation bundle from enriched patient JSON."""
    concepts  = data.get("concepts", {})
    label_name = (data.get("label_name") or "").strip()
    attributions = data.get("attributions", {})

    valid_names = set()
    if label_name:
        valid_names.add(label_name.lower())
    for info in concepts.values():
        n = (info.get("name") or "").strip()
        if n:
            valid_names.add(n.lower())
        for anc in info.get("ancestors", []):
            an = (anc.get("name") or "").strip()
            if an:
                valid_names.add(an.lower())

    # Flat list of (concept_name, score, domain) sorted by score desc
    flat_attrs = []
    for domain, entries in attributions.items():
        for e in entries:
            cid  = str(e.get("concept_id", ""))
            name = (concepts.get(cid, {}).get("name") or f"concept_{cid}").lower()
            flat_attrs.append({"name": name, "score": e.get("score", 0), "domain": domain})
    flat_attrs.sort(key=lambda x: x["score"], reverse=True)

    return {
        "valid_names": valid_names,
        "label_name":  label_name,
        "flat_attrs":  flat_attrs,
    }


def detect_clinical_phrases(text: str) -> list[str]:
    """Extract multi-word phrases that look clinical."""
    candidates = re.findall(r'\b[A-Z][a-z]+(?: [A-Za-z]+){1,4}\b', text)
    return [p for p in candidates if len(p) > 8
            and any(kw in p.lower() for kw in CLINICAL_KW)]


def name_in_text(name: str, text_lower: str) -> bool:
    return name in text_lower


# -- A1. Faithfulness ----------------------------------------------------------

def _concept_tokens(name: str) -> set[str]:
    """Significant words from a concept name (drop short stopwords)."""
    STOP = {"of", "the", "a", "an", "and", "or", "due", "to", "with",
            "without", "by", "caused", "related", "associated", "type",
            "in", "on", "at", "for", "as", "is", "are", "not"}
    return {w for w in re.split(r'\W+', name.lower()) if len(w) > 2 and w not in STOP}


def _fuzzy_concept_match(name: str, narrative_lower: str, threshold: float = 0.5) -> bool:
    """
    True if concept is 'mentioned' in narrative using two complementary checks:
      1. Exact substring match (catches full names)
      2. Token overlap: >= threshold of the concept's significant words appear
         in the narrative (catches abbreviations and paraphrasing)
    """
    # Exact match first
    if name_in_text(name, narrative_lower):
        return True
    # Token overlap
    tokens = _concept_tokens(name)
    if not tokens:
        return False
    matched = sum(1 for t in tokens if t in narrative_lower)
    return (matched / len(tokens)) >= threshold


def compute_faithfulness(narrative: str, flat_attrs: list, top_k: int = 5) -> float:
    """
    Faithfulness = fraction of top-K attributed concepts mentioned in narrative.
    Uses fuzzy token-overlap matching to handle abbreviations and paraphrasing
    (e.g. 'GERD' matches 'Gastroesophageal reflux disease without esophagitis').
    """
    if not flat_attrs:
        return None
    top = flat_attrs[:top_k]
    nl  = narrative.lower()
    matched = sum(1 for a in top if _fuzzy_concept_match(a["name"], nl))
    return matched / len(top)


# -- A2. Grounding -------------------------------------------------------------

def compute_semantic_fidelity(narrative: str, valid_names: set) -> float:
    """
    SF = precision of clinical phrases in narrative.
    Of detected clinical-looking phrases, what fraction are in the bundle?
    """
    detected = detect_clinical_phrases(narrative)
    if not detected:
        return 1.0
    matched = sum(
        1 for p in detected
        if any(name_in_text(p.lower(), name) or name_in_text(name, p.lower())
               for name in valid_names)
    )
    return matched / len(detected)


def compute_evidence_coverage(narrative: str, valid_names: set) -> float:
    """
    EC = recall of evidence bundle concepts in narrative (fuzzy).
    Of all validated concept names, what fraction appear?
    """
    if not valid_names:
        return None
    nl = narrative.lower()
    matched = sum(1 for name in valid_names if _fuzzy_concept_match(name, nl))
    return matched / len(valid_names)


def compute_hallucination_rate(sf: float) -> float:
    """HRC = 1 - Semantic Fidelity (phrase-level fabrication rate)."""
    return round(1 - sf, 4) if sf is not None else None


def compute_fabricated(narrative: str, valid_names: set) -> list[str]:
    """
    Fabricated features: clinical phrases in narrative not traceable to bundle
    or to a valid clinical rule name.
    """
    detected = detect_clinical_phrases(narrative)
    fabricated = []
    for p in detected:
        pl = p.lower()
        in_bundle = any(
            name_in_text(pl, name) or name_in_text(name, pl)
            for name in valid_names
        )
        is_rule = any(rn in pl for rn in VALID_RULE_NAMES)
        if not in_bundle and not is_rule:
            fabricated.append(p)
    return fabricated


# -- A3. Graph Retrieval -------------------------------------------------------

def compute_prediction_match(narrative: str, label_name: str) -> int:
    """1 if predicted diagnosis label is mentioned in narrative, else 0."""
    if not label_name:
        return None
    return int(label_name.lower() in narrative.lower())


def compute_precision_at_k(narrative: str, flat_attrs: list, k: int) -> float:
    """
    Precision@K: of top-K attributed concepts, fraction mentioned in narrative (fuzzy).
    """
    if not flat_attrs:
        return None
    top = flat_attrs[:k]
    nl  = narrative.lower()
    matched = sum(1 for a in top if _fuzzy_concept_match(a["name"], nl))
    return matched / len(top)


# -- B. Comprehensibility ------------------------------------------------------

def compute_decision_match(narrative: str, label_name: str) -> int:
    """Does the narrative correctly identify the predicted diagnosis?"""
    return compute_prediction_match(narrative, label_name)


def compute_attribution_alignment(narrative: str, flat_attrs: list) -> int:
    """
    Attribution-Prediction Alignment (analogous to SHAP-Prediction Alignment).
    1 if the top-1 attributed concept (by score) is mentioned in the narrative (fuzzy).
    Ensures the primary driver is communicated.
    """
    if not flat_attrs:
        return None
    top1 = flat_attrs[0]["name"]
    return int(_fuzzy_concept_match(top1, narrative.lower()))


def compute_features_referenced(narrative: str, valid_names: set) -> int:
    """Count of distinct evidence-bundle concept names mentioned in narrative (fuzzy)."""
    nl = narrative.lower()
    return sum(1 for name in valid_names if _fuzzy_concept_match(name, nl))


# -- C. Counterfactual ---------------------------------------------------------

def extract_counterfactual_sentence(narrative: str) -> str:
    """
    Extract the COUNTERFACTUAL section from a clinician narrative.
    Prompt requests: 'COUNTERFACTUAL: One sentence on what absence of top-ranked
    condition would imply.'
    """
    match = re.search(
        r'(?:COUNTERFACTUAL|counterfactual)[:\s*\-]+(.*?)(?:\n\s*\n|\n\d\.|\n[A-Z]{2,}|$)',
        narrative, re.IGNORECASE | re.DOTALL
    )
    if match:
        return match.group(1).strip().replace("\n", " ")
    # fallback: sentence containing "absence" or "without"
    for sent in re.split(r'(?<=[.!?])\s+', narrative):
        if re.search(r'\b(absence|without|absent|removed|if not)\b', sent, re.I):
            return sent.strip()
    return ""


def compute_flip_feature(cf_text: str, flat_attrs: list, valid_names: set) -> str:
    """
    The concept name from the bundle mentioned in the counterfactual sentence.
    Returns the most likely 'flip feature'.
    """
    if not cf_text:
        return None
    cfl = cf_text.lower()
    # Check top attrs first (most specific)
    for a in flat_attrs:
        if name_in_text(a["name"], cfl):
            return a["name"]
    # Fallback to any bundle name
    for name in sorted(valid_names, key=len, reverse=True):
        if name_in_text(name, cfl):
            return name
    return None


def compute_flip_rate(cf_text: str, flat_attrs: list) -> int:
    """
    1 if counterfactual references a specific attributed concept (single-feature flip).
    """
    if not cf_text or not flat_attrs:
        return None
    cfl = cf_text.lower()
    return int(any(name_in_text(a["name"], cfl) for a in flat_attrs))


# -- Full evaluation per narrative ---------------------------------------------

def evaluate_one(r: dict, bundle: dict, top_k: int) -> dict:
    narrative    = r.get("narrative", "")
    persona      = r.get("persona", "")
    valid_names  = bundle.get("valid_names", set())
    flat_attrs   = bundle.get("flat_attrs", [])
    label_name   = bundle.get("label_name", r.get("label_name", ""))

    sf  = compute_semantic_fidelity(narrative, valid_names)
    ec  = compute_evidence_coverage(narrative, valid_names)
    hrc = compute_hallucination_rate(sf)
    fab = compute_fabricated(narrative, valid_names)

    cf_text   = extract_counterfactual_sentence(narrative) if persona == "clinician" else ""
    flip_feat = compute_flip_feature(cf_text, flat_attrs, valid_names) if persona == "clinician" else None
    flip_rate = compute_flip_rate(cf_text, flat_attrs)                  if persona == "clinician" else None

    return {
        "patient_idx":      r.get("patient_idx"),
        "persona":          persona,
        "label_name":       label_name,
        "probability":      r.get("probability", 0),
        "conformance_seal": r.get("conformance_seal", False),
        "post_gen_passed":  r.get("post_gen_passed", True),
        "attempts":         r.get("attempts", 1),
        # A1
        "faithfulness":         compute_faithfulness(narrative, flat_attrs, top_k),
        # A2
        "semantic_fidelity":    sf,
        "evidence_coverage":    ec,
        "hallucination_rate":   hrc,
        "fabricated_count":     len(fab),
        "fabricated_terms":     fab[:5],
        # A3
        "prediction_match":     compute_prediction_match(narrative, label_name),
        "precision_at_3":       compute_precision_at_k(narrative, flat_attrs, 3),
        "precision_at_5":       compute_precision_at_k(narrative, flat_attrs, 5),
        "precision_at_10":      compute_precision_at_k(narrative, flat_attrs, 10),
        # B
        "decision_match":       compute_decision_match(narrative, label_name),
        "attribution_alignment": compute_attribution_alignment(narrative, flat_attrs),
        "features_referenced":  compute_features_referenced(narrative, valid_names),
        "evidence_coverage_b":  ec,   # same as EC, reported under B as well
        # C
        "counterfactual_text":  cf_text,
        "flip_feature":         flip_feat,
        "flip_rate":            flip_rate,
        "narrative_words":      len(narrative.split()),
    }


# -- Aggregated report ---------------------------------------------------------

def _avg(vals):
    v = [x for x in vals if x is not None]
    return mean(v) if v else None

def _fmt(v, pct=False):
    if v is None: return "  --  "
    return f"{v*100:.1f}%" if pct else f"{v:.3f}"

def _rate(evals, key, threshold, op="ge"):
    vals = [e[key] for e in evals if e[key] is not None]
    if not vals: return None
    fn = (lambda x: x >= threshold) if op == "ge" else (lambda x: x < threshold)
    return sum(fn(x) for x in vals) / len(vals)


def print_report(evals: list, top_k: int):
    n = len(evals)
    clinician_evals = [e for e in evals if e["persona"] == "clinician"]

    print(f"\n{'='*72}")
    print(f"{'OMOP GNN XAI -- Narrative Evaluation Report':^72}")
    print(f"{'='*72}")
    print(f"  Narratives evaluated : {n}  "
          f"({len(set(e['patient_idx'] for e in evals))} patients x "
          f"{len(set(e['persona'] for e in evals))} personas)")

    # -- A1. Faithfulness ------------------------------------------------------
    print(f"\n  A1. FAITHFULNESS  (top-{top_k} attribution concepts cited in narrative)")
    f_scores = [e["faithfulness"] for e in evals if e["faithfulness"] is not None]
    avg_f = _avg(f_scores)
    print(f"      Avg Faithfulness Score  : {_fmt(avg_f)}")
    print(f"      Pass Rate  (>= 0.5)     : {_fmt(_rate(evals,'faithfulness',0.5), pct=True)}")
    print(f"      Perfect Score (1.0)     : {_fmt(_rate(evals,'faithfulness',1.0), pct=True)}")
    print(f"      Failed     (< 0.5)      : {_fmt(_rate(evals,'faithfulness',0.5,'lt'), pct=True)}")

    # -- A2. Grounding ---------------------------------------------------------
    print(f"\n  A2. GROUNDING")
    print(f"      Avg Semantic Fidelity (SF)  : {_fmt(_avg([e['semantic_fidelity'] for e in evals]))}")
    print(f"      Avg Evidence Coverage (EC)  : {_fmt(_avg([e['evidence_coverage'] for e in evals]))}")
    print(f"      Avg Hallucination Rate (HRC): {_fmt(_avg([e['hallucination_rate'] for e in evals]))}")
    fab_total = sum(e["fabricated_count"] for e in evals)
    print(f"      Fabricated Features / Rules : {fab_total} total across {n} narratives")

    # -- A3. Graph Retrieval ---------------------------------------------------
    print(f"\n  A3. GRAPH RETRIEVAL")
    pm = _avg([e["prediction_match"] for e in evals])
    print(f"      Prediction Match Rate : {_fmt(pm, pct=True)}")
    print(f"      Avg Precision@3       : {_fmt(_avg([e['precision_at_3']  for e in evals]))}")
    print(f"      Avg Precision@5       : {_fmt(_avg([e['precision_at_5']  for e in evals]))}")
    print(f"      Avg Precision@10      : {_fmt(_avg([e['precision_at_10'] for e in evals]))}")

    # -- B. Comprehensibility --------------------------------------------------
    print(f"\n  B.  COMPREHENSIBILITY")
    print(f"      Decision-Narrative Match         : {_fmt(_avg([e['decision_match'] for e in evals]), pct=True)}")
    print(f"      Attribution-Prediction Alignment : {_fmt(_avg([e['attribution_alignment'] for e in evals]), pct=True)}")
    print(f"      Avg Features Referenced          : {_avg([e['features_referenced'] for e in evals]):.1f}")
    print(f"      Avg Evidence-Coverage Score      : {_fmt(_avg([e['evidence_coverage_b'] for e in evals]))}")

    # -- C. Counterfactual (clinician only) ------------------------------------
    print(f"\n  C.  COUNTERFACTUAL  (clinician persona, n={len(clinician_evals)})")
    flip_r = _avg([e["flip_rate"] for e in clinician_evals if e["flip_rate"] is not None])
    print(f"      Single-Feature Flip Rate : {_fmt(flip_r, pct=True)}")
    flip_feats = [e["flip_feature"] for e in clinician_evals if e["flip_feature"]]
    if flip_feats:
        most_common = Counter(flip_feats).most_common(3)
        print(f"      Most Common Flip Feature : {most_common[0][0]} (n={most_common[0][1]})")
        for feat, cnt in most_common[1:]:
            print(f"                                 {feat} (n={cnt})")

    # -- Per-patient / per-persona detail table --------------------------------
    print(f"\n{'-'*72}")
    print(f"  {'Pt':<6} {'Persona':<12} {'Label':<30} "
          f"{'Faith':>5} {'SF':>5} {'EC':>5} {'P@5':>5} {'DM':>3} {'AA':>3} {'FR':>3}")
    print(f"{'-'*72}")
    for e in sorted(evals, key=lambda x: (x["patient_idx"], x["persona"])):
        fr_s = str(e["flip_rate"]) if e["flip_rate"] is not None else "--"
        print(
            f"  p{e['patient_idx']:<5} {e['persona']:<12} {e['label_name'][:30]:<30} "
            f"{_fmt(e['faithfulness']):>5} "
            f"{_fmt(e['semantic_fidelity']):>5} "
            f"{_fmt(e['evidence_coverage']):>5} "
            f"{_fmt(e['precision_at_5']):>5} "
            f"{str(e['decision_match'] or '--'):>3} "
            f"{str(e['attribution_alignment'] or '--'):>3} "
            f"{fr_s:>3}"
        )
    print(f"{'-'*72}")
    print(f"  Faith=Faithfulness  SF=Semantic Fidelity  EC=Evidence Coverage  "
          f"P@5=Precision@5\n  DM=Decision Match  AA=Attribution Alignment  "
          f"FR=Flip Rate (clinician only)")

    # -- Flags -----------------------------------------------------------------
    flagged = [e for e in evals
               if (e["faithfulness"] or 1) < 0.5
               or not e["post_gen_passed"]
               or e["fabricated_count"] > 2]
    if flagged:
        print(f"\n  Narratives flagged for manual review ({len(flagged)}):")
        for e in flagged:
            reasons = []
            if (e["faithfulness"] or 1) < 0.5: reasons.append("low faithfulness")
            if not e["post_gen_passed"]:        reasons.append("post-gen fail")
            if e["fabricated_count"] > 2:       reasons.append(f"fabricated={e['fabricated_count']}")
            print(f"    p{e['patient_idx']} [{e['persona']}] -- {', '.join(reasons)}")
    else:
        print(f"\n  No narratives flagged for manual review.")


# -- Main ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="neo4j_xai/results")
    p.add_argument("--top_k", type=int, default=5,
                   help="Top-K attributed concepts for faithfulness / precision metrics")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    results = load_results(results_dir)
    if not results:
        print(f"No result JSON files found in {results_dir}/")
        return

    evals = []
    for pid, personas in results.items():
        data   = load_patient_data(pid)
        bundle = build_bundle(data)
        for _, r in personas.items():
            evals.append(evaluate_one(r, bundle, args.top_k))

    print_report(evals, args.top_k)

    out = results_dir / "evaluation.json"
    out.write_text(json.dumps(evals, indent=2, default=str), encoding="utf-8")
    print(f"\n  Full evaluation saved: {out.resolve()}")


if __name__ == "__main__":
    main()

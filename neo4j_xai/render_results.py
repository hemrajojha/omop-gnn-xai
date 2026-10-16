"""
render_results.py
=================
Generates a self-contained HTML report from all narrative JSON files
in neo4j_xai/results/.  Markdown is rendered via marked.js.

Usage:
    py -3 neo4j_xai/render_results.py
"""

import json
from pathlib import Path

RESULTS_DIR = Path("neo4j_xai/results")
OUT_FILE    = RESULTS_DIR / "report.html"

PERSONA_LABELS = {
    "clinician":   "Clinician",
    "patient":     "Patient",
    "auditor":     "Auditor",
    "ml_engineer": "ML Engineer",
}
PERSONA_COLORS = {
    "clinician":   "#1a6fa8",
    "patient":     "#2e8b57",
    "auditor":     "#8b2252",
    "ml_engineer": "#b8600a",
}


def load_results():
    results = {}
    for f in sorted(RESULTS_DIR.glob("p*_*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            pid, persona = d.get("patient_idx"), d.get("persona")
            if pid is not None and persona:
                results.setdefault(pid, {})[persona] = d
        except Exception:
            pass
    return results


def seal_badge(sealed):
    if sealed:
        return '<span class="badge pass">&#10003; SEALED</span>'
    return '<span class="badge fail">&#10007; UNSEALED</span>'


def attempt_badge(n, passed):
    cls   = "ok" if passed else "warn"
    label = "OK" if passed else "WARN"
    return f'<span class="attempt-badge {cls}">attempt {n} &middot; {label}</span>'


def render_html(results: dict) -> str:
    # ── Collect all narrative text into a plain dict keyed by tab_id ─────────
    # Stored as JSON in a <script> tag — avoids ALL escaping issues with
    # backticks, pipes, angle brackets, etc. inside narrative Markdown.
    narratives: dict[str, str] = {}

    patient_sections = []

    for pid in sorted(results.keys()):
        personas = results[pid]
        meta   = next(iter(personas.values()))
        label  = meta.get("label_name", "Unknown")
        prob   = meta.get("probability", 0)
        sealed = meta.get("conformance_seal", False)

        tabs_html   = []
        panels_html = []

        for persona in ["clinician", "patient", "auditor", "ml_engineer"]:
            if persona not in personas:
                continue
            r      = personas[persona]
            color  = PERSONA_COLORS.get(persona, "#555")
            plabel = PERSONA_LABELS.get(persona, persona)
            tab_id = f"p{pid}_{persona}"

            # Store raw Markdown in the side-channel dict (no escaping needed)
            narratives[tab_id] = r.get("narrative", "")

            tabs_html.append(
                f'<button class="tab-btn" data-tab="{tab_id}" '
                f'style="--accent:{color}" '
                f'onclick="showTab(\'{tab_id}\')">{plabel}</button>'
            )
            panels_html.append(f"""
        <div id="{tab_id}" class="tab-panel" style="display:none">
          <div class="narrative-meta">
            {attempt_badge(r.get("attempts", 1), r.get("post_gen_passed", True))}
          </div>
          <div class="narrative-body md-render" data-tab-id="{tab_id}"></div>
        </div>""")

        if not tabs_html:
            continue

        patient_sections.append(f"""
    <div class="patient-card">
      <div class="patient-header" onclick="toggleCard(this)" title="Click to expand / collapse">
        <span class="toggle-icon">&#9660;</span>
        <span class="patient-id">Patient {pid}</span>
        <span class="patient-label">{label}</span>
        <span class="patient-prob">p&nbsp;=&nbsp;{prob:.4f}</span>
        {seal_badge(sealed)}
      </div>
      <div class="card-body">
        <div class="tab-bar">{"".join(tabs_html)}</div>
        {"".join(panels_html)}
      </div>
    </div>""")

    n_patients = len(results)
    n_total    = sum(len(v) for v in results.values())

    # Safely encode all Markdown as JSON — handles every special character
    narratives_json = json.dumps(narratives, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OMOP GNN XAI — Narrative Results</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f0f2f5; color: #1a1a2e; line-height: 1.6;
    padding: 2rem 1.5rem; max-width: 960px; margin: 0 auto;
  }}
  h1 {{ font-size: 1.6rem; font-weight: 700; margin-bottom: 0.2rem; }}
  .subtitle {{
    color: #666; font-size: 0.88rem; margin-bottom: 2rem;
    border-bottom: 1px solid #dde; padding-bottom: 1rem;
  }}

  .patient-card {{
    background: #fff; border-radius: 10px; margin-bottom: 1.75rem;
    box-shadow: 0 2px 8px rgba(0,0,0,.08); overflow: hidden;
  }}
  .patient-header {{
    padding: 0.85rem 1.3rem; background: #fafbff;
    border-bottom: 1px solid #e8eaf0;
    display: flex; align-items: center; gap: 0.7rem; flex-wrap: wrap;
    cursor: pointer; user-select: none;
    transition: background .15s;
  }}
  .patient-header:hover {{ background: #eef0f8; }}
  .patient-id    {{ font-weight: 800; font-size: 1rem; color: #1a1a2e; }}
  .patient-label {{ font-size: 0.95rem; color: #333; flex: 1; min-width: 0; }}
  .patient-prob  {{ font-size: 0.82rem; color: #666; white-space: nowrap; }}

  .toggle-icon {{
    font-size: 0.7rem; color: #999; width: 1rem; text-align: center;
    transition: transform .2s ease; display: inline-block; flex-shrink: 0;
  }}
  .patient-card.collapsed .toggle-icon {{ transform: rotate(-90deg); }}
  .patient-card.collapsed .card-body   {{ display: none; }}
  .patient-card.collapsed .patient-header {{ border-bottom: none; }}

  .badge {{
    padding: 3px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 700; white-space: nowrap;
  }}
  .badge.pass {{ background: #d1fae5; color: #065f46; }}
  .badge.fail {{ background: #fee2e2; color: #991b1b; }}
  .attempt-badge {{
    font-size: 0.75rem; padding: 2px 9px; border-radius: 20px; font-weight: 600;
  }}
  .attempt-badge.ok   {{ background: #d1fae5; color: #065f46; }}
  .attempt-badge.warn {{ background: #fef3c7; color: #92400e; }}

  .tab-bar {{
    display: flex; padding: 0 1rem;
    border-bottom: 1px solid #e8eaf0; background: #fff;
  }}
  .tab-btn {{
    background: none; border: none;
    border-bottom: 3px solid transparent;
    padding: 0.55rem 1rem; cursor: pointer;
    font-size: 0.85rem; font-weight: 600; color: #888;
    transition: color .15s, border-color .15s; white-space: nowrap;
  }}
  .tab-btn:hover  {{ color: #333; }}
  .tab-btn.active {{ color: var(--accent); border-bottom-color: var(--accent); }}

  .tab-panel      {{ padding: 1.3rem 1.5rem 1.5rem; }}
  .narrative-meta {{ margin-bottom: 0.75rem; }}

  /* ── Markdown body ── */
  .narrative-body {{ font-size: 0.9rem; line-height: 1.85; color: #2c2c3e; }}

  .narrative-body h1,
  .narrative-body h2,
  .narrative-body h3 {{
    margin-top: 1.2rem; margin-bottom: 0.3rem;
    font-weight: 700; line-height: 1.3; color: #1a1a2e;
  }}
  .narrative-body h1 {{ font-size: 1.1rem; }}
  .narrative-body h2 {{ font-size: 1.0rem; }}
  .narrative-body h3 {{
    font-size: 0.88rem; text-transform: uppercase;
    letter-spacing: .05em; color: #555;
  }}
  .narrative-body p {{ margin-bottom: 0.65rem; }}

  .narrative-body strong {{ font-weight: 700; color: #1a1a2e; }}
  .narrative-body em     {{ font-style: italic; color: #555; }}

  .narrative-body code {{
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 0.82em;
    background: #f0f2f5; border: 1px solid #dde;
    border-radius: 4px; padding: 1px 5px; color: #b8600a;
  }}
  .narrative-body pre {{
    background: #f8f9fb; border: 1px solid #dde; border-radius: 6px;
    padding: 0.8rem 1rem; overflow-x: auto; margin-bottom: 0.65rem;
  }}
  .narrative-body pre code {{
    background: none; border: none; padding: 0;
    font-size: 0.83em; color: #333;
  }}
  .narrative-body ul,
  .narrative-body ol {{ margin: 0.4rem 0 0.65rem 1.5rem; }}
  .narrative-body li  {{ margin-bottom: 0.2rem; }}

  .narrative-body table {{
    border-collapse: collapse; width: 100%;
    margin-bottom: 0.75rem; font-size: 0.85rem;
  }}
  .narrative-body th,
  .narrative-body td {{
    border: 1px solid #dde; padding: 0.4rem 0.7rem; text-align: left;
  }}
  .narrative-body th {{
    background: #f0f2f5; font-weight: 700; color: #1a1a2e;
  }}
  .narrative-body tr:nth-child(even) td {{ background: #fafbff; }}

  .narrative-body hr {{
    border: none; border-top: 1px solid #e8eaf0; margin: 1rem 0;
  }}
  .narrative-body blockquote {{
    border-left: 3px solid #c7d2fe; padding-left: 0.75rem;
    color: #555; margin: 0.5rem 0;
  }}
</style>
</head>
<body>

<h1>OMOP GNN XAI &mdash; Narrative Explanations</h1>
<p class="subtitle">
  {n_patients} patients &nbsp;&middot;&nbsp;
  {n_total} narratives &nbsp;&middot;&nbsp;
  E-TRANS model (AUROC&nbsp;0.909) &nbsp;&middot;&nbsp;
  MIMIC-IV &rarr; OMOP CDM
</p>

{"".join(patient_sections)}

<!-- Narrative Markdown stored as JSON — safe for all special characters -->
<script id="narrative-data" type="application/json">
{narratives_json}
</script>

<script>
const NARRATIVES = JSON.parse(
  document.getElementById('narrative-data').textContent
);

// Render Markdown into each narrative body
document.querySelectorAll('.md-render').forEach(el => {{
  const raw = NARRATIVES[el.dataset.tabId] || '';
  el.innerHTML = marked.parse(raw);
}});

// Collapse / expand card by clicking header
function toggleCard(header) {{
  header.closest('.patient-card').classList.toggle('collapsed');
}}

// Tab switching (guard: don't switch if card is collapsed)
function showTab(id) {{
  const card = document.getElementById(id).closest('.patient-card');
  if (card.classList.contains('collapsed')) {{
    card.classList.remove('collapsed');   // expand on tab click
  }}
  card.querySelectorAll('.tab-panel').forEach(p => p.style.display = 'none');
  card.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(id).style.display = 'block';
  card.querySelector('[data-tab="' + id + '"]').classList.add('active');
}}

// Activate first tab per card; collapse all except the first card
document.querySelectorAll('.patient-card').forEach((card, idx) => {{
  const firstPanel = card.querySelector('.tab-panel');
  const firstBtn   = card.querySelector('.tab-btn');
  if (firstPanel) firstPanel.style.display = 'block';
  if (firstBtn)   firstBtn.classList.add('active');
  if (idx > 0)    card.classList.add('collapsed');   // start collapsed
}});
</script>
</body>
</html>"""


def main():
    results = load_results()
    if not results:
        print(f"No result JSON files found in {RESULTS_DIR}/")
        return

    html = render_html(results)
    OUT_FILE.write_text(html, encoding="utf-8")
    print(f"Report written : {OUT_FILE.resolve()}")
    print(f"  {len(results)} patients, "
          f"{sum(len(v) for v in results.values())} narratives")

    import webbrowser
    webbrowser.open(OUT_FILE.resolve().as_uri())


if __name__ == "__main__":
    main()

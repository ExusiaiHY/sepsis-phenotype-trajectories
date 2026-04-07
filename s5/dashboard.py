"""
dashboard.py - Generate a bedside-ready HTML view for phenotype trajectory alerts.
"""
from __future__ import annotations

import json
from pathlib import Path


def render_clinical_dashboard_html(
    *,
    patient_id: str,
    snapshots: list[dict],
    output_path: Path,
    model_meta: dict | None = None,
) -> str:
    """Render a single-file HTML dashboard for local review or EMR embedding."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    current = snapshots[-1] if snapshots else {}
    risk_series = [float(item.get("risk_probability", 0.0)) for item in snapshots]
    phenotype_series = [item.get("phenotype") for item in snapshots]
    treatment_series = [item.get("top_treatment_signal", "none") for item in snapshots]
    current_status = _current_alert_status(current)

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ICU Sepsis Real-time Phenotype Monitor</title>
  <style>
    :root {{
      --ink: #0d2235;
      --muted: #4e687d;
      --paper: #f6f2e9;
      --card: rgba(255,255,255,0.78);
      --line: rgba(13,34,53,0.12);
      --accent: #c4552d;
      --accent-soft: #f0d2c7;
      --signal: #0f766e;
      --warn: #b45309;
      --danger: #b91c1c;
      --shadow: 0 18px 50px rgba(13,34,53,0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 10% 10%, rgba(196,85,45,0.14), transparent 26%),
        radial-gradient(circle at 90% 20%, rgba(15,118,110,0.12), transparent 28%),
        linear-gradient(180deg, #f8f3e6 0%, #ede4d2 100%);
      font-family: "Times New Roman", Times, serif;
    }}
    .shell {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 20px 48px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.5fr 1fr;
      gap: 18px;
      margin-bottom: 20px;
    }}
    .panel {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 22px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }}
    .title {{
      margin: 0 0 8px;
      font-family: "Times New Roman", Times, serif;
      font-size: 34px;
      line-height: 1.05;
    }}
    .subtitle {{
      color: var(--muted);
      margin: 0;
      line-height: 1.5;
    }}
    .status {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .metric {{
      padding: 16px;
      border-radius: 18px;
      background: rgba(255,255,255,0.78);
      border: 1px solid var(--line);
    }}
    .metric .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .metric .value {{
      margin-top: 8px;
      font-size: 28px;
      font-weight: 700;
    }}
    .risk-pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: {"rgba(185,28,28,0.12)" if current.get("risk_alert") else "rgba(15,118,110,0.12)"};
      color: {"var(--danger)" if current.get("risk_alert") else "var(--signal)"};
      font-weight: 700;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 18px;
    }}
    .timeline {{
      display: grid;
      gap: 12px;
      margin-top: 14px;
    }}
    .step {{
      display: grid;
      grid-template-columns: 90px 1fr;
      gap: 14px;
      padding: 14px;
      border-radius: 18px;
      background: rgba(255,255,255,0.68);
      border: 1px solid var(--line);
    }}
    .step .hour {{
      font-size: 13px;
      color: var(--muted);
      padding-top: 4px;
    }}
    .step strong {{
      display: block;
      font-size: 16px;
      margin-bottom: 4px;
    }}
    .chart {{
      display: flex;
      align-items: flex-end;
      gap: 8px;
      height: 180px;
      padding: 16px 0 4px;
    }}
    .bar {{
      flex: 1;
      min-width: 24px;
      border-radius: 14px 14px 8px 8px;
      background: linear-gradient(180deg, var(--accent), #eb9f5a);
      position: relative;
    }}
    .bar span {{
      position: absolute;
      bottom: -24px;
      left: 50%;
      transform: translateX(-50%);
      font-size: 11px;
      color: var(--muted);
    }}
    .meta {{
      margin-top: 14px;
      padding-top: 14px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    .workflow {{
      display: grid;
      gap: 12px;
      margin-top: 14px;
    }}
    .workflow-card {{
      padding: 16px;
      border-radius: 18px;
      background: linear-gradient(180deg, rgba(255,255,255,0.84), rgba(240,210,199,0.42));
      border: 1px solid var(--line);
    }}
    .workflow-card h3 {{
      margin: 0 0 6px;
      font-size: 16px;
    }}
    .workflow-card p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
      color: var(--muted);
    }}
    @media (max-width: 880px) {{
      .hero, .grid {{ grid-template-columns: 1fr; }}
      .status {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="panel">
        <h1 class="title">Sepsis Trajectory Monitor</h1>
        <p class="subtitle">Patient <strong>{patient_id}</strong> · bedside-ready rolling phenotype and mortality surveillance built on the distilled treatment-aware S1.5 backbone.</p>
        <div style="margin-top:14px"><span class="risk-pill">{current_status}</span></div>
        <div class="status">
          <div class="metric">
            <div class="label">Current Risk</div>
            <div class="value">{current.get("risk_probability", 0):.1%}</div>
          </div>
          <div class="metric">
            <div class="label">Current Phenotype</div>
            <div class="value">{current.get("phenotype", "N/A")}</div>
          </div>
          <div class="metric">
            <div class="label">Observed Hours</div>
            <div class="value">{current.get("hours_seen", 0)}</div>
          </div>
        </div>
      </div>
      <div class="panel">
        <h2 style="margin:0 0 8px;font-size:20px;">Deployment Snapshot</h2>
        <div class="meta">
          <div>CPU-quantized student for low-power ICU terminals.</div>
          <div>Outputs are formatted to align with rounds, order review, and handoff.</div>
          <div>All predictions are traceable to structured vitals, treatment events, and optional note embeddings.</div>
        </div>
        <div class="meta"><pre>{json.dumps(model_meta or {{}}, ensure_ascii=False, indent=2)}</pre></div>
      </div>
    </section>

    <section class="grid">
      <div class="panel">
        <h2 style="margin:0;font-size:22px;">Risk Trajectory</h2>
        <div class="chart">
          {_bars_html(risk_series)}
        </div>
        <div class="meta">The chart shows rolling risk updates from the online buffer. Deployment logic can delay firing until minimum history, consecutive high-risk persistence, and alert-budget constraints are satisfied.</div>
      </div>
      <div class="panel">
        <h2 style="margin:0;font-size:22px;">Workflow Fit</h2>
        <div class="workflow">
          <div class="workflow-card">
            <h3>Pre-round review</h3>
            <p>Show phenotype trend, current high-risk flag, and treatment exposures since the last shift without forcing extra clicks.</p>
          </div>
          <div class="workflow-card">
            <h3>Order entry support</h3>
            <p>Expose causal-analysis caveats next to phenotype-specific treatment hypotheses. Recommendations stay advisory, not prescriptive.</p>
          </div>
          <div class="workflow-card">
            <h3>Handoff trace</h3>
            <p>Persist model version, thresholds, and feature availability so downstream review remains auditable for quality and compliance.</p>
          </div>
        </div>
      </div>
    </section>

    <section class="panel" style="margin-top:18px;">
      <h2 style="margin:0;font-size:22px;">Rolling Timeline</h2>
      <div class="timeline">
        {_timeline_html(snapshots, phenotype_series, treatment_series)}
      </div>
    </section>
  </div>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    return html


def _bars_html(risk_series: list[float]) -> str:
    if not risk_series:
        return "<div class='bar' style='height:8%'><span>no data</span></div>"
    bars = []
    for idx, value in enumerate(risk_series):
        height = max(8.0, min(100.0, value * 100.0))
        bars.append(f"<div class='bar' style='height:{height:.1f}%'><span>t{idx}</span></div>")
    return "".join(bars)


def _timeline_html(snapshots: list[dict], phenotype_series: list, treatment_series: list) -> str:
    if not snapshots:
        return "<div class='step'><div class='hour'>No data</div><div><strong>Waiting for buffer fill</strong><div>No real-time updates have been recorded yet.</div></div></div>"
    rows = []
    for idx, snap in enumerate(snapshots):
        rows.append(
            f"""
            <div class="step">
              <div class="hour">Update {idx + 1}</div>
              <div>
                <strong>Risk {float(snap.get("risk_probability", 0.0)):.1%} · Phenotype {snap.get("phenotype", "N/A")}</strong>
                <div>Status: {_current_alert_status(snap)} · Dominant treatment signal: {treatment_series[idx] or "none"} · Hours seen: {snap.get("hours_seen", 0)}</div>
              </div>
            </div>
            """
        )
    return "".join(rows)


def _current_alert_status(snapshot: dict) -> str:
    if not snapshot:
        return "No current high-risk alert"
    if not snapshot.get("deployment_ready", True):
        return "Monitoring only"
    if snapshot.get("alert_event"):
        return "Alert event emitted"
    if snapshot.get("risk_alert"):
        return "High-risk state active"
    return "No current high-risk alert"

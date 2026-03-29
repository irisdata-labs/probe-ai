"""
HTML Report Export.

Renders an EvaluationReport as a self-contained HTML file.
No external dependencies — just open in a browser.

Usage:
    from agentprobe.engine.export import export_html
    export_html(report, "evaluation_report.html")
"""

from __future__ import annotations

import html
from collections import defaultdict

from agentprobe.engine.report import EvaluationReport


def export_html(report: EvaluationReport, path: str) -> str:
    """Export an EvaluationReport as a self-contained HTML file."""
    grade = report.resilience.get("grade", "?") if report.resilience else "?"
    grade_color = {"A": "#22c55e", "B": "#84cc16", "C": "#eab308", "D": "#f97316", "F": "#ef4444"}.get(grade, "#6b7280")
    resilience_score = report.resilience.get("overall", 0) if report.resilience else 0

    # Group verdicts by category
    by_category = defaultdict(list)
    for v in report.verdicts:
        by_category[v.category].append(v)

    # Build grouped verdict rows
    verdict_rows = ""
    for category in sorted(by_category.keys()):
        verdicts = by_category[category]
        cat_passed = sum(1 for v in verdicts if v.verdict == "passed")
        cat_degraded = sum(1 for v in verdicts if v.verdict == "degraded")
        cat_failed = sum(1 for v in verdicts if v.verdict == "failed")
        cat_total = len(verdicts)

        parts = []
        if cat_passed: parts.append(f'<span style="color:#22c55e">{cat_passed} passed</span>')
        if cat_degraded: parts.append(f'<span style="color:#d97706">{cat_degraded} degraded</span>')
        if cat_failed: parts.append(f'<span style="color:#ef4444">{cat_failed} failed</span>')
        summary_html = ", ".join(parts) if parts else f'<span style="color:#22c55e">{cat_total} passed</span>'

        verdict_rows += f"""<tr class="cat-header">
            <td colspan="5"><strong>{html.escape(category)}</strong>
            <span style="float:right;font-weight:600">{summary_html}</span></td>
        </tr>\n"""

        for v in verdicts:
            if v.verdict == "passed":
                status_icon = "&#10003;"
                status_class = "pass"
            elif v.verdict == "degraded":
                status_icon = "&#9888;"
                status_class = "degraded"
            else:
                status_icon = "&#10007;"
                status_class = "fail"

            dest = html.escape(v.destination or v.customer_message[:40])
            diff = v.difficulty.level if v.difficulty else "?"
            diff_color = {"easy": "#22c55e", "medium": "#eab308", "hard": "#f97316", "adversarial": "#ef4444"}.get(diff, "#6b7280")

            # Chaos column
            if v.tool_failures:
                chaos_parts = []
                for tf in v.tool_failures:
                    parts = tf.split(":")
                    tool = parts[0] if parts else tf
                    mode = parts[1] if len(parts) > 1 else ""
                    if mode:
                        chaos_parts.append(f'<span class="chaos-tag">{html.escape(tool)}<span class="chaos-mode">:{html.escape(mode)}</span></span>')
                    else:
                        chaos_parts.append(f'<span class="chaos-tag">{html.escape(tool)}</span>')
                chaos_html = " ".join(chaos_parts)
            else:
                chaos_html = '<span style="color:#cbd5e1">none</span>'

            # Result column
            if v.verdict == "passed":
                result_html = '<span class="tag tag-pass">passed</span>'
            elif v.verdict == "degraded":
                result_html = '<span class="tag tag-degraded">degraded</span>'
                # Build specific reason from which tools failed
                if v.tool_failures:
                    failed_tools = [tf.split(":")[0] for tf in v.tool_failures]
                    modes = [tf.split(":")[1] if ":" in tf else "failed" for tf in v.tool_failures]
                    details = ", ".join(f"{t} ({m})" for t, m in zip(failed_tools, modes))
                    result_html += f'<div class="reason">{html.escape(details)}</div>'
                else:
                    result_html += '<div class="reason">Output incomplete due to tool failures</div>'
            else:
                tag_label = v.failure_tag or "failed"
                result_html = f'<span class="tag tag-fail">{html.escape(tag_label)}</span>'
                if v.failure_reason:
                    result_html += f'<div class="reason">{html.escape(v.failure_reason[:80])}</div>'

            verdict_rows += f"""<tr class="{status_class}">
                <td class="status-cell">{status_icon}</td>
                <td>{dest}</td>
                <td><span style="color:{diff_color}">{diff}</span></td>
                <td>{chaos_html}</td>
                <td>{result_html}</td>
            </tr>\n"""

    # Difficulty distribution — compute per-level verdict counts from verdicts
    diff_dist = report.difficulty_summary.get("distribution", {})
    total_scenarios = report.total_scenarios or 1

    # Build per-level counts from verdicts
    diff_verdicts = {}
    for v in report.verdicts:
        level = v.difficulty.level if v.difficulty else "easy"
        if level not in diff_verdicts:
            diff_verdicts[level] = {"passed": 0, "degraded": 0, "failed": 0}
        diff_verdicts[level][v.verdict] += 1

    diff_bars = ""
    for level in ["easy", "medium", "hard", "adversarial"]:
        count = diff_dist.get(level, 0)
        if count == 0:
            continue
        rate = report.pass_rate_by_difficulty.get(level, 0)
        dv = diff_verdicts.get(level, {"passed": 0, "degraded": 0, "failed": 0})
        color = {"easy": "#22c55e", "medium": "#eab308", "hard": "#f97316", "adversarial": "#ef4444"}.get(level, "#6b7280")
        pct = round(count / total_scenarios * 100)

        parts = []
        if dv["passed"]: parts.append(f'<span style="color:#22c55e">{dv["passed"]} passed</span>')
        if dv["degraded"]: parts.append(f'<span style="color:#d97706">{dv["degraded"]} degraded</span>')
        if dv["failed"]: parts.append(f'<span style="color:#ef4444">{dv["failed"]} failed</span>')
        breakdown = ", ".join(parts) if parts else "0 scenarios"

        diff_bars += f"""<div class="diff-row">
            <span class="diff-label">{level}</span>
            <div class="diff-bar-bg"><div class="diff-bar" style="width:{pct}%;background:{color}"></div></div>
            <span class="diff-stat">{count} scenarios &mdash; {breakdown}</span>
        </div>\n"""

    # Failure by tag
    tag_rows = ""
    for tag, count in sorted(report.failures_by_tag.items(), key=lambda x: -x[1]):
        tag_rows += f"<tr><td>{html.escape(tag)}</td><td>{count}</td></tr>\n"

    # Cluster rows
    cluster_rows = ""
    for c in report.clusters:
        rules = ", ".join(c.get("rules", [])) if c.get("rules") else "&mdash;"
        tools = ", ".join(c.get("tools", [])) if c.get("tools") else "&mdash;"
        cluster_rows += f"""<tr>
            <td>#{c['id']}</td><td>{c['count']}</td>
            <td>{html.escape(rules)}</td>
            <td>{html.escape(tools)}</td>
            <td>{html.escape(c.get('fix', '') or '')}</td>
        </tr>\n"""

    # Pass rate color
    pr_color = '#22c55e' if report.pass_rate >= 90 else '#eab308' if report.pass_rate >= 70 else '#ef4444'

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>probe-ai &mdash; Evaluation Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f8fafc; color: #1e293b; padding: 24px; line-height: 1.5; }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ font-size: 24px; margin-bottom: 4px; }}
  h2 {{ font-size: 18px; margin: 28px 0 12px; color: #475569; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; }}
  .subtitle {{ color: #64748b; margin-bottom: 20px; }}

  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; margin: 16px 0; }}
  .card {{ background: white; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .card .label {{ font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }}
  .card .value {{ font-size: 28px; font-weight: 700; margin-top: 4px; }}
  .card .detail {{ font-size: 12px; color: #94a3b8; margin-top: 2px; }}
  .grade {{ display: inline-block; width: 48px; height: 48px; border-radius: 50%; color: white; font-size: 24px; font-weight: 700; text-align: center; line-height: 48px; }}

  .legend {{ background: white; border-radius: 8px; padding: 14px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin: 16px 0; font-size: 13px; color: #475569; display: flex; gap: 24px; flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}

  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin: 12px 0; }}
  th {{ background: #f1f5f9; padding: 10px 12px; text-align: left; font-size: 11px; text-transform: uppercase; color: #64748b; letter-spacing: 0.3px; }}
  td {{ padding: 8px 12px; border-top: 1px solid #f1f5f9; font-size: 13px; vertical-align: top; }}
  tr.fail {{ background: #fef2f2; }}
  tr.degraded {{ background: #fffbeb; }}
  tr.pass {{ }}
  tr.cat-header {{ background: #f8fafc; }}
  tr.cat-header td {{ padding: 10px 12px; border-top: 2px solid #e2e8f0; font-size: 14px; }}
  tr:hover:not(.cat-header) {{ background: #f1f5f9; }}

  .status-cell {{ font-size: 16px; width: 30px; text-align: center; }}
  tr.pass .status-cell {{ color: #22c55e; }}
  tr.degraded .status-cell {{ color: #d97706; }}
  tr.fail .status-cell {{ color: #ef4444; }}

  .tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; white-space: nowrap; }}
  .tag-pass {{ background: #dcfce7; color: #166534; }}
  .tag-degraded {{ background: #fef3c7; color: #92400e; }}
  .tag-fail {{ background: #fee2e2; color: #991b1b; }}

  .chaos-tag {{ display: inline-block; background: #f1f5f9; color: #64748b; padding: 1px 6px; border-radius: 3px; font-size: 11px; margin: 1px; }}
  .chaos-mode {{ color: #94a3b8; }}

  .reason {{ font-size: 11px; color: #64748b; margin-top: 3px; }}
  tr.fail .reason {{ color: #991b1b; }}

  .diff-row {{ display: flex; align-items: center; margin: 6px 0; }}
  .diff-label {{ width: 90px; font-size: 13px; font-weight: 500; }}
  .diff-bar-bg {{ flex: 1; height: 22px; background: #f1f5f9; border-radius: 4px; overflow: hidden; margin: 0 12px; }}
  .diff-bar {{ height: 100%; border-radius: 4px; }}
  .diff-stat {{ font-size: 12px; color: #64748b; white-space: nowrap; }}

  .footer {{ margin-top: 32px; padding-top: 16px; border-top: 1px solid #e2e8f0; color: #94a3b8; font-size: 12px; }}
</style>
</head>
<body>
<div class="container">
  <h1>probe-ai Evaluation Report</h1>
  <p class="subtitle">{html.escape(report.agent_name or 'Agent')} &mdash; {html.escape(report.plan_name or 'Evaluation')} &mdash; {report.generated_at[:10]}</p>

  <div class="cards">
    <div class="card">
      <div class="label">Scenarios</div>
      <div class="value">{report.total_scenarios}</div>
    </div>
    <div class="card">
      <div class="label">Passed</div>
      <div class="value" style="color:#22c55e">{report.total_passed}</div>
      <div class="detail">complete, correct output</div>
    </div>
    <div class="card">
      <div class="label">Degraded</div>
      <div class="value" style="color:#d97706">{report.total_degraded}</div>
      <div class="detail">tools failed, agent survived</div>
    </div>
    <div class="card">
      <div class="label">Failed</div>
      <div class="value" style="color:#ef4444">{report.total_failed}</div>
      <div class="detail">crash, policy violation, or bad output</div>
    </div>
    <div class="card">
      <div class="label">Pass Rate</div>
      <div class="value" style="color:{pr_color}">{report.pass_rate}%</div>
      <div class="detail">excludes degraded</div>
    </div>
    <div class="card">
      <div class="label">Resilience</div>
      <div class="value"><span class="grade" style="background:{grade_color}">{grade}</span></div>
      <div class="detail">{resilience_score:.0f}/100</div>
    </div>
    <div class="card">
      <div class="label">Avg Difficulty</div>
      <div class="value">{report.difficulty_summary.get('average_difficulty', 0):.0f}</div>
      <div class="detail">out of 100</div>
    </div>
  </div>

  <h2>Difficulty Distribution</h2>
  <div style="background:white;padding:16px;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.08)">
    {diff_bars}
  </div>

  {"<h2>Failures by Type</h2><table><tr><th>Type</th><th>Count</th></tr>" + tag_rows + "</table>" if tag_rows else ""}

  {"<h2>Failure Clusters</h2><table><tr><th>#</th><th>Count</th><th>Rules Violated</th><th>Tools Involved</th><th>Suggested Fix</th></tr>" + cluster_rows + "</table>" if cluster_rows else ""}

  <h2>Scenarios by Category</h2>

  <div class="legend">
    <div class="legend-item"><span style="color:#22c55e;font-size:16px">&#10003;</span> <strong>Passed</strong> &mdash; correct, complete output</div>
    <div class="legend-item"><span style="color:#d97706;font-size:16px">&#9888;</span> <strong>Degraded</strong> &mdash; tools failed, agent didn't crash but output is incomplete</div>
    <div class="legend-item"><span style="color:#ef4444;font-size:16px">&#10007;</span> <strong>Failed</strong> &mdash; crash, policy violation, or bad output</div>
  </div>

  <table>
    <tr><th style="width:30px"></th><th>Input</th><th>Difficulty</th><th>Chaos Injected</th><th>Result</th></tr>
    {verdict_rows}
  </table>

  <div class="footer">
    Generated by <strong>probe-ai</strong> &mdash; Chaos engineering for AI agents
  </div>
</div>
</body>
</html>"""

    with open(path, "w") as f:
        f.write(html_content)
    return path

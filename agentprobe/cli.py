"""
CLI for viewing and inspecting agent traces.

Commands:
    agentprobe list <directory>       List all trace files
    agentprobe view <file>            View a single trace in detail
    agentprobe summary <session_dir>  View session aggregate stats
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional


def _fmt_ms(ms: Optional[float]) -> str:
    if ms is None:
        return "?"
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


def _status_icon(status: Optional[str]) -> str:
    return "\u2713" if status == "ok" else "\u2717" if status == "error" else "?"


def _step_icon(span_type: Optional[str]) -> str:
    icons = {"llm_call": "\u2605", "tool_call": "\u2692", "reasoning": "\u2699",
             "retrieval": "\u2315", "generic": "\u2022"}
    return icons.get(span_type or "", "\u2022")


# ============================================================================
# Commands
# ============================================================================

def cmd_view(args: argparse.Namespace) -> None:
    """View a single trace file."""
    with open(args.file) as f:
        data = json.load(f)

    print(f"\n{'=' * 60}")
    print(f"  Trace: {data.get('agent_name', '?')}  ({data.get('trace_id', '?')[:12]})")
    print(f"{'=' * 60}")

    status = data.get("status", "?")
    print(f"  Status:   {_status_icon(status)} {status}")
    print(f"  Duration: {_fmt_ms(data.get('duration_ms'))}")

    summary = data.get("summary", {})
    print(f"  Steps:    {summary.get('step_count', 0)}")
    print(f"  LLM:      {summary.get('llm_calls', 0)} calls, "
          f"{summary.get('total_tokens', 0)} tokens")
    print(f"  Tools:    {summary.get('tool_calls', 0)} calls")

    if data.get("tags"):
        print(f"  Tags:     {', '.join(data['tags'])}")
    if data.get("thread_id"):
        print(f"  Thread:   {data['thread_id']}")

    # Input/Output
    input_data = data.get("input_data")
    if input_data:
        inp_str = json.dumps(input_data, default=str)
        if len(inp_str) > 200:
            inp_str = inp_str[:200] + "..."
        print(f"\n  Input:  {inp_str}")

    output_data = data.get("output_data")
    if output_data:
        out_str = json.dumps(output_data, default=str)
        if len(out_str) > 200:
            out_str = out_str[:200] + "..."
        print(f"  Output: {out_str}")

    # Steps
    spans = data.get("spans", [])
    if spans:
        print(f"\n  {'Steps':}")
        print(f"  {'-' * 50}")
        for span in spans:
            icon = _step_icon(span.get("span_type"))
            dur = _fmt_ms(span.get("duration_ms"))
            detail = ""
            llm = span.get("llm", {})
            tool = span.get("tool", {})
            if llm and llm.get("model"):
                detail = f" model={llm['model']}"
            elif tool and tool.get("tool_name"):
                detail = f" tool={tool['tool_name']}"
            parent = f" (child of {span.get('parent_span_id', '')[:8]})" if span.get("parent_span_id") else ""
            print(f"  {icon} {span.get('name', '?'):24s} [{span.get('span_type', '?'):10s}] "
                  f"{dur:>8s}{detail}{parent}")

    # State changes
    state_changes = data.get("state_changes", [])
    if state_changes:
        print(f"\n  State Changes ({len(state_changes)}):")
        for sc in state_changes:
            print(f"    {sc['key']}: {sc.get('before')} \u2192 {sc.get('after')} "
                  f"(in {sc.get('step_name', '?')})")

    # Decisions
    decisions = data.get("decisions", [])
    if decisions:
        print(f"\n  Decisions ({len(decisions)}):")
        for dec in decisions:
            alts = ", ".join(dec.get("alternatives", []))
            print(f"    chose={dec['chosen']} (alternatives: {alts or 'none'}) "
                  f"in {dec.get('step_name', '?')}")

    # Error
    exc = data.get("exception_info")
    if exc:
        print(f"\n  ERROR: {exc.get('type', '?')}: {exc.get('message', '?')}")
        if args.verbose:
            print(f"  {exc.get('traceback', '')}")

    # Verbose: raw JSON
    if args.verbose:
        print(f"\n  {'Raw JSON':}")
        print(f"  {'-' * 50}")
        print(json.dumps(data, indent=2, default=str))

    print()


def cmd_list(args: argparse.Namespace) -> None:
    """List all trace files in a directory."""
    directory = args.directory

    if not os.path.isdir(directory):
        print(f"Not a directory: {directory}")
        sys.exit(1)

    files = sorted(f for f in os.listdir(directory) if f.endswith(".json") and not f.startswith("_"))

    if not files:
        print(f"No trace files found in {directory}")
        return

    print(f"\n{'File':<40s} {'Agent':<20s} {'Status':>8s} {'Duration':>10s} {'Steps':>6s}")
    print("-" * 90)

    for filename in files:
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath) as f:
                data = json.load(f)
            agent = data.get("agent_name", "?")
            status = data.get("status", "?")
            dur = _fmt_ms(data.get("duration_ms"))
            steps = str(data.get("summary", {}).get("step_count", "?"))
            print(f"{filename:<40s} {agent:<20s} {_status_icon(status) + ' ' + status:>8s} "
                  f"{dur:>10s} {steps:>6s}")
        except (json.JSONDecodeError, KeyError):
            print(f"{filename:<40s} {'(invalid)':>20s}")

    print()


def cmd_summary(args: argparse.Namespace) -> None:
    """View session summary."""
    session_dir = args.session_dir
    index_path = os.path.join(session_dir, "_session.json")

    if not os.path.exists(index_path):
        print(f"No _session.json found in {session_dir}")
        sys.exit(1)

    with open(index_path) as f:
        index = json.load(f)

    summary = index.get("summary", {})
    traces = index.get("traces", [])

    print(f"\n{'=' * 50}")
    print(f"  Session: {index.get('name', index.get('session_id', '?'))}")
    print(f"{'=' * 50}")

    print(f"  Total traces:  {summary.get('total_traces', 0)}")
    counts = summary.get("status_counts", {})
    print(f"  Status:        {counts.get('ok', 0)} ok, {counts.get('error', 0)} error")
    print(f"  Pass rate:     {summary.get('pass_rate', 0):.0%}")
    print(f"  Avg duration:  {_fmt_ms(summary.get('avg_duration_ms'))}")
    print(f"  Avg steps:     {summary.get('avg_steps', 0):.1f}")
    print(f"  Total tokens:  {summary.get('total_tokens', 0)}")
    print(f"  LLM calls:     {summary.get('total_llm_calls', 0)}")
    print(f"  Tool calls:    {summary.get('total_tool_calls', 0)}")

    if traces:
        print(f"\n  Traces:")
        for t in traces:
            icon = _status_icon(t.get("status"))
            dur = _fmt_ms(t.get("duration_ms"))
            print(f"    {icon} {t.get('agent_name', '?')} ({t.get('trace_id', '?')[:8]}) — {dur}")

    print()


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agentprobe",
        description="Inspect and manage AI agent traces",
    )
    subparsers = parser.add_subparsers(dest="command")

    # view
    view_parser = subparsers.add_parser("view", help="View a single trace file")
    view_parser.add_argument("file", help="Path to trace JSON file")
    view_parser.add_argument("-v", "--verbose", action="store_true",
                             help="Show raw JSON and full tracebacks")

    # list
    list_parser = subparsers.add_parser("list", help="List trace files in a directory")
    list_parser.add_argument("directory", help="Directory containing trace JSON files")

    # summary
    summary_parser = subparsers.add_parser("summary", help="View session summary")
    summary_parser.add_argument("session_dir", help="Path to saved session directory")

    args = parser.parse_args()

    if args.command == "view":
        cmd_view(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "summary":
        cmd_summary(args)
    else:
        parser.print_help()

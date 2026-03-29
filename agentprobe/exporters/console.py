"""Console exporter — prints trace summaries to stdout."""

from agentprobe.models import TraceRecord


class ConsoleExporter:
    """Prints human-readable trace summaries to the terminal."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def export(self, trace: TraceRecord) -> None:
        icon = "\u2713" if trace.status == "ok" else "\u2717"
        tid = trace.trace_id[:8]
        dur = f"{trace.duration_ms:.0f}ms" if trace.duration_ms else "?"
        toks = trace.total_tokens

        print(f"[{icon}] {trace.agent_name} ({tid}) "
              f"| {dur} | {trace.step_count} steps "
              f"| {len(trace.llm_calls)} llm | {len(trace.tool_calls)} tool "
              f"| {toks} tokens")

        if self.verbose:
            for span in trace.spans:
                s_dur = f"{span.duration_ms:.0f}ms" if span.duration_ms else "?"
                detail = ""
                if span.model:
                    detail = f" model={span.model}"
                elif span.tool_name:
                    detail = f" tool={span.tool_name}"
                print(f"  [{span.span_type}] {span.name} — {s_dur}{detail}")

            if trace.state_changes:
                print(f"  State changes: {len(trace.state_changes)}")
                for sc in trace.state_changes:
                    print(f"    {sc.key}: {sc.before} → {sc.after}")

            if trace.decisions:
                print(f"  Decisions: {len(trace.decisions)}")
                for dec in trace.decisions:
                    alts = ", ".join(dec.alternatives) if dec.alternatives else "none"
                    print(f"    chose={dec.chosen} (alternatives: {alts})")

        if trace.status == "error" and trace.exception_info:
            print(f"  ERROR: {trace.exception_info['type']}: "
                  f"{trace.exception_info['message']}")

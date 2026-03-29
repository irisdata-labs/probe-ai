"""JSON file exporter — writes each trace as a separate JSON file."""

import json
import os
from agentprobe.models import TraceRecord


class JSONFileExporter:
    """Writes each trace as a JSON file to a directory."""

    def __init__(self, output_dir: str = "traces"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export(self, trace: TraceRecord) -> str:
        filename = f"{trace.agent_name}_{trace.trace_id[:8]}.json"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(trace.to_dict(), f, indent=2, default=str)
        return filepath

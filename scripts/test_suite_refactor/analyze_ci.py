"""Parse timestamped gh job logs, retaining interrupted tests and timing uncertainty."""

import collections
import datetime as dt
import json
import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[2] / "reports/test-suite-refactor"
rows = []
for path in sorted(ROOT.glob("ci-*.log")):
    content = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", path.read_text(encoding="utf-8-sig"))
    events = []
    current = None
    for line in content.splitlines():
        m = re.match(r"(\d{4}-\S+) (.*)", line)
        if not m:
            continue
        timestamp = dt.datetime.fromisoformat(m[1]).timestamp()
        text = m[2]
        n = re.match(r"(tests/\S+::.*?)(?:\s+(PASSED|FAILED|SKIPPED|ERROR).*)?\s*$", text)
        if n:
            current = {"nodeid": n[1], "start_or_inline_end": timestamp, "status": n[2]}
            events.append(current)
        elif current:
            status = re.match(r"(PASSED|FAILED|SKIPPED|ERROR)\b", text)
            if status:
                current["status"] = status[1]
        if current and re.match(r"(PASSED|FAILED|SKIPPED|ERROR)\b", text):
            current["end"] = timestamp
    modules = collections.defaultdict(lambda: {"interval_seconds": 0.0, "cases": 0, "failures": 0})
    for i, event in enumerate(events):
        module = event["nodeid"].split("::")[0]
        data = modules[module]
        data["cases"] += 1
        data["failures"] += event["status"] in ("FAILED", "ERROR")
        if i + 1 < len(events):
            data["interval_seconds"] += events[i + 1]["start_or_inline_end"] - event["start_or_inline_end"]
    rows.append(
        {
            "platform": path.stem.removeprefix("ci-"),
            "collected": 4101,
            "observed": len(events),
            "completed": sum(bool(e["status"]) for e in events),
            "failures": sum(e["status"] in ("FAILED", "ERROR") for e in events),
            "observed_span_seconds": events[-1]["start_or_inline_end"] - events[0]["start_or_inline_end"],
            "last_node": events[-1],
            "modules": dict(sorted(modules.items(), key=lambda item: -item[1]["interval_seconds"])),
        }
    )
(ROOT / "ci-timings.json").write_text(
    json.dumps(
        {
            "method": (
                "Adjacent test log events; includes setup/call/teardown and logging; "
                "inline completion vs separate start makes boundaries approximate. "
                "Final in-flight test excluded. Not pytest phase timings."
            ),
            "runs": rows,
        },
        indent=2,
    )
)
for row in rows:
    print(
        row["platform"],
        "completed",
        row["completed"],
        "failures",
        row["failures"],
        "observed seconds",
        round(row["observed_span_seconds"]),
    )
    for name, data in list(row["modules"].items())[:12]:
        print("  ", round(data["interval_seconds"], 1), data["cases"], name)

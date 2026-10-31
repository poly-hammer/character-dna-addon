"""Compare the production JSON sanitizer with a bounded-token experiment.

Input is SDK JSON retained by profile_document_io.py --capture-json. This does
not patch the addon. It checks byte-for-byte equivalence before reporting time.
"""

import argparse
import hashlib
import json
import re
import time

from pathlib import Path


LEGACY = rb'"(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?[eE]-\d+'
BOUNDED = rb'"(?:[^"\\]|\\.)*"|(?<![0-9.])-?\d+(?:\.\d+)?[eE]-\d+'


def normalize(match: re.Match) -> bytes:
    token = match.group()
    if token.startswith(b'"'):
        return token
    return b"0.0" if 0 < abs(float(token)) < 2.0**-126 else token


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    # Preserve escaped strings, ordinary numbers, and signs; normalize only the
    # same subnormal tokens as the production sanitizer.
    fixtures = [
        b'{"n":[1e-40,-1e-40,1.17549435e-38,0.12345678912345678,-2.1e-6,0,1e+2]}',
        b'{"text":"1e-40 \\"still quoted\\" -1e-40","n":-2.5e-43}',
        b'{"names":["DNA_body::x_1e-40","1.1.0","\\\\","\\u00e9"]}',
    ]
    for fixture in fixtures:
        assert re.sub(LEGACY, normalize, fixture) == re.sub(BOUNDED, normalize, fixture)
    data = args.input.read_bytes()
    result = {"input_bytes": len(data), "input_sha256": hashlib.sha256(data).hexdigest(), "samples": []}
    reference = None
    for name, pattern in [("legacy", LEGACY)] + [("bounded_experiment", BOUNDED)] * 3:
        start = time.perf_counter()
        output = re.sub(pattern, normalize, data)
        wall = time.perf_counter() - start
        if reference is None:
            reference = output
        assert output == reference
        result["samples"].append({"name": name, "wall_s": wall})
        print(name, wall, flush=True)
    result["byte_identical_to_production"] = True
    result["normalized_sha256"] = hashlib.sha256(reference).hexdigest()
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

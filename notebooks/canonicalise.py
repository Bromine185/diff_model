"""Canonicalise an executed notebook so that two seeded runs compare byte for byte.

`CLAUDE.md` non-negotiable #4 asks that a seeded run twice produce bit-identical output.
`01_wavelet_ddpm.ipynb` satisfies that for everything it *prints* — no wall clock reaches
stored output — but `jupyter nbconvert --execute` adds two things to the file that are
transport artifacts rather than notebook content, and both vary with timing:

1. **`cell.metadata.execution`** — four ISO timestamps per code cell, recording when the
   kernel started and finished it. Nothing reads them.
2. **stdout chunking** — nbclient coalesces the stream messages that happen to arrive
   together, so the same printed text lands as one `stream` output on one run and two on
   the next. Identical characters, different number of records.

Strip the first, merge the second, and a re-execution is byte-identical to the committed
file. Run it on both sides of the comparison:

    jupyter nbconvert --to notebook --execute \\
        --output /tmp/rerun.ipynb notebooks/01_wavelet_ddpm.ipynb
    python notebooks/canonicalise.py /tmp/rerun.ipynb
    cmp /tmp/rerun.ipynb notebooks/01_wavelet_ddpm.ipynb

This touches only *how* outputs are recorded. It never edits output text, never edits cell
source, and never removes an output — so it cannot paper over a numeric difference, which
is the only kind of difference the check is looking for.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat


def canonicalise(nb: dict) -> tuple[int, int]:
    """Strip execution timestamps and merge adjacent same-stream outputs, in place."""
    stripped = merged = 0
    for cell in nb["cells"]:
        if cell.get("metadata", {}).pop("execution", None) is not None:
            stripped += 1

        outputs = cell.get("outputs")
        if not outputs:
            continue
        kept: list[dict] = []
        for out in outputs:
            same_stream = (
                kept
                and out.get("output_type") == "stream"
                and kept[-1].get("output_type") == "stream"
                and kept[-1].get("name") == out.get("name")
            )
            if same_stream:
                prev = kept[-1]
                as_list = lambda t: t if isinstance(t, list) else [t]  # noqa: E731
                prev["text"] = as_list(prev["text"]) + as_list(out["text"])
                merged += 1
            else:
                kept.append(out)
        cell["outputs"] = kept
    return stripped, merged


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[0])
        print(f"usage: {Path(argv[0]).name} NOTEBOOK.ipynb", file=sys.stderr)
        return 2
    path = Path(argv[1])
    nb = json.loads(path.read_text())
    stripped, merged = canonicalise(nb)
    nbformat.write(nbformat.from_dict(nb), str(path))
    print(f"{path.name}: stripped {stripped} execution-metadata blocks, "
          f"merged {merged} stream records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

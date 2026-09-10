#!/usr/bin/env bash
set -euo pipefail
softmax_source="$(cd -- "$(dirname -- "$0")/.." && pwd)"
python3 - "$softmax_source/static" "${1:?expected output bundle directory}" <<'PY'
import shutil
import sys
from pathlib import Path
source, output = (Path(arg).resolve() for arg in sys.argv[1:])
# This hook owns only the generated output, never any source directory.
if output == source or output in source.parents or source in output.parents:
    raise SystemExit('refusing to overwrite replay sources')
if output.exists():
    shutil.rmtree(output)
shutil.copytree(source, output)
PY

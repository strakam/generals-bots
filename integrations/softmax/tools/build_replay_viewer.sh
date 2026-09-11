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
assets = source.parents[2] / 'generals/assets/images'
(output / 'assets').mkdir()
for name in ('crownie.png', 'citie.png', 'mountainie.png'):
    shutil.copyfile(assets / name, output / 'assets' / name)
fonts = source.parents[2] / 'generals/assets/fonts'
(output / 'fonts').mkdir()
for name in ('Quicksand-VariableFont_wght.ttf', 'OFL.txt'):
    shutil.copyfile(fonts / name, output / 'fonts' / name)
PY

#!/bin/bash
# Print the campaign rows that have no successful result yet, space-separated.
#
# A row that FAILED leaves a record at the same path a successful one does, so
# "missing" here means what `--skip-existing` means: no file, or a file carrying
# an `error`. Feed it straight to the array:
#
#   ROWS="$(bash missing_rows.sh)" sbatch coverage_64_gpu_array.sh
set -uo pipefail

SUBMISSION_REPO=${SUBMISSION_REPO:-/lustre/fswork/projects/rech/ney/ulx23va/projects/radio/repos/submission-scripts}
RADIO_ROOT=${RADIO_ROOT:-/lustre/fswork/projects/rech/ney/ulx23va/projects/radio}
CODE_REPO=${CODE_REPO:-$RADIO_ROOT/repos/UQsuite}
CONFIG=${CONFIG:-$SUBMISSION_REPO/jean_zay/raw/equivariant_bootstrap/configs/coverage_64_gpu.yaml}
OUT_DIR=${OUT_DIR:-$RADIO_ROOT/outputs/uq_campaign/coverage_64_gpu}

python - "$CONFIG" "$OUT_DIR" "$CODE_REPO" <<'PY'
import pathlib, sys
config, out_dir, code_repo = (pathlib.Path(a) for a in sys.argv[1:4])
sys.path.insert(0, str(code_repo / "scripts"))
import yaml
from uq_campaign import row_completed

cfg = yaml.safe_load(config.read_text())
todo = [
    f"{family}/{arm}/{seed}"
    for family in cfg["families"]
    for arm in cfg["arms"]
    for seed in cfg["seeds"]
    if not row_completed(out_dir / family / f"{arm}_seed{seed}.json")
]
print(" ".join(todo))
PY

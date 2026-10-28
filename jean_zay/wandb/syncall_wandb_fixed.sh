#!/bin/bash
# Sync offline wandb runs, working around the upstream bug that made every
# previous sync report success and upload nothing. See wandb_fixed.py for the
# mechanism; the two things that matter here:
#
#   * the upload is routed through `wandb_fixed.py`, which patches the
#     null-handling bug in `upsert_run`'s server probe;
#   * `--include-synced` is passed, because the failed syncs still wrote
#     `<run>.wandb.synced` (that marker records "the local run finished", not
#     "the upload worked"), so without it every affected run is skipped.
#
# RUN THIS ON A LOGIN NODE. Compute nodes have no outbound network.
#
#   ./syncall_wandb_fixed.sh <wandb-dir> [glob]
#
#   ./syncall_wandb_fixed.sh $RADIO/outputs/trained_models/wandb
#   ./syncall_wandb_fixed.sh $RADIO/outputs/trained_models/wandb 'offline-run-202610*'
#
# The glob is worth using. That directory accumulates every offline run ever
# written there -- 49 of them as of 2026-10-28, going back to August across
# several different experiments -- and syncing the lot publishes 49 runs. Narrow
# it to what you actually want on the site.
#
# Unlike the original, this does not trust the exit status alone: `wandb sync`
# exits 0 on the failure described above, so each run is checked for a project
# URL in the output and the script reports a per-run verdict and a final count.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WANDB_DIR="${1:-}"
PATTERN="${2:-offline-run-*}"

if [ -z "$WANDB_DIR" ] || [ ! -d "$WANDB_DIR" ]; then
    echo "usage: $0 <wandb-dir> [glob]" >&2
    echo "  <wandb-dir> must be the directory holding the offline-run-* dirs" >&2
    exit 2
fi

module purge
module load pytorch-gpu/py3/2.7.0

if [ ! -f "$HOME/.netrc" ] || ! grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    echo "[sync] no api.wandb.ai credentials in ~/.netrc; run \`wandb login\` first." >&2
    exit 3
fi

shopt -s nullglob
runs=("$WANDB_DIR"/$PATTERN)
shopt -u nullglob
if [ ${#runs[@]} -eq 0 ]; then
    echo "[sync] no runs matching '$PATTERN' under $WANDB_DIR" >&2
    exit 4
fi

echo "[sync] $WANDB_DIR"
echo "[sync] ${#runs[@]} run(s) matching '$PATTERN'"
echo

ok=0
bad=0
for run in "${runs[@]}"; do
    [ -d "$run" ] || continue
    name="$(basename "$run")"
    out="$(python "$HERE/wandb_fixed.py" sync --include-synced "$run" 2>&1)"
    # A real upload prints the run's URL. The bug's signature is the CommError
    # in the output with no URL, while still exiting 0.
    if grep -qE "https://wandb\.ai/|\.wandb\.ai/" <<<"$out"; then
        url="$(grep -oE "https://[^ ]*wandb\.ai/[^ ]*" <<<"$out" | head -1)"
        echo "  OK    $name  ->  $url"
        ok=$((ok + 1))
    else
        echo "  FAIL  $name"
        grep -iE "error|commerror|exception" <<<"$out" | head -2 | sed 's/^/          /'
        bad=$((bad + 1))
    fi
    sleep 1
done

echo
echo "[sync] $ok uploaded, $bad failed, of ${#runs[@]}"
[ "$bad" -eq 0 ] || exit 1

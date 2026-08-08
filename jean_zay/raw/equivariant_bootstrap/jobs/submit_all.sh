#!/bin/bash
# Submit the whole campaign, in the order the results are useful in.
# Run on a LOGIN NODE after prepare_cluster.sh has passed.
#
#   bash submit_all.sh          # submits everything
#   bash submit_all.sh level_b  # just one
#
# level_b is deliberately first and unchained: it is minutes of compute and it
# can descope the two 360^2 jobs before they consume anything.

set -euo pipefail
cd "$(dirname "$0")"

JOBS=${@:-level_b campaign_64 campaign_360_ar campaign_360_unrolled}

for job in $JOBS; do
    id=$(sbatch --parsable "$job.sh")
    echo "submitted $job.sh -> $id"
done

cat <<'EOF'

Watch with:   squeue -u $USER
Logs in:      ./logs/R-<jobname>_<jobid>.out

campaign_* jobs self-chain: each queues its successor before starting, so
`squeue` will show a pending job per campaign. That is expected, not a
double submission. Cancel the pending one to stop a chain.
EOF

#!/usr/bin/env bash
# Convenience: CONDITION=low_budget (Fig. 3, 400 queries)
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDITION=low_budget bash "$DIR/run_condition.sh" "$@"

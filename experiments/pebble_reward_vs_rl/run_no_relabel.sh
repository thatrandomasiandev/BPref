#!/usr/bin/env bash
# Convenience: CONDITION=no_relabel (Fig. 5a)
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDITION=no_relabel bash "$DIR/run_condition.sh" "$@"

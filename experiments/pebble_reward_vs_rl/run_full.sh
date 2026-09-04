#!/usr/bin/env bash
# Convenience: CONDITION=full
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDITION=full bash "$DIR/run_condition.sh" "$@"

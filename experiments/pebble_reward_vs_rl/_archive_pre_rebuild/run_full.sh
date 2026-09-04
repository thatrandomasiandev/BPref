#!/usr/bin/env bash
# Convenience wrappers — paper ablation conditions.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDITION=full bash "$DIR/run_condition.sh" "$@"

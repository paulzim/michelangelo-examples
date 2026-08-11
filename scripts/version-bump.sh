#!/usr/bin/env bash
# Bump michelangelo-examples' own version, or its michelangelo floor constraint.
#
# Usage:
#   ./scripts/version-bump.sh <new-version>          bump this package's own version
#   ./scripts/version-bump.sh --pin <new-floor>       bump the michelangelo floor constraint
set -euo pipefail

if [ "${1:-}" = "--pin" ]; then
    NEW_FLOOR="${2:?usage: version-bump.sh --pin <new-floor>}"
    sed -i.bak -E "s/\"michelangelo>=[^\"]*\"/\"michelangelo>=${NEW_FLOOR}\"/" pyproject.toml
    rm pyproject.toml.bak
    echo "Bumped michelangelo floor to >=${NEW_FLOOR}"
else
    NEW_VERSION="${1:?usage: version-bump.sh <new-version>}"
    sed -i.bak -E "s/^version = \".*\"/version = \"${NEW_VERSION}\"/" pyproject.toml
    rm pyproject.toml.bak
    echo "Bumped michelangelo-examples version to ${NEW_VERSION}"
fi

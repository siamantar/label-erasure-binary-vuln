#!/usr/bin/env bash
# Download the two labelled corpora the experiments use and verify them.
#
# Source: release ds-v0.0.1 of L1NNA/Juliet-NDSS18-BinaryCodeSum. The files
# are third-party data and are not redistributed in this repository.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/fetch_data.sh [--dest DIR]

Downloads mne-opr-desp-juliet-labelled.pkl and mne-opr-desp-ndss-labelled.pkl
from release ds-v0.0.1 of L1NNA/Juliet-NDSS18-BinaryCodeSum and checks their
SHA-256 digests. Files that already exist with the right digest are kept.

Options:
  --dest DIR   where to put the files (default: data/ in this repository,
               which is where scrub.py, gate_inspect.py and
               leak_position.py look for them)
  -h, --help   show this help and exit
EOF
}

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO/data"
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dest) [ $# -ge 2 ] || { echo "error: --dest needs a directory" >&2; exit 2; }
                DEST="$2"; shift 2 ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

BASE="https://github.com/L1NNA/Juliet-NDSS18-BinaryCodeSum/releases/download/ds-v0.0.1"
FILES=(
    "mne-opr-desp-juliet-labelled.pkl 8a13310e7ecac1fe118997b45e7a737dad7b33528b76b8e6c5b8051e847920e3"
    "mne-opr-desp-ndss-labelled.pkl 691d6d890411dfe8635bb7b69b0600219c7d2c8780236abcb763a0a899982c29"
)

command -v curl >/dev/null || { echo "error: curl is required" >&2; exit 2; }
command -v sha256sum >/dev/null || { echo "error: sha256sum is required" >&2; exit 2; }
mkdir -p "$DEST"

status=0
for entry in "${FILES[@]}"; do
    name="${entry%% *}"
    want="${entry##* }"
    path="$DEST/$name"
    if [ -f "$path" ] && [ "$(sha256sum "$path" | cut -d' ' -f1)" = "$want" ]; then
        echo "ok (already present)  $name"
        continue
    fi
    echo "downloading           $name"
    curl -fL --retry 3 -o "$path.part" "$BASE/$name"
    got="$(sha256sum "$path.part" | cut -d' ' -f1)"
    if [ "$got" = "$want" ]; then
        mv "$path.part" "$path"
        echo "ok                    $name"
    else
        echo "error: SHA-256 mismatch for $name" >&2
        echo "  expected $want" >&2
        echo "  got      $got" >&2
        echo "  The download is corrupt or the release changed. Kept as $path.part." >&2
        status=1
    fi
done
exit $status

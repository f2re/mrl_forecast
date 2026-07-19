#!/bin/bash
set -e

STATION=${1:-ess}
DATE=${2:?Usage: bash scripts/download_dwd.sh STATION YYYY-MM-DD [COUNT]}
COUNT=${3:-100}

source venv/bin/activate
python3 src/download_dwd_archive.py \
    --station "$STATION" \
    --date "$DATE" \
    --count "$COUNT" \
    --output data/raw/archive

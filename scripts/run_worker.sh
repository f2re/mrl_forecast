#!/bin/bash

set -e

source venv/bin/activate
exec python3 scripts/job_worker.py "$@"

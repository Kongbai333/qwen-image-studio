#!/bin/sh
set -eu

data_dir="${QWEN_DATA_DIR:-/data}"
mkdir -p "$data_dir/uploads" "$data_dir/outputs"

# A new Docker volume can be created as root-owned. Repair it before the
# unprivileged application process starts.
chown -R app:app "$data_dir"

exec gosu app "$@"

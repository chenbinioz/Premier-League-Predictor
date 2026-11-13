#!/bin/sh
# Always serve with the same interpreter used by the versioned model bundle.
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_python="$project_dir/venv/bin/python"
if [ ! -x "$project_python" ]; then
    echo "Project environment missing. Create venv and install requirements.txt first." >&2
    exit 1
fi
cd "$project_dir"
exec "$project_python" -m streamlit run "$project_dir/app.py" "$@"

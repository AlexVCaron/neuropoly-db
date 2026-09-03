#!/bin/sh
set -e

# Load DB password from docker secret if present
if [ -f /run/secrets/db_user_password ]; then
  export NB_GRAPH_PASSWORD=$(cat /run/secrets/db_user_password)
fi

# Run injection script if present in any of the expected locations.
if [ -f /usr/src/neurobagel/scripts/inject_vocab_patch.py ]; then
  python3 /usr/src/neurobagel/scripts/inject_vocab_patch.py || true
elif [ -f /usr/src/inject_vocab_patch.py ]; then
  python3 /usr/src/inject_vocab_patch.py || true
elif [ -f /usr/src/neurobagel/inject_vocab_patch.py ]; then
  python3 /usr/src/neurobagel/inject_vocab_patch.py || true
fi

exec uvicorn app.main:app --proxy-headers --host 0.0.0.0 --port ${NB_API_PORT:-8000}

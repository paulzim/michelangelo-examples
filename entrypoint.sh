#!/bin/bash
set -e

# Load auto-detected JAVA_HOME (written at image build time to handle arm64/amd64).
# TODO(#1295): remove once Dockerfile no longer hardcodes ENV JAVA_HOME=${TARGETARCH}.
# Only pull JAVA_HOME out of /etc/environment -- sourcing the whole file also
# assigns its stock Ubuntu PATH= line, clobbering the venv-first PATH already
# set via the image's own ENV PATH (breaks `python3` resolution for any
# subprocess spawned through this entrypoint, e.g. Ray job entrypoints).
if [ -f /etc/environment ]; then
  # shellcheck disable=SC1090
  JAVA_HOME="$(. /etc/environment && echo "$JAVA_HOME")"
fi
export JAVA_HOME

if [[ "$1" == "driver" || "$1" == "executor" ]]; then
  echo "[entrypoint] Removing Spark role argument: $1"
  shift
fi

# If the command starts with something other than spark-submit or Spark job args, treat it as non-Spark (e.g., Ray)
if [[ "$1" != *.py && "$1" != "--"* && "$1" != "spark-submit" && "$1" != *.jar ]]; then
  echo "[entrypoint] Detected non-Spark command, running as-is: $*"
  exec "$@"
fi

# Remove --properties-file argument if present
ARGS=()
SKIP_NEXT=0
for arg in "$@"; do
  if [[ $SKIP_NEXT -eq 1 ]]; then
    SKIP_NEXT=0
    continue
  fi

  if [[ "$arg" == "--properties-file" ]]; then
    SKIP_NEXT=1
    continue
  fi

  ARGS+=("$arg")
done

echo "[entrypoint] Executing: /opt/spark/bin/spark-submit ${ARGS[*]}"
exec /opt/spark/bin/spark-submit "${ARGS[@]}"

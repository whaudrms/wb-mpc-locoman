#!/usr/bin/env bash
# Use the existing Drake 1.30 / Python 3.10 image; no host Python changes.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
idto_dir="${IDTO_DIR:-/home/tony/4_idto}"
image="${IDTO_IMAGE:-idto:latest}"
if [[ ! -f "$idto_dir/build/python_bindings/pyidto.cpython-310-x86_64-linux-gnu.so" ]]; then
  echo "Missing Python 3.10 IDTO build in $idto_dir/build/python_bindings" >&2
  exit 1
fi
docker_args=(--rm --network none --read-only --tmpfs /tmp
  -u "$(id -u):$(id -g)" -e MPLCONFIGDIR=/tmp/matplotlib
  -v "$project_dir:/wb:ro" -v "$idto_dir:/idto:ro"
  -e PYTHONPATH=/home/drake/lib/python3.10/site-packages:/idto/build/python_bindings:/wb
  -w /wb --entrypoint python3)
if [[ -n "${IDTO_OUTPUT_DIR:-}" ]]; then
  mkdir -p -- "$IDTO_OUTPUT_DIR"
  docker_args+=(-v "$(cd -- "$IDTO_OUTPUT_DIR" && pwd):/output:rw")
fi
# Optional interactive visualization needs a published port and stdin.
for arg in "$@"; do
  if [[ "$arg" == "--meshcat" ]]; then
    # Replace network none with bridge for localhost-only port publishing.
    for i in "${!docker_args[@]}"; do
      if [[ "${docker_args[$i]}" == "none" ]]; then docker_args[$i]=bridge; fi
    done
    docker_args+=(-i -p 127.0.0.1:7000:7000)
    break
  fi
done
if [[ "${1:-}" == "--server" ]]; then
  shift
  exec docker run "${docker_args[@]}" -i "$image" -B /wb/idto_server.py "$@"
fi
if [[ "${1:-}" == "--test" ]]; then
  shift
  exec docker run "${docker_args[@]}" "$image" -B -m unittest discover -s tests -p test_idto.py "$@"
fi
exec docker run "${docker_args[@]}" "$image" -B /wb/main_idto.py "$@"

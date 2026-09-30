#!/usr/bin/env bash
# Isolated, pinned HPIPM/BLASFEO build; no sudo or conda/pip modifications.
set -euo pipefail
task_deps_root="${1:-/tmp/wb_mpc_hpipm_bench}"
mkdir -p "$task_deps_root"
task_deps_root="$(cd "$task_deps_root" && pwd)"
fetch_source() {
    local project="$1" revision="$2" directory="$task_deps_root/$1"
    if [[ ! -d "$directory/.git" ]]; then
        git init "$directory"
        git -C "$directory" remote add origin "https://github.com/giaf/$project.git"
        git -C "$directory" fetch --depth 1 origin "$revision"
        git -C "$directory" checkout --detach FETCH_HEAD
    fi
    if [[ "$(git -C "$directory" rev-parse HEAD)" != "$revision" ]]; then
        printf 'Unexpected %s revision; use a new dependency directory.\n' "$project" >&2
        exit 1
    fi
}
fetch_source hpipm bad1123a96136fce3cdd8bbcbe24b853d012b840
fetch_source blasfeo 9628393214623b29e5dd2a8784f7423edbc93e04
make -C "$task_deps_root/blasfeo" shared_library -j 4 CC="${CC:-gcc} -Wl,-soname,libblasfeo.so" TARGET="${BLASFEO_TARGET:-X64_INTEL_HASWELL}"
make -C "$task_deps_root/blasfeo" install_shared PREFIX="$task_deps_root/install"
make -C "$task_deps_root/hpipm" shared_library -j 4 CC="${CC:-gcc} -Wl,-soname,libhpipm.so" TARGET="${HPIPM_TARGET:-AVX}" BLASFEO_PATH="$task_deps_root/install/blasfeo"
make -C "$task_deps_root/hpipm" install_shared PREFIX="$task_deps_root/install"
printf 'Built HPIPM and BLASFEO under %s/install\n' "$task_deps_root"

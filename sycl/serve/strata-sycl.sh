#!/usr/bin/env bash
# strata-sycl.sh: the SYCL-built engine as a drop-in `exe` for serve/server.py (--engine strata). The port's binary
# needs the oneAPI runtime of the dev image, so this runs it there with stdin/stdout attached (the serve protocol is
# lines on those pipes) and stderr going to the server's log. Paths in the config's args are the container's:
# the data root is mounted at /work.
#   STRATA_SYCL_ROOT    host directory mounted at /work        (default: two levels above this script's repo)
#   STRATA_SYCL_IMAGE   the runtime image                      (default: strata-sycl-dev)
#   STRATA_SYCL_BIN     the engine binary, relative to the repo (default: build-sycl-aot/strata)
#   STRATA_SYCL_NAME    the container's name                   (default: strata-sycl-serve)
#   STRATA_SYCL_SHM     1 (default): the host's /dev/shm is the container's /dev/shm. Pictures: the server (on the host) writes the
#                       image embeddings of a request into /dev/shm and names that file on the engine's GENI line, so the engine in
#                       the container must see the same file. 0: the container keeps a private /dev/shm.
#   STRATA_SYCL_MOUNTS  host paths outside the data root the args name (a model on another disk), mounted at the same path,
#                       separated by spaces (default: none)
#   STRATA_SYCL_RUNNER  docker (default), podman, native (no container: oneAPI installed on the host) or distrobox:<name>
#   STRATA_SYCL_SECCOMP the container's seccomp profile. Default: seccomp-io_uring.json beside this script for strata-dsv41 (it
#                       pins host RAM through io_uring, which the runtime's own profile refuses), the runtime's own profile for
#                       any other engine. Empty: the runtime's own profile always.
#   STRATA_SYCL_NUMA    set (e.g. interleave): the container may set NUMA memory policies (--cap-add SYS_NICE: the default seccomp
#                       profiles allow mbind and set_mempolicy only with it). For strata-dsv41's DS4_STORE_NUMA=interleave.
#   STRATA_SYCL_AUTO    1 (default): for strata-dsv41, switches picked from this machine (below); 0: none
# Passed into the container: every variable of this script's environment that starts with STRATA_ (the config's
# "env" block arrives that way: the server puts it in the engine's environment), ONEAPI_ (level_zero:* for a
# two-card split, #423), UR_, IGC_, SYCL_, ZES_ or DS4_ (strata-dsv41's own switches). The port's own defaults (below)
# are overridden the same way, e.g. "env": {"STRATA_VERIFY_NO_HOST": "0"} in the config.
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)                 # the repo
root=${STRATA_SYCL_ROOT:-$(dirname "$here")}
repo_in=/work/$(basename "$here")
name=${STRATA_SYCL_NAME:-strata-sycl-serve}
bin=${STRATA_SYCL_BIN:-build-sycl-aot/strata}
args=""
for a in "$@"; do args+=" $(printf '%q' "$a")"; done
# the port's run-time switches: the device-built verify plan, and no host handshakes in it. NO_HOST is for a card on
# the xe driver (the host<->GPU flag stores are not visible there); an i915 card (Arc A-series) runs without it
# (docs/INTEL.md)
declare -A setting=([STRATA_VERIFY_DEVICE_PLAN]=1 [STRATA_VERIFY_NO_HOST]=1 [STRATA_STAGER_THREADS]=12)
# strata-dsv41's switches picked from this machine (STRATA_SYCL_AUTO=0: none). Each rule is one measured on an Arc Pro B60
# (2026-11-01, 9352-token prompt): the XMX expert tiles, the XMX prompt attention and the copy engine took the prompt from
# 135 to ~283 tok/s; NUMA interleave took decode from 11.6 to 14.3 tok/s on a 2-socket Xeon (2026-10-31). A variable already
# in the environment wins over its rule; each choice is printed with its reason
auto_notes=()
auto_args=()
if [[ "$(basename "$bin")" == strata-dsv41* ]] && [ "${STRATA_SYCL_AUTO:-1}" != 0 ]; then
    auto() {   # name value reason
        if [ -n "${!1+x}" ]; then
            [ "${!1}" = "$2" ] || auto_notes+=("$1=${!1} kept from the environment (this machine's pick: $2, $3)")
        else
            setting[$1]=$2
            auto_notes+=("$1=$2 ($3)")
        fi
    }
    # the binary's device code: an AOT build for Battlemage (bmg-*) only runs on Battlemage, where the XMX kernels were checked
    aot=$(sed -n 's/^STRATA_SYCL_AOT:[A-Z]*=//p' "$here/$(dirname "$bin")/CMakeCache.txt" 2>/dev/null || true)
    if [[ "$aot" == bmg* ]]; then
        auto DS4_XMX 1 "AOT build for $aot: the experts' tiles on the XMX units"
        auto DS4_FAST_ATTN 1 "AOT build for $aot: the prompt's attention on the XMX units"
        auto DS4_COPY_ENGINE 1 "AOT build for $aot: the expert copies on the copy engine, beside the kernels"
    fi
    nodes=$(ls -d /sys/devices/system/node/node[0-9]* 2>/dev/null | wc -l)
    if [ "$nodes" -gt 1 ]; then
        auto DS4_STORE_NUMA interleave "$nodes NUMA nodes"
        [ "${setting[DS4_STORE_NUMA]:-}" != interleave ] || STRATA_SYCL_NUMA=${STRATA_SYCL_NUMA:-1}
    fi
    # the RAM store: a third of the RAM free now, at most 80 GB (the measured size), never the whole model - other programs
    # share the machine. Below 16 GB the engine's own default stays
    avail=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo 2>/dev/null || echo 0)
    store=$(( avail / 3 > 80 ? 80 : avail / 3 ))
    if [ "$store" -ge 16 ]; then auto DS4_STORE_GB "$store" "a third of the ${avail} GB of RAM free, at most 80"; fi
    # one prompt chunk up to 16384 tokens (a 9352-token prompt: 132 tok/s in one chunk against ~100 in two)
    case " $* " in
        *" --prefill-chunk "*) ;;
        *) ctx=0; prev=""
           for a in "$@"; do [ "$prev" = --max-context ] && ctx=$a; prev=$a; done
           chunk=$(( ctx > 0 && ctx < 16384 ? ctx : 16384 ))
           auto_args=(--prefill-chunk "$chunk"); auto_notes+=("--prefill-chunk $chunk (one chunk up to that many prompt tokens)") ;;
    esac
    for n in "${auto_notes[@]}"; do echo "strata-sycl.sh: $n" >&2; done
    echo "strata-sycl.sh: STRATA_SYCL_AUTO=0 turns these picks off" >&2
fi
for a in "${auto_args[@]}"; do args+=" $(printf '%q' "$a")"; done
while IFS= read -r k; do
    case "$k" in
        STRATA_SYCL_*) ;;                                 # this script's own settings
        STRATA_*|ONEAPI_*|UR_*|IGC_*|SYCL_*|ZES_*|DS4_*|NEOReadDebugKeys|OverrideDefaultFP64Settings) setting[$k]=${!k} ;;
    esac
done < <(compgen -e)
envs=()
for k in "${!setting[@]}"; do
    # the Qwen engine tests its switches by presence: a value of 0 (or empty) means "not set", so it is not passed on.
    # strata-dsv41 reads its own (DS4_*, STRATA_DS4_*) by value, and some are on unless set to 0 (DS4_PIN, STRATA_DS4_REUSE):
    # a 0 there is passed on
    case "$k=${setting[$k]}" in
        *=) ;;
        DS4_*|STRATA_DS4_*) envs+=(-e "$k=${setting[$k]}") ;;
        *=0) ;;
        *) envs+=(-e "$k=${setting[$k]}") ;;
    esac
done
run_cmd="cd $repo_in && exec $bin$args"
runner=${STRATA_SYCL_RUNNER:-docker}
case "$runner" in
native|distrobox:?*)
    # no container: the same settings exported in a shell with oneAPI's environment (the image's single-card
    # selector unless one is set), run from the repo; the args' paths are the host's (there is no /work)
    exports="export ONEAPI_DEVICE_SELECTOR=$(printf '%q' "${ONEAPI_DEVICE_SELECTOR:-level_zero:0}") SYCL_CACHE_PERSISTENT=0 ZES_ENABLE_SYSMAN=1"
    for e in "${envs[@]}"; do [ "$e" = -e ] || exports+=" $(printf '%q' "$e")"; done
    script="set +u; . /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; $exports; cd $(printf '%q' "$here") && exec $bin$args"
    if [ "$runner" = native ]; then exec bash -c "$script"; fi
    exec distrobox enter "${runner#distrobox:}" -- bash -c "$script" ;;
docker) ctr=(docker); extra=() ;;
podman) ctr=(podman); extra=(--security-opt label=disable) ;;
*) echo "strata-sycl.sh: STRATA_SYCL_RUNNER is docker, podman, native or distrobox:<name>" >&2; exit 2 ;;
esac
# strata-dsv41 pins the RAM the GPU copies from through io_uring (ds4_sycl.hpp ds4_host_pin), which the runtimes' default
# seccomp profiles refuse: seccomp-io_uring.json is their default profile (moby's) plus io_uring_setup / _enter / _register.
# Without it the engine warns and runs unpinned
case "$(basename "$bin")" in
    strata-dsv41*) seccomp_default="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/seccomp-io_uring.json" ;;
    *) seccomp_default="" ;;
esac
seccomp=${STRATA_SYCL_SECCOMP-$seccomp_default}
if [ -n "$seccomp" ] && [ -f "$seccomp" ]; then extra+=(--security-opt "seccomp=$seccomp"); fi
# NUMA memory policies (mbind, set_mempolicy) need CAP_SYS_NICE under the default seccomp profiles
if [ -n "${STRATA_SYCL_NUMA:-}" ]; then extra+=(--cap-add SYS_NICE); fi
mounts=()
for m in ${STRATA_SYCL_MOUNTS:-}; do mounts+=(-v "$m:$m"); done
shm=()
if [ "${STRATA_SYCL_SHM:-1}" != 0 ] && [ -d /dev/shm ]; then shm=(-v /dev/shm:/dev/shm); fi
"${ctr[@]}" rm -f "$name" >/dev/null 2>&1 || true        # a container left behind by a killed server
exec "${ctr[@]}" run --rm -i --name "$name" --device /dev/dri --oom-score-adj 1000 --stop-timeout 30 --no-healthcheck \
    --ulimit memlock=-1 "${extra[@]}" \
    -v "$root:/work" \
    "${mounts[@]}" \
    "${shm[@]}" \
    "${envs[@]}" \
    "${STRATA_SYCL_IMAGE:-strata-sycl-dev}" \
    "$run_cmd"

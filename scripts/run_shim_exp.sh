#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: run_shim_exp.sh <B1|L1|L2|smoke-B1|smoke-L1|smoke-L2>
       [--out DIR] [--sync-dir DIR] [--seed N] [--total-iterations N]
       [--resume-auto] [--sync-interval SEC] [--dry-run] [-- EXTRA_ARGS...]
USAGE
}

die() { printf '%s\n' "$*" >&2; exit 1; }

case ${1:-} in
    -h|--help) usage; exit 0 ;;
    B1|L1|L2|smoke-B1|smoke-L1|smoke-L2) preset=$1; shift ;;
    *) usage >&2; exit 2 ;;
esac

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
case ${preset#smoke-} in
    B1) name=B1_himloco_amp_rough_cmdcur_4096; method=himloco; task=himloco_a1_amp_rough_cmdcur ;;
    L1) name=L1_shim_nlif_rough_cmdcur_4096; method=s-him; task=s_him_a1_rough_cmdcur ;;
    L2) name=L2_shim_spike_rough_cmdcur_4096; method=s-him; task=s_him_a1_spike_rough_cmdcur ;;
esac

total=
smoke_args=()
if [[ $preset == smoke-* ]]; then
    name=smoke-$name
    total=5
    smoke_args+=(--checkpoint-interval 0 --diagnostics-interval 1)
fi
out="$repo/runs/s-him-long/$name"
sync_dir=
sync_interval=300
seed=
resume=0
dry_run=0
while (($#)); do
    case $1 in
        --out|--sync-dir|--seed|--total-iterations|--sync-interval)
            [[ -n ${2:-} && $2 != --* ]] || die "Missing value for $1"
            case $1 in
                --out) out=$2 ;;
                --sync-dir) sync_dir=$2 ;;
                --seed) seed=$2 ;;
                --total-iterations) total=$2 ;;
                --sync-interval) sync_interval=$2 ;;
            esac
            shift 2 ;;
        --resume-auto) resume=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        --) shift; break ;;
        *) die "Unknown launcher option: $1 (pass training arguments after --)" ;;
    esac
done
[[ -z $total || $total =~ ^(0|[1-9][0-9]*)$ ]] || die "--total-iterations must be a nonnegative integer"
[[ -z $seed || $seed =~ ^[0-9]+$ ]] || die "--seed must be a nonnegative integer"
[[ $sync_interval =~ ^[1-9][0-9]*$ ]] || die "--sync-interval must be a positive integer"
out=$(realpath -m -- "$out")
if [[ -n $sync_dir ]]; then
    sync_dir=$(realpath -m -- "$sync_dir")
    [[ $sync_dir != "$out" ]] || sync_dir=
fi

if [[ -n ${SHIM_PYTHON:-} ]]; then
    python=$(command -v -- "$SHIM_PYTHON") || die "Cannot find SHIM_PYTHON: $SHIM_PYTHON"
else
    env_name=${SHIM_ENV_NAME:-shim}
    python=
    candidates=()
    if [[ -n ${MAMBA_ROOT_PREFIX:-} ]]; then
        candidates+=("$MAMBA_ROOT_PREFIX/envs/$env_name/bin/python")
    fi
    candidates+=("/content/shim-mamba/envs/$env_name/bin/python"
                 "$HOME/.local/share/shim-mamba/envs/$env_name/bin/python"
                 /home/qyw/miniconda3/envs/himloco/bin/python)
    for candidate in "${candidates[@]}"; do
        if [[ -x $candidate ]]; then
            python=$candidate
            break
        fi
    done
    if [[ -z $python ]]; then
        python=$(command -v python3) || die "Cannot find python3"
    fi
fi
# Keep the environment's Python symlink intact.
[[ $python == /* ]] || python="$PWD/$python"
cd -- "$repo"
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 MPLBACKEND=Agg
if [[ -z $total ]]; then
    total=$("$python" - "$task" <<'PYTHON'
import sys

from graduation.legacy import load_config

_, training = load_config(sys.argv[1])
print(training["runner"]["max_iterations"])
PYTHON
    )
fi

shopt -s nullglob
copy_files() {
    local destination=$1 file temp
    shift
    if (($# == 0)); then
        return 0
    fi
    mkdir -p -- "$destination" || return
    if command -v rsync >/dev/null 2>&1; then
        rsync -a -- "$@" "$destination/"
        return
    fi
    for file; do
        if [[ -f $destination/${file##*/} ]] &&
            [[ $(stat -c '%s:%y' -- "$file") == "$(stat -c '%s:%y' -- "$destination/${file##*/}")" ]]; then
            continue
        fi
        # Publish each copy atomically so a later resume never sees an interrupted transfer.
        temp=$(mktemp -- "$destination/.shim-sync.XXXXXX") || return
        if ! cp -a -- "$file" "$temp" || ! mv -f -- "$temp" "$destination/${file##*/}"; then
            rm -f -- "$temp"
            return 1
        fi
    done
}

copy_artifacts() {
    local source=$1 destination=$2 file
    local files=()
    for file in "$source"/console.log "$source"/validation.json "$source"/model*.pt; do
        [[ ! -f $file ]] || files+=("$file")
    done
    copy_files "$destination" "${files[@]}"
}

remaining=$total
checkpoint_args=()
if ((resume)); then
    if ((!dry_run)) && [[ -n $sync_dir ]]; then
        mkdir -p -- "$out"
        if [[ -z $(find "$out" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
            copy_artifacts "$sync_dir" "$out"
        fi
    fi
    checkpoints=("$out"/model_*.pt "$out"/model.pt)
    if [[ -n $sync_dir ]]; then
        checkpoints+=("$sync_dir"/model_*.pt "$sync_dir"/model.pt)
        if ((!dry_run)) && [[ ! -e $out/console.log && -f $sync_dir/console.log ]]; then
            copy_files "$out" "$sync_dir/console.log"
        fi
    fi
    selected=$("$python" - "${checkpoints[@]}" <<'PY'
import sys
from pathlib import Path

paths = [Path(name) for name in sys.argv[1:] if Path(name).is_file()]
if paths:
    import torch

    best_iteration, best_path = -1, None
    for path in paths:
        try:
            try:
                state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            except RuntimeError:
                state = torch.load(path, map_location="cpu", weights_only=True)
            iteration = int(state.get("iteration", state.get("iter", -1)))
            if iteration < 0:
                raise ValueError("checkpoint is missing its iteration")
        except Exception as exc:
            print(f"Skipping unreadable checkpoint {path}: {exc}", file=sys.stderr)
            continue
        if iteration > best_iteration:
            best_iteration, best_path = iteration, path
    if best_path is None:
        sys.exit("No readable checkpoint found.")
    print(best_iteration)
    print(best_path)
PY
)
    if [[ -n $selected ]]; then
        completed=${selected%%$'\n'*}
        checkpoint=${selected#*$'\n'}
        if ((!dry_run)) && [[ -n $sync_dir && ${checkpoint%/*} == "$sync_dir" ]]; then
            copy_files "$out" "$checkpoint"
            checkpoint="$out/${checkpoint##*/}"
            if [[ -f $sync_dir/console.log ]]; then
                "$python" - "$out/console.log" "$sync_dir/console.log" <<'PY'
import sys
from pathlib import Path

local, backup = map(Path, sys.argv[1:])
local_lines = local.read_bytes().splitlines(keepends=True)
backup_lines = backup.read_bytes().splitlines(keepends=True)
common = 0
for a, b in zip(local_lines, backup_lines):
    if a != b:
        break
    common += 1
if common < len(backup_lines):
    temp = local.with_suffix(".log.tmp")
    temp.write_bytes(b"".join(backup_lines + local_lines[common:]))
    temp.replace(local)
PY
            fi
        fi
        remaining=$((total - completed))
        if ((remaining <= 0)); then
            if ((!dry_run)) && [[ -n $sync_dir ]]; then
                copy_artifacts "$out" "$sync_dir"
            fi
            printf 'Already completed %s of %s iterations (%s).\n' "$completed" "$total" "$checkpoint"
            exit 0
        fi
        checkpoint_args=(--checkpoint "$checkpoint")
        printf 'Resuming at iteration %s with %s iterations remaining.\n' "$completed" "$remaining"
    fi
fi

if ((remaining == 0)); then
    printf 'No training iterations requested.\n'
    exit 0
fi

cmd=("$python" scripts/bench_blind.py "$method" --task "$task"
     --iterations "$remaining" --headless --out "$out" "${smoke_args[@]}")
if [[ -n $seed ]]; then
    cmd+=(--seed "$seed")
fi
if [[ ${DISABLE_KIT_DRIVER_BYPASS:-0} != 1 ]]; then
    cmd+=(--kit_args=--/rtx/verifyDriverVersion/enabled=false)
fi
cmd+=("${checkpoint_args[@]}" "$@")

printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'
if ((dry_run)); then
    if [[ -n $sync_dir ]]; then
        printf 'Sync every %ss to %s\n' "$sync_interval" "$sync_dir"
    fi
    exit 0
fi

mkdir -p -- "$out"

sync_outputs() {
    if ! copy_artifacts "$out" "$sync_dir"; then
        printf 'Could not sync artifacts to %s\n' "$sync_dir" >&2
        return 1
    fi
}

sync_loop() {
    local sleeper=
    trap 'if [[ -n $sleeper ]]; then kill "$sleeper" 2>/dev/null || :; fi; exit 0' TERM
    while true; do
        sleep "$sync_interval" &
        sleeper=$!
        wait "$sleeper"
        sleeper=
        sync_outputs || :
    done
}

trainer=
logger=
sync_pid=
cleanup() {
    local status=$?
    trap '' INT TERM
    if [[ -n $trainer ]]; then
        kill -TERM -- "-$trainer" 2>/dev/null || :
        wait "$trainer" || :
    fi
    if [[ -n $logger ]] && ! wait "$logger" && ((status == 0)); then
        status=1
    fi
    if [[ -n $sync_pid ]]; then
        kill -TERM "$sync_pid" 2>/dev/null || :
        wait "$sync_pid" || :
        if ! sync_outputs && ((status == 0)); then
            status=1
        fi
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ -n $sync_dir ]]; then
    sync_loop &
    sync_pid=$!
fi
tee_args=()
if ((resume)); then
    tee_args+=(-a)
fi
exec {log_fd}> >(tee "${tee_args[@]}" "$out/console.log")
logger=$!
# Give the trainer its own process group so wrappers such as Isaac Sim's python.sh
# and their Python children both receive the stop signal.
set -m
run_marker=$(mktemp "$out/.shim-run-start.XXXXXX")
"${cmd[@]}" >&"$log_fd" 2>&1 &
trainer=$!
set +m
exec {log_fd}>&-
status=0
wait "$trainer" || status=$?
trainer=
if ((status == 139)) && [[ -f $out/validation.json && $out/validation.json -nt $run_marker ]] &&
    grep -q '"status": *"passed"' "$out/validation.json" 2>/dev/null; then
    printf 'Note: SimulationApp.close() exited with 139 after validation.json passed; treating run as successful.\n' >&2
    status=0
fi
rm -f -- "$run_marker"
exit "$status"

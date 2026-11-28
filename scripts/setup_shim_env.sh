#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: setup_shim_env.sh [--skip-smoke] [--save-cache] [--help]

Overrides: MAMBA_ROOT_PREFIX, SHIM_ENV_NAME (default shim), ISAACLAB_DIR,
           ISAACLAB_TAG (default v2.3.2), TORCH_CUDA_INDEX, ENV_ARCHIVE.
Set ENV_ARCHIVE to a local or mounted .tar.zst/.tar.gz archive to reuse it.
--save-cache writes ENV_ARCHIVE, or an archive under MAMBA_ROOT_PREFIX.
USAGE
}

die() { printf '%s\n' "$*" >&2; exit 1; }

smoke=1
save_cache=0
for option in "$@"; do
    case $option in
        --skip-smoke) smoke=0 ;;
        --save-cache) save_cache=1 ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $option" ;;
    esac
done

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
default_root="$HOME/.local/share/shim-mamba"
if [[ -d /content ]]; then
    default_root=/content/shim-mamba
fi
export MAMBA_ROOT_PREFIX
MAMBA_ROOT_PREFIX=$(realpath -m -- "${MAMBA_ROOT_PREFIX:-$default_root}")
env_name=${SHIM_ENV_NAME:-shim}
[[ $env_name =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$ ]] || die "Invalid SHIM_ENV_NAME: $env_name"
prefix="$MAMBA_ROOT_PREFIX/envs/$env_name"
mamba="$MAMBA_ROOT_PREFIX/bin/micromamba"
lab_dir=$(realpath -m -- "${ISAACLAB_DIR:-$MAMBA_ROOT_PREFIX/IsaacLab}")
lab_tag=${ISAACLAB_TAG:-v2.3.2}
migration="$repo/third_party/MyHIMLoco/isaaclab_migration"
archive=${ENV_ARCHIVE:-}
if [[ -n $archive ]]; then
    archive=$(realpath -m -- "$archive")
fi
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1

[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || die "This setup targets Linux x86_64."

mkdir -p -- "$MAMBA_ROOT_PREFIX/envs"
scratch=$(mktemp -d "$MAMBA_ROOT_PREFIX/.shim-setup.XXXXXX")
cache_temp=
cleanup() {
    rm -rf -- "$scratch"
    if [[ -n $cache_temp ]]; then
        rm -f -- "$cache_temp"
    fi
}
trap cleanup EXIT

ensure_mamba() {
    if [[ ! -x $mamba ]]; then
        curl -fsSL --retry 3 https://micro.mamba.pm/api/micromamba/linux-64/latest -o "$scratch/micromamba.tar.bz2"
        tar -xjf "$scratch/micromamba.tar.bz2" -C "$scratch" bin/micromamba
        install -Dm755 -- "$scratch/bin/micromamba" "$mamba"
    fi
}

cuda_index=${TORCH_CUDA_INDEX:-}
if [[ -z $cuda_index ]]; then
    smi=$(nvidia-smi 2>/dev/null || true)
    if [[ $smi =~ CUDA[[:space:]]+Version:[[:space:]]+([0-9]+)\.([0-9]+) ]]; then
        major=${BASH_REMATCH[1]}
        minor=${BASH_REMATCH[2]}
        if ((major > 12 || (major == 12 && minor >= 8))); then
            cuda_index=cu128
        elif ((major == 12 && minor >= 6)); then
            cuda_index=cu126
        else
            cuda_index=cu124
        fi
    else
        printf 'No driver CUDA version found. Using cu128; set TORCH_CUDA_INDEX to override.\n' >&2
        cuda_index=cu128
    fi
fi
if [[ $cuda_index == cu124 ]]; then
    printf 'PyTorch 2.7.0 has no cu124 wheels. Using the published cu118 build.\n' >&2
    cuda_index=cu118
fi
[[ $cuda_index =~ ^cu[0-9]+$ ]] || die "Invalid TORCH_CUDA_INDEX: $cuda_index"

python="$prefix/bin/python"
if [[ ! -x $python ]]; then
    if [[ -n $archive && -f $archive ]]; then
        printf 'Restoring environment from %s\n' "$archive"
        mkdir -- "$scratch/env"
        # GNU tar identifies gzip from its header even if ENV_ARCHIVE ends in .zst.
        magic=$(od -An -tx1 -N4 -- "$archive" | tr -d ' \n')
        if [[ $magic == 28b52ffd ]] && ! command -v zstd >/dev/null 2>&1; then
            ensure_mamba
            "$mamba" create -y -p "$scratch/zstd" --override-channels -c conda-forge zstd
            PATH="$scratch/zstd/bin:$PATH" tar -xf "$archive" -C "$scratch/env"
        else
            tar -xf "$archive" -C "$scratch/env"
        fi
        [[ -x $scratch/env/bin/python && -f $scratch/env/bin/conda-unpack ]] ||
            die "Archive must contain a packed environment from --save-cache."
        mv -T -- "$scratch/env" "$prefix"
        "$python" "$prefix/bin/conda-unpack"
    else
        ensure_mamba
        "$mamba" create -y -n "$env_name" --override-channels -c conda-forge python=3.11 pip
    fi
fi
if [[ ! -x $prefix/bin/uv ]]; then
    "$python" -m pip install uv
fi
uv=("$prefix/bin/uv" pip install --python "$python")

lab_source="$lab_dir/source/isaaclab"
if [[ ! -e $lab_dir ]]; then
    mkdir -p -- "$(dirname -- "$lab_dir")"
    git clone --depth 1 --branch "$lab_tag" https://github.com/isaac-sim/IsaacLab.git "$lab_dir"
fi
[[ -f $lab_source/setup.py ]] || die "Isaac Lab source missing under $lab_source"

lab_revision=$(git -C "$lab_dir" rev-parse HEAD)
cache_key="sim5.1.0-torch2.7.0-$cuda_index-numpy1.26.0-scipy1.15.3-lab$lab_revision"
stamp="$prefix/.shim-env-ready"
constraints="$scratch/constraints.txt"
cat > "$constraints" <<PINS
torch==2.7.0+$cuda_index
torchvision==0.22.0+$cuda_index
torchaudio==2.7.0+$cuda_index
numpy==1.26.0
scipy==1.15.3
setuptools<81
PINS

lab_options=(-c "$constraints")
if [[ -f $stamp && $(<"$stamp") == "$cache_key" ]]; then
    printf 'Using installed dependencies (%s).\n' "$cuda_index"
    lab_options+=(--no-deps)
else
    printf 'Installing Isaac Sim and PyTorch (%s).\n' "$cuda_index"
    rm -f -- "$stamp"
    "${uv[@]}" --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio \
        --index-url "https://download.pytorch.org/whl/$cuda_index" \
        torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
    "${uv[@]}" -c "$constraints" --extra-index-url https://pypi.nvidia.com 'isaacsim[all,extscache]==5.1.0'
    "${uv[@]}" -c "$constraints" 'setuptools>=68,<81' wheel toml \
        numpy==1.26.0 scipy==1.15.3 h5py PyYAML websockets psutil tensorboard trimesh matplotlib
fi
# Rebind editables to the current machine paths even after restoring an archive.
if ! "${uv[@]}" "${lab_options[@]}" --no-build-isolation -e "$lab_source"; then
    "$python" -m pip install "${lab_options[@]}" --no-build-isolation -e "$lab_source"
fi
"${uv[@]}" --no-deps --no-build-isolation -e "$migration"
rm -rf -- "$migration/.cache/a1"

"$python" - <<'PY'
import torch
import isaacsim
import isaaclab
import himloco_lab

print(f"torch {torch.__version__}, torch.cuda.is_available() = {torch.cuda.is_available()}")
for module in (isaacsim, isaaclab, himloco_lab):
    print(f"{module.__name__}: {module.__file__}")
PY
printf '%s\n' "$cache_key" > "$stamp"

if ((save_cache)); then
    format=tar.gz
    pack_dependencies=(conda-pack==0.8.1)
    if [[ $archive != *.tar.gz ]] && command -v zstd >/dev/null 2>&1; then
        format=tar.zst
        pack_dependencies+=(zstandard)
    fi
    archive=${archive:-"$MAMBA_ROOT_PREFIX/$env_name.$format"}
    if [[ $archive == *.tar.zst && $format == tar.gz ]]; then
        printf 'zstd is unavailable. Writing a gzip-compressed archive at %s.\n' "$archive" >&2
    fi
    "${uv[@]}" "${pack_dependencies[@]}"
    mkdir -p -- "$(dirname -- "$archive")"
    cache_temp=$(mktemp -- "$archive.XXXXXX")
    "$prefix/bin/conda-pack" -p "$prefix" --ignore-editable-packages --format "$format" --force -o "$cache_temp"
    mv -f -- "$cache_temp" "$archive"
    cache_temp=
    printf 'Environment archive saved to %s\n' "$archive"
fi

if ((smoke)); then
    cd -- "$repo"
    "$python" scripts/bench_blind.py himloco --headless --terrain stairs --amp --amp-motion-set a1_diagonal \
        --num-envs 8 --steps 12 --iterations 2 --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
        --out runs/shim-setup-smoke
fi

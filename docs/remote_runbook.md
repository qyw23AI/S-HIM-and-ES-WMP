# S-HIM / HIMLoco+AMP 远程训练

| 预设 | 实验 | 规模 |
| --- | --- | --- |
| `B1` | HIMLoco + AMP（对角步态） | 4096 环境 × 5000 轮 |
| `L1` | S-HIM + AMP，NLIF 读出 | 4096 环境 × 5000 轮 |
| `L2` | S-HIM + AMP，脉冲读出 | 4096 环境 × 5000 轮 |

加上 `smoke-` 前缀可按相同规模跑 5 轮测速。用 `--dry-run` 查看完整命令，用 `--` 传递额外训练参数。每个预设和种子使用独立的输出目录。

## VS Code + Google Colab

1. 安装 `Google.colab` 扩展，在 VS Code 打开 [colab_shim.ipynb](../colab_shim.ipynb)。[Google 的连接说明](https://developers.googleblog.com/en/google-colab-is-coming-to-vs-code/)
2. 点击 **Select Kernel → Colab → L4**，登录 Google 账号。A100 缺少 Isaac Sim 5.1 要求的 RT Core。T4 可尝试，但 4096 环境可能超出其显存。优先用 L4 做 5 轮测速。[NVIDIA 硬件要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
3. 配置 `BRANCH`、`EXP_PRESET` 和 `SEED`，点击 **Run All**，按提示授权 Google Drive。分支需包含工作流脚本和 AMP 训练代码。测速单元从日志估算训练时间和 CU，可将费率替换为 Colab 会话显示的值。断线后用相同预设及种子重跑，启动器会从 Drive 上已同步的最新检查点继续。

`USE_ENV_CACHE=True` 会使用 Drive 中的环境归档，缺少归档时安装并保存。环境解包、训练和测速都在 `/content` 本地磁盘进行。检查点每 250 轮保存，本地日志和检查点默认每 300 秒同步一次。运行时被强制回收时，可以恢复到最后一次成功同步的检查点。[Colab 运行时说明](https://research.google.com/colaboratory/faq.html)

## 实验室服务器（免 sudo）

在有 NVIDIA 驱动及 Vulkan 驱动的 Linux x86_64 GPU 节点执行。Isaac Sim 5.1 的 Python wheel 要求 glibc ≥ 2.35。环境默认安装到 `$HOME/.local/share/shim-mamba`，可用 `MAMBA_ROOT_PREFIX` 改到配额充足的目录。PyTorch 2.7.0 提供 `cu128`、`cu126` 和 `cu118`。驱动报告 CUDA ≤ 12.5 时脚本使用 `cu118`，可通过 `TORCH_CUDA_INDEX` 显式指定。[Isaac Sim Python 安装](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html) · [PyTorch 2.7.0 wheel](https://pytorch.org/get-started/previous-versions/#v270)

在已分配的 GPU 节点上，用 tmux 保持终端会话：

```bash
tmux new -s shim
bash scripts/setup_shim_env.sh --skip-smoke
bash scripts/run_shim_exp.sh smoke-L1
bash scripts/run_shim_exp.sh L1 --seed 1 --resume-auto \
  --out "${TMPDIR:-/tmp}/$USER/shim/L1_seed1" --sync-dir "$HOME/shim_runs/L1_seed1"
```

跨节点复用环境时，可设 `ENV_ARCHIVE="$HOME/shim_cache/shim_env.tar.gz"` 并在首次安装时加 `--save-cache`。环境读取方使用同一归档，并按当前路径重新安装两个 editable 包。并行作业使用不同的输出目录和同步目录。

Slurm 示例，在仓库根目录保存为 `shim.sbatch`，执行 `sbatch shim.sbatch`。按集群规则补充 partition/account。

```bash
#!/usr/bin/env bash
#SBATCH --job-name=shim-l1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=shim-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
exec bash scripts/run_shim_exp.sh L1 --seed 1 --resume-auto \
  --out "${SLURM_TMPDIR:-/tmp}/$USER/shim/$SLURM_JOB_ID/L1_seed1" \
  --sync-dir "$HOME/shim_runs/L1_seed1" --sync-interval 300
```

## Docker / Apptainer（Singularity）

在允许构建镜像的机器上运行以下命令。Docker GPU 运行需要 NVIDIA Container Toolkit。项目目录挂载到 `/workspace/shim`，容器需要写入 A1 转换缓存及训练输出。

```bash
docker build -f docker/Dockerfile.shim -t shim:2.3.2 .
docker run --rm --gpus all --entrypoint /bin/bash \
  -e NVIDIA_DRIVER_CAPABILITIES=all -v "$PWD:/workspace/shim" \
  shim:2.3.2 scripts/run_shim_exp.sh smoke-B1
```

从 Docker 镜像生成 SIF 并传到 GPU 节点。站点只提供 `singularity` 时可替换下列命令中的 `apptainer`。

```bash
docker save -o shim-image.tar shim:2.3.2
apptainer build shim.sif docker-archive://"$PWD/shim-image.tar"
apptainer exec --nv --bind "$PWD:/workspace/shim" --pwd /workspace/shim \
  shim.sif bash scripts/run_shim_exp.sh B1 --resume-auto
```

# S-HIM 单阶段越障长训实验方案

日期：2026-11-28 · 平台：Unitree A1 · Isaac Lab / PhysX · 单卡 RTX 5060 Ti 16GB

本方案替代原有短训诊断方案（第一阶段短训结果与问题诊断已归档至 [`S-HIM第一次大范围短训方向总结.md`](S-HIM第一次大范围短训方向总结.md)），用于指导 **单阶段 `S-HIM` 一次性 `5000` 轮长训冲击 `terrain_level > 5.0`**，以及同条件下 **`HIMLoco + AMP` ANN 基线**的公平对比。

---

## 一、核心目标与严格约束

1. **单阶段定义**：从随机初始化（`iter = 0`）出发，在渐进金字塔楼梯课程地形（`--terrain stairs`，`step_height = 0.05 + 0.18 * difficulty`，`step_width = 0.31 m`，`platform_size = 2.0 m`）上，单阶段端到端同步训练 `GRSN` 估计器、脉冲 `Actor`、特权 `Critic` 与 `AMP` 判别器。零教师权重、零教师动作、零模仿热启动。
2. **一次性 `5000` 轮铁律**：
   - 每次正式长训预算固定为 **`4096 环境 × 100 步 × 5000 轮`**（单轮 `409,600` 步，总计 `2.048B` 控制步，等价于 `2048 × 10000` 轮且单卡实测吞吐高 `~24%`，约 `9.8~10.3 s/轮`，全程约 `13.5~14.5` 小时）。
   - **到 `5000` 轮即停止**，不做续训微调；
   - **中途一旦因崩溃或停滞被终止，分析并修复根因后必须从第 `0` 轮重新开始**，绝不从异常中途检查点续训。
3. **成功门槛**：
   - **S-HIM 达标线**：在单次 `<= 5000` 轮从零长训中，稳定实现 **`terrain_level > 5.0`**（并在 `eval_shim_panel.py` 平地 `+0.6 / 0.0 / -0.6 m/s` 及 `0.04 / 0.08 / 0.12 / 0.16 m` 双段楼梯评测中验证良好的盲走与越障迁移效果，零跌倒或极低跌倒率）。
   - **同条件基线对比**：在取得达标的 S-HIM 长训模型后，使用**完全相同的奖励预设、AMP 对角步态数据与权重、地形、并行规模与轮次**训练 `HIMLoco + AMP`（ANN 基线），最后完成全维度横向对比（地形等级、平地速度追踪、四档楼梯通过率与失败归因、导出一致性 `check_and_export`、45 nm 突触能耗 `SynapticMeter` 与可视化步态视频）。

---

## 二、统一环境、奖励与 AMP 配置

| 项目 | 固定配置 | 说明 |
| :--- | :--- | :--- |
| 地形课程 | `--terrain stairs`（`CompactPlatformStairTerrain`） | 保留原版 `0.05 + 0.18 * difficulty`（`num_rows=10`，`level 0..9`），出生平台 `2.0 m`，60% 上楼梯 / 15% 下楼梯 |
| 奖励预设 | `--reward-preset snn-stair` | `base_height_target=0.30, tracking_lin_vel=3.0, tracking_ang_vel=1.2, base_height=-0.5, stumble=-0.1, collision=-0.25, stand_still=-0.5, dof_acc=-5e-8, joint_power=-5e-6, action_rate=0, smoothness=0` |
| AMP 专家数据 | `--amp --amp-motion-set a1_diagonal` | 仅加载 `third_party/WMP/datasets/mocap_motions/{trot1.txt, trot2.txt}`（A1 对角抬腿前进，`0.39 / 0.80 m/s`，`9.5~10.1 cm` 摆动抬脚，零腾空零弹跳零侧偏；严格剔除 `hop1/hop2` 小跳与不对称动捕） |
| AMP 融合方式 | `--amp-reward-coef 0.4 --amp-task-lerp 0.0 --amp-discr-updates 4 --amp-discr-lr 1e-4 --amp-batch-size 1024 --amp-preload 200000` | 叠加式温和步态引导，避免压制后退、转向与高阶攀爬自适应 |
| 优化与防发散 | `--lr-schedule adaptive --learning-rate 8e-4 --max-lr 8e-4 --desired-kl 0.045 --aux-learning-rate 3e-4 --entropy-coef 0.002 --max-action-std 0.55 --log-std-lr-scale 0.05 --max-grad-norm 5.0 --separate-grad-clip --actor-weight-decay 1e-4` | 防止 `action_std` 爆炸与 `lr` 失控冲高/锁死 |
| 校准与留证 | `--normalization care --freeze-aux-estimator-stats --balanced-calibration --ppo-rollout-features --calibration-interval 5 --diagnostics-interval 10 --checkpoint-interval 250` | 跨 5 轮环形累积大样本后 `commit()`，每 250 轮存盘 |

---

## 三、长训执行序列

### 阶段 1：S-HIM 主力长训（`L1_shim_nlif_diag_4096`）
- **架构**：`GRSN (CaRe-BN, width=64, layers=2) + LIFActor (NLIF 潜态读出, actor_norm=none, Rect, threshold=0.5, ticks=8, widths=256 128, population=5)`。
- **启动命令**：
```bash
/home/qyw/miniconda3/envs/himloco/bin/python scripts/bench_blind.py s-him \
  --num-envs 4096 --steps 100 --seed 1 --iterations 5000 \
  --terrain stairs --reward-preset snn-stair --paper-training \
  --estimator grsn --normalization care \
  --actor-type lif --actor-norm none --actor-readout-mode nlif \
  --actor-surrogate rect --actor-threshold 0.5 --ticks 8 --actor-widths 256 128 \
  --learning-rate 8e-4 --aux-learning-rate 3e-4 --lr-schedule adaptive \
  --desired-kl 0.045 --max-lr 8e-4 --max-grad-norm 5.0 --separate-grad-clip \
  --log-std-lr-scale 0.05 --entropy-coef 0.002 --max-action-std 0.55 \
  --actor-weight-decay 1e-4 --freeze-aux-estimator-stats \
  --balanced-calibration --ppo-rollout-features \
  --diagnostics-interval 10 --calibration-interval 5 --checkpoint-interval 250 \
  --amp --amp-motion-set a1_diagonal --amp-reward-coef 0.4 --amp-task-lerp 0.0 \
  --amp-discr-lr 1e-4 --amp-discr-updates 4 --amp-batch-size 1024 \
  --amp-replay-size 1000000 --amp-preload 200000 --amp-discr-widths 1024 512 \
  --headless --out runs/s-him-long/L1_shim_nlif_diag_4096
```

### 阶段 2（按需/备选）：若 `L1` 需排错或补充全二值脉冲对照（`L2_shim_spike_diag_4096`）
- 若 `L1` 达到 `terrain_level > 5.0` 后需要纯二值脉冲对比，可同配置将 `--actor-readout-mode nlif` 改为 `--actor-readout-mode spike` 从零跑 `5000` 轮；
- 若 `L1` 中途触发早停，根据诊断根因（如 `amp_reward_coef` 偏大/偏小、`desired_kl` 过紧导致 `lr` 锁死、`max-action-std` 过窄或奖励比例）调整后新建 `L1_v2_*` 目录**从 `0` 轮重新起跑**。

### 阶段 3：同条件 `HIMLoco + AMP`（ANN 基线，`B1_himloco_amp_diag_4096`）
- **在 S-HIM 取得满意的 `terrain_level > 5.0` 成果后启动**，严格采用与胜出 S-HIM 完全相同的奖励预设、AMP 对角步态数据、AMP 权重、地形、并行规模与核心优化参数：
```bash
/home/qyw/miniconda3/envs/himloco/bin/python scripts/bench_blind.py himloco \
  --num-envs 4096 --steps 100 --seed 1 --iterations 5000 \
  --terrain stairs --reward-preset snn-stair \
  --learning-rate 8e-4 --lr-schedule adaptive --desired-kl 0.045 --max-lr 8e-4 \
  --max-grad-norm 5.0 --entropy-coef 0.002 --max-action-std 0.55 \
  --diagnostics-interval 10 --checkpoint-interval 250 \
  --amp --amp-motion-set a1_diagonal --amp-reward-coef 0.4 --amp-task-lerp 0.0 \
  --amp-discr-lr 1e-4 --amp-discr-updates 4 --amp-batch-size 1024 \
  --amp-replay-size 1000000 --amp-preload 200000 --amp-discr-widths 1024 512 \
  --headless --out runs/s-him-long/B1_himloco_amp_diag_4096
```
> 注：若 `L1` 在调参重跑过程中更新了奖励、AMP 权重或优化参数，`B1` 将严格同步最终胜出的 S-HIM 参数配置。

---

## 四、每 30 分钟监督检查与早停重来规则

训练启动后，每 **30 分钟**（约推进 `170~180` 轮）读取最新 `console.log`，检查以下健康指标并记录里程碑：

1. **硬崩溃（立即终止 -> 定位根因 -> 从零重训）**：
   - 进程异常退出、CUDA OOM，或任意指标出现 `NaN / Inf`；
   - `explained_variance` 持续为负且 `mean_reward` 断崖暴跌超过 `50%` 且 `> 100` 轮无恢复；
   - 脉冲死寂：`actor_layer_silent_ratios` 任一层超过 `0.25` 或隐层发放率趋近 `0` / `1`。
2. **长期停滞（确认无救后终止 -> 定位根因 -> 从零重训）**：
   - **冷启动失败**：到第 `500` 轮时 `terrain_level < 0.3` 且 `command_aligned_velocity_mps < 0.15 m/s`；
   - **学习率锁死**：连续 `>= 300` 轮 `learning_rate` 被锁在 `<= 2e-5` 且 `terrain_level` 停滞；
   - **中后期平台停滞**：超过 `1000` 轮时，连续 `>= 800` 轮 `terrain_level` 毫无净增长（波动 `< 0.15`）且停在 `< 4.5`（明显冲不上 `level 5.0`），或持续倒退 `> 0.8` 级且无恢复趋势。
3. **正常推进判据**：
   - `0 -> 500` 轮：平地起跑破冰，`terrain_level` 预期攀至 `1.5 ~ 3.0+`；
   - `500 -> 2000` 轮：`terrain_level` 稳步迈向 `4.0 ~ 5.0+`，`command_aligned_velocity_mps >= 0.45 m/s`；
   - `2000 -> 5000` 轮：稳定突破并保持 **`terrain_level > 5.0`**（力争 `5.5 ~ 6.5+`），跑满 `5000` 轮自然结束。

---

## 五、最终验收与对比交付

每条跑满 `5000` 轮的主模型及 `HIMLoco + AMP` 基线完成后，统一执行：
1. **平地 + 四档楼梯全套评测**：`scripts/eval_shim_panel.py --panel all --num-envs 16 --steps 800 --step-heights 0.04 0.08 0.12 0.16 --forward-speed 0.6 --seed 21`（以及高档大样本 `--num-envs 50` 留存种子评测）；
2. **部署融合一致性与 45 nm 能耗测量**：验证 `check_and_export`（`fusion_max_error <= 1e-5`），通过 `bench_blind.py --eval --energy` 统计 S-HIM 与 `HIMLoco + AMP` 的脉冲率、AC/MAC 次数与能耗比；
3. **可视化步态留证**：通过 `scripts/play_shim.py` 录制 S-HIM 与 `HIMLoco + AMP` 的平地与高台阶攀爬 MP4 与关键帧对比图；
4. **单元测试回归**：`/home/qyw/miniconda3/envs/himloco/bin/python -m unittest discover -s tests -v` 全部通过。

# S-HIM 单阶段盲走复杂地形长训实验方案

日期：2026-11-28 · 平台：Unitree A1 · Isaac Lab / PhysX · 单卡 RTX 5060 Ti 16GB

本方案替代原有短训诊断方案（第一阶段短训结果与问题诊断已归档至 [`S-HIM第一次大范围短训方向总结.md`](S-HIM第一次大范围短训方向总结.md)），用于指导 **单阶段 `S-HIM` 在原版全套盲走复杂地形与渐进速度课程下，一次性 `5000` 轮长训冲击 `terrain_level > 5.0`**，以及同条件下 **`HIMLoco + AMP` ANN 基线**的公平对比。

> **2026-11-28 09:40 口径校准记录（`L1` 提前终止 -> 切换 `L2`）**：
> - `L1_shim_nlif_diag_4096`（纯楼梯 `--terrain stairs`，固定 `[-1, 1]` 速度范围）跑至 `Iter 3500` 时已达 **`terrain_level = 6.2881`（峰值）**，可视化验证 `0.12 m` 双层高台阶全程零摔倒登顶、`0.16 m` 爬至顶层最后一级、平地与粗糙地形稳定直走。
> - 按最新明确要求对齐原版盲爬任务：**地形改为原版五类复杂粗糙地形（`--terrain rough`）**，**速度指令改为渐进速度课程（`[-0.2, 0.2] -> [-2.0, 2.0]`，同时修复上游 `common_step_counter % max_episode_length == 0` 与 `episode_length_buf > max_episode_length` 错位 1 步导致存活机器人无法触发速度课程扩容的隐蔽 Bug）**；故终止 `L1`，从 `Iter 0` 正式启动 `L2_shim_rough_cmdcur_4096`。

---

## 一、核心目标与严格约束

1. **单阶段全盲走定义**：从随机初始化（`iter = 0`）出发，在原版 HIMLoco 五类盲走复杂课程地形（`--terrain rough`：10% 平缓斜坡 + 20% 粗糙斜坡 + 30% 上楼梯 + 30% 下楼梯 + 10% 离散障碍物，`num_rows=10`、`num_cols=20`、`max_init_terrain_level=5`）与渐进速度课程（`lin_vel_x: [-0.2, 0.2] -> [-2.0, 2.0]`，`lin_vel_y: [-0.2, 0.2] -> [-1.0, 1.0]`）上，单阶段端到端同步训练 `GRSN` 估计器、脉冲 `Actor`、特权 `Critic` 与 `AMP` 判别器。零教师权重、零教师动作、零模仿热启动。
2. **一次性 `5000` 轮铁律**：
   - 每次正式长训预算固定为 **`4096 环境 × 100 步 × 5000 轮`**（单轮 `409,600` 步，总计 `2.048B` 控制步，等价于 `2048 × 10000` 轮，约 `9.3~10.0 s/轮`，全程约 `13~14` 小时）。
   - **到 `5000` 轮即停止**，不做续训微调；
   - **中途一旦因崩溃或停滞被终止，分析并修复根因后必须从第 `0` 轮重新开始**，绝不从异常中途检查点续训。
3. **成功门槛**：
   - **S-HIM 达标线**：在单次 `<= 5000` 轮从零长训中，稳定实现 **`terrain_level > 5.0`** 且速度课程正常爬升（记录完整 `command_range_x / command_range_y` 演进轨迹），并在平地、`0.04 / 0.08 / 0.12 / 0.16 m` 双段楼梯及粗糙地形可视化评测中验证良好的全地形盲走与越障能力。
   - **同条件基线对比**：在取得达标的 S-HIM 长训模型后，使用**完全相同的地形、渐进速度课程、奖励预设、AMP 对角步态数据与权重、并行规模与轮次**训练 `HIMLoco + AMP`（ANN 基线），最后完成全维度横向对比（地形等级、速度课程演进、平地与四档楼梯通过率、导出一致性 `check_and_export`、45 nm 突触能耗 `SynapticMeter` 与可视化步态视频）。

---

## 二、统一环境、速度课程、奖励与 AMP 配置

| 项目 | 固定配置 | 说明 |
| :--- | :--- | :--- |
| 地形课程 | `--terrain rough`（原版 `A1RoughCfg` 五类复杂障碍） | `terrain_proportions=[0.1, 0.2, 0.3, 0.3, 0.1]`（平缓斜坡/粗糙斜坡/上楼梯/下楼梯/离散障碍），`num_rows=10 (level 0..9)`，`num_cols=20`，`max_init_terrain_level=5` |
| 渐进速度课程 | `--cmd-curriculum-init-x 0.2 --cmd-curriculum-max-x 2.0 --cmd-curriculum-step 0.2 --cmd-curriculum-threshold 0.8` | 初始 `lin_vel_x/y = [-0.2, 0.2]`；每满一个完整回合窗口（`1000` 步），高低速组追踪奖励均 `> 80%` 满分即对称步进 `±0.2`：`lin_vel_x` 扩至 `[-2.0, 2.0]`、`lin_vel_y` 扩至 `[-1.0, 1.0]`（超过 `1.0 m/s` 时按原版 HIMLoco 仅前 20% 高速组探索高速冲刺且高速时侧向置零，后 80% 保持 `[-1.0, 1.0]` 全向越障） |
| 奖励预设 | 任务配置内置 `snn-stair` 奖励 | `base_height_target=0.30, tracking_lin_vel=3.0, tracking_ang_vel=1.2, base_height=-0.5, stumble=-0.1, collision=-0.25, stand_still=-0.5, dof_acc=-5e-8, joint_power=-5e-6, action_rate=0, smoothness=0` |
| AMP 专家数据 | `--amp --amp-motion-set a1_diagonal` | 仅加载 `third_party/WMP/datasets/mocap_motions/{trot1.txt, trot2.txt}`（A1 对角抬腿前进，`0.39 / 0.80 m/s`，`9.5~10.1 cm` 摆动抬脚，零腾空零弹跳零侧偏；严格剔除 `hop1/hop2` 小跳与不对称动捕） |
| AMP 融合方式 | `--amp-reward-coef 0.5 --amp-task-lerp 0.0 --amp-discr-updates 2 --amp-discr-lr 5e-5 --amp-batch-size 1024 --amp-preload 200000` | 相对 `L1`（`updates=4, lr=1e-4` 导致判别器 100 轮即饱和 `±1.0`）适度放缓判别器步速，使对角抬腿风格梯度在速度课程爬升期间持续有效引导步态 |
| 优化与防发散 | `--lr-schedule adaptive --learning-rate 8e-4 --max-lr 8e-4 --desired-kl 0.045 --aux-learning-rate 3e-4 --entropy-coef 0.002 --max-action-std 0.55 --log-std-lr-scale 0.05 --max-grad-norm 5.0 --separate-grad-clip --actor-weight-decay 1e-4` | 防止 `action_std` 爆炸与 `lr` 失控冲高/锁死 |
| 校准与留证 | `--normalization care --freeze-aux-estimator-stats --balanced-calibration --ppo-rollout-features --calibration-interval 5 --diagnostics-interval 10 --checkpoint-interval 250` | 跨 5 轮环形累积大样本后 `commit()`，每 250 轮存盘；每轮记录 `command_range_x / command_range_y` |

---

## 三、长训执行序列

### 阶段 1：S-HIM 全地形+渐进速度课程主力长训（`L1_shim_nlif_rough_cmdcur_4096`）
- **架构**：`GRSN (CaRe-BN, width=64, layers=2) + LIFActor (NLIF 潜态读出, actor_norm=none, Rect, threshold=0.5, ticks=8, widths=256 128, population=5)`。
- **启动命令**：
```bash
/home/qyw/miniconda3/envs/himloco/bin/python scripts/bench_blind.py s-him --task s_him_a1_rough_cmdcur \
  --num-envs 4096 --steps 100 --seed 1 --iterations 5000 \
  --terrain rough \
  --cmd-curriculum-init-x 0.2 --cmd-curriculum-max-x 2.0 --cmd-curriculum-step 0.2 --cmd-curriculum-threshold 0.8 \
  --paper-training \
  --estimator grsn --normalization care \
  --actor-type lif --actor-norm none --actor-readout-mode nlif \
  --actor-surrogate rect --actor-threshold 0.5 --ticks 8 --actor-widths 256 128 \
  --learning-rate 8e-4 --aux-learning-rate 3e-4 --lr-schedule adaptive \
  --desired-kl 0.045 --max-lr 8e-4 --max-grad-norm 5.0 --separate-grad-clip \
  --log-std-lr-scale 0.05 --entropy-coef 0.002 --max-action-std 0.55 \
  --actor-weight-decay 1e-4 --freeze-aux-estimator-stats \
  --balanced-calibration --ppo-rollout-features \
  --diagnostics-interval 10 --calibration-interval 5 --checkpoint-interval 250 \
  --amp --amp-motion-set a1_diagonal --amp-reward-coef 0.5 --amp-task-lerp 0.0 \
  --amp-discr-lr 5e-5 --amp-discr-updates 2 --amp-batch-size 1024 \
  --amp-replay-size 1000000 --amp-preload 200000 --amp-discr-widths 1024 512 \
  --headless --out runs/s-him-long/L1_shim_nlif_rough_cmdcur_4096
```

### 阶段 2：同条件 `HIMLoco + AMP`（ANN 基线，`B1_himloco_amp_rough_cmdcur_4096`）
- **在 `L1` 取得满意的 `terrain_level > 5.0` 成果后启动**，严格采用与胜出 S-HIM 完全相同的五类复杂地形、渐进速度课程、奖励预设、AMP 对角步态数据与权重、并行规模与核心优化参数：
```bash
/home/qyw/miniconda3/envs/himloco/bin/python scripts/bench_blind.py --task himloco_a1_amp_rough_cmdcur \
  --num-envs 4096 --steps 100 --seed 1 --iterations 5000 \
  --terrain rough \
  --cmd-curriculum-init-x 0.2 --cmd-curriculum-max-x 2.0 --cmd-curriculum-step 0.2 --cmd-curriculum-threshold 0.8 \
  --learning-rate 8e-4 --lr-schedule adaptive --desired-kl 0.045 --max-lr 8e-4 \
  --max-grad-norm 5.0 --entropy-coef 0.002 --max-action-std 0.55 \
  --diagnostics-interval 10 --checkpoint-interval 250 \
  --amp --amp-motion-set a1_diagonal --amp-reward-coef 0.5 --amp-task-lerp 0.0 \
  --amp-discr-lr 5e-5 --amp-discr-updates 2 --amp-batch-size 1024 \
  --amp-replay-size 1000000 --amp-preload 200000 --amp-discr-widths 1024 512 \
  --headless --out runs/s-him-long/B1_himloco_amp_rough_cmdcur_4096
```

---

## 四、每 30 分钟监督检查与早停重来规则

训练启动后，每 **30 分钟**（约推进 `185~195` 轮）读取最新 `console.log`，检查以下健康指标并记录里程碑：

1. **硬崩溃（立即终止 -> 定位根因 -> 从零重训）**：
   - 进程异常退出、CUDA OOM，或任意指标出现 `NaN / Inf`；
   - `explained_variance` 持续为负且 `mean_reward` 断崖暴跌超过 `50%` 且 `> 100` 轮无恢复；
   - 脉冲死寂：`actor_layer_silent_ratios` 任一层超过 `0.25` 或隐层发放率趋近 `0` / `1`。
2. **长期停滞（确认无救后终止 -> 定位根因 -> 从零重训）**：
   - **冷启动失败**：到第 `500` 轮时 `terrain_level < 0.3` 且 `command_range_x` 仍停留在 `[-0.2, 0.2]` 未扩容；
   - **学习率锁死**：连续 `>= 300` 轮 `learning_rate` 被锁在 `<= 2e-5` 且 `terrain_level` 停滞；
   - **中后期平台停滞**：超过 `1000` 轮时，连续 `>= 800` 轮 `terrain_level` 毫无净增长（波动 `< 0.15`）且停在 `< 4.5`（明显冲不上 `level 5.0`），或持续倒退 `> 0.8` 级且无恢复趋势。
3. **重点监控新增项**：
   - 每 30 分钟同步追踪 `command_range_x` 与 `command_range_y` 扩容进程（确认从 `[-0.2, 0.2]` 顺利逐步爬升至 `[-1.0, 1.0]` 及 `[-2.0, 2.0]`）和 `amp_style_reward_per_sec` / `amp_expert_d` / `amp_policy_d`（确认判别器不早熟饱和）。

---

## 五、最终验收与对比交付

每条跑满 `5000` 轮的主模型及 `HIMLoco + AMP` 基线完成后，统一执行：
1. **平地 + 四档楼梯全套评测**：`scripts/eval_shim_panel.py --panel all --num-envs 16 --steps 800 --step-heights 0.04 0.08 0.12 0.16 --forward-speed 0.6 --seed 21`（并补充高速平地指令测试）；
2. **部署融合一致性与 45 nm 能耗测量**：验证 `check_and_export`（`fusion_max_error <= 1e-5`），通过 `bench_blind.py --eval --energy` 统计 S-HIM 与 `HIMLoco + AMP` 的脉冲率、AC/MAC 次数与能耗比；
3. **可视化步态留证**：通过 `scripts/play_shim.py` 录制 S-HIM 与 `HIMLoco + AMP` 在平地、`0.12 m / 0.16 m` 高台阶与复杂粗糙地形的 MP4 与关键帧对比图；
4. **单元测试回归**：`/home/qyw/miniconda3/envs/himloco/bin/python -m unittest discover -s tests -v` 全部通过。

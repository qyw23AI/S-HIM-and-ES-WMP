# A1 复现与评测手册

以下命令沿用原入口。实验流程位于 `graduation/experiments/`，方法代码位于 `graduation/methods/`，共享环境工厂位于 `graduation/sim/`。各入口在进程内选择对应的第三方依赖。[当前目录与迁移验证](重构方案与实验改进清单.md)

## 环境和资产

从 `Graduation project` 根目录运行。先确认可用的 CUDA 设备。现有 Conda 复用本机 Isaac Lab 2.3.2、Isaac Sim 5.1.0 和 PyTorch 2.7.0+cu128。主要仿真依赖及 A1 资产已在本机和仓库内。两份迁移的 `rsl_rl` 由各自入口设定搜索路径。

```bash
HIM_PY=/home/qyw/miniconda3/envs/himloco/bin/python
WMP_PY=/home/qyw/miniconda3/envs/wmp_lab/bin/python
nvidia-smi
"$HIM_PY" -c 'import torch, isaacsim; print(torch.cuda.is_available(), isaacsim.__file__)'
"$WMP_PY" -m unittest discover -s tests -v
```

## 盲走基线和 S-HIM

```bash
"$HIM_PY" third_party/MyHIMLoco/isaaclab_migration/scripts/train.py \
  --task a1 --stage 4 --num_envs 8 --rollout_steps 8 --iterations 1 --headless \
  --log_dir runs/himloco --export_policy runs/himloco/policy.pt
for method in ppo-mlp fully-snn s-him; do
  "$HIM_PY" scripts/bench_blind.py "$method" --headless --num-envs 8 \
    --steps 12 --iterations 2 --out "runs/$method"
done
"$HIM_PY" scripts/bench_blind.py s-him --headless --num-envs 8 --steps 16 \
  --eval --energy --checkpoint runs/s-him/model.pt --out runs/s-him-clean
"$HIM_PY" scripts/bench_blind.py s-him --headless --num-envs 8 --steps 16 \
  --eval --energy --checkpoint runs/s-him/model.pt --impulse 1 --impulse-step 5 \
  --out runs/s-him-impulse
"$HIM_PY" scripts/bench_blind.py s-him --headless --num-envs 8 --steps 16 \
  --eval --energy --checkpoint runs/s-him/model.pt --imu-noise .15 --out runs/s-him-imu
```

- RMA 保留作者的 PopSAN、特权编码和历史适应网络，统一对照使用 HIMLoco 的 45 维本体帧与 12 个关节目标。作者默认 RMA 的 PMTG、320 维输入、16 维策略输出和 40 帧历史是不同的实验设置。可分别使用 `--rma-phase policy` 和 `--rma-phase adaptation --checkpoint ...` 验证两段，也可按作者发布实现的更新思路运行 `joint`。
- S-HIM 使用 5 帧、3 维速度与 16 维潜变量、两层 GRSN、LIF 群体读出。控制环境保留一阶和二阶动作平滑代价。消融开关为 `--normalization care|tdbn|none`、`--swap-weight 0`、`--estimator lif` 等。正值横向冲量沿机体 +Y，以 `J/physics_dt` 施加一物理子步。陀螺仪噪声使用独立随机流。
- 可选的官方 FullySNN A1 AMP 是原地面走配置，含公开专家动作、参考姿态初始化、AMP 和 PopSAN。策略 42 维、特权 48 维、AMP 43 维，控制周期 30 ms。该配置的对照口径与 20 ms 的共享盲走不同。

```bash
"$WMP_PY" scripts/bench_fully_snn_amp.py --headless --out runs/fully-snn-amp
"$WMP_PY" scripts/bench_fully_snn_amp.py --headless --eval --steps 12 \
  --checkpoint runs/fully-snn-amp/model.pt --out runs/fully-snn-amp-eval
```

## WMP 教师、ES-Parkour 与 Event-Spiking-WMP

短回合用于在两轮内产生可训练的 WMP 世界模型回放。正式训练应按独立实验配置保存自然回合长度与并行数。

```bash
"$WMP_PY" third_party/WMP/isaaclab_migration/scripts/train.py --task a1_amp \
  --num_envs 4 --camera_envs 4 --iterations 2 --episode_seconds .3 \
  --rollout_steps 24 --wm_batch 1 --wm_length 2 --wm_start 0 --wm_updates 1 \
  --amp_preload 64 --amp_replay_size 512 --depth_batch 8 --terrain mixed \
  --require_wm_updates --headless --log_dir runs/wmp
teacher=artifacts/teachers/wmp-a1-15000.pt
"$WMP_PY" scripts/bench_parkour.py wmp-events --headless --num-envs 4 \
  --steps 20 --checkpoint "$teacher" --out runs/wmp-events
"$WMP_PY" scripts/bench_parkour.py teacher --headless --num-envs 4 \
  --iterations 2 --steps 25 --out runs/es-teacher
"$WMP_PY" scripts/bench_parkour.py distill --headless --num-envs 4 \
  --iterations 2 --warmup-iterations 1 --steps 10 \
  --teacher runs/es-teacher/teacher.pt --out runs/es-parkour
"$WMP_PY" scripts/train_event_wmp.py --headless --num-envs 4 \
  --iterations 2 --rollout-steps 15 --teacher "$teacher" --out runs/es-wmp
```

- ES-Parkour 的四类赛道为沟壑、台阶、栏障和复合落脚点。教师读扫描点和目标航向，学生用 IF ResNet-18、浮点 GRU 和三层脉冲 MLP，50 Hz 控制和 10 Hz 事件采样。动作和航向分别监督。默认软配置降低接触惩罚并允许负奖励，`--reward-profile strict` 切回 Extreme Parkour 对照权重与奖励裁剪。原文方法和实验章节对通用 LIF 与实验 IF 有不同描述，实验实现按 IF、4 个脉冲步。细节选择和限制见总报告。
- Event-Spiking-WMP 采用冻结教师的 512 维确定性 RSSM 特征，加动作 MSE、隐状态 MSE 和余弦损失。训练入口支持教师暖身、学生驱动、`--action-path end-to-end|actor`、`--world-weight 0` 和 `--memory grsn|lif|gru`。输出可重开训练的 `student.pt` 和融合归一化的 `policy.pt`。
- 冻结文件可以用 `scripts/freeze_wmp.py 信任的本机检查点 输出路径` 从原训练检查点重建。迁移版 WMP 检查点保留 Python 反序列化对象，入口仅用于可信本机文件。
- 事件模拟依赖深度伪亮度与 EOLO 风格随机流向，使用双极性二值事件帧，可独立输入已知光流做单元验证。真实光照和硬件动态范围需单独测试。

## 同场比较和汇总

同一评测入口加载四个方法，固定种子、场景、控制和成功距离。使用与研究问题匹配的地形和门槛，记录成功、终止、超时或观察窗截止、跟踪误差、发放率和 45 nm 理论突触能耗。保留单层操作统计。WMP 同场接口包括推理路径中实际执行的 CNN、RSSM 仿射运算和策略层。忽略访存、控制电路、模拟神经元和其他逐元素运算。

```bash
"$WMP_PY" scripts/eval_visual.py event-spiking-wmp --headless --num-envs 4 \
  --steps 20 --energy --checkpoint runs/es-wmp/student.pt --out runs/visual/es-wmp
"$WMP_PY" scripts/eval_visual.py event-spiking-wmp --headless --num-envs 4 \
  --steps 20 --energy --delay-frames 1 --frame-stride 2 --blur 3 --pause-at 5 --pause-steps 5 \
  --checkpoint runs/es-wmp/student.pt --out runs/visual/es-wmp-degraded
"$WMP_PY" scripts/summarize_experiments.py runs/visual --out runs/summary
```

对另外三个方法以相同参数运行 `wmp`、`es-parkour` 和 `s-him`，各传入相应的检查点。v2 成功率的分母为首次判定成功、失败或超时的回合。观察窗截尾另计，零步重置回合省略。同场正式记录按相同条件与共同种子汇总，保守首层 MAC 和严格事件 AC 分别生成帕累托前沿。旧记录和冒烟记录保留在 CSV。WMP 教师轨迹录制与同场评测均输出策略和世界模型计量，录制依然提供 `actor_affine_energy` 供既有读取方使用。 [改前改后短验证](../artifacts/evaluation-protocol-v2/README.md)

模型变体、解析光流和速度指令停顿的入口、短验证配置及结果见 [剩余 P0 短验证](../artifacts/p0-validation/README.md)。

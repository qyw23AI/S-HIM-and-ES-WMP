# 剩余 P0 短验证

2026-11-26。分为模型侧和传感器评测侧。项目目录已应用两批改动。工作区的 Git 元数据不可用，两份补丁由临时仓库生成，供正常 Git 检出按顺序应用。 [模型补丁](0001-model-variants.patch) · [传感器与评测补丁](0002-event-stop-and-go.patch)

## S-HIM 与 ES-WMP 静态首层

共享 LIFActor 将静态线性投影移到时间展开前。随后仍展开到 4 个脉冲步再做归一化，因此 CaRe-BN 的有效样本数、置信度更新和重校准流程得到保留。旧参数名和张量形状保持兼容。可使用 `--legacy-actor-projection` 对照。

| 旧检查点 | 首层 MAC/决策，改前 → 改后 | 动作和脉冲最大差 | 最大梯度绝对差 |
|---|---:|---:|---:|
| S-HIM | 32,768 → 8,192 | 0 | (5.96 × 10⁻⁸) |
| ES-WMP | 570,368 → 142,592 | 0 | (1.97 × 10⁻⁶) |

CPU 验证使用两份旧权重、固定输入和相同损失。检查 `fixed`、`adapt`、`calibrate` 的动作、各 tick 脉冲、输入及参数梯度、归一化缓冲区。重校准时每个隐藏层看到的样本数仍为 32。浮点梯度按容差比较。 [实测 JSON](actor-equivalence.json)

### GRSN 单变量短训

每组 4 个环境、每轮 16 步、3 轮、种子 1。下面的百分比区间覆盖两层、全部三轮，计的是每层实际产生的输出脉冲。每轮日志另有阈值和预重置膜电位 P05/P50/P95。

| 选项 | 发放率范围 | 原始记录 |
|---|---:|---|
| 默认配置 | 2.99%–3.51% | [结果](shim-default/validation.json) |
| 只改阈值 `0.60` | **21.68%–22.26%** | [结果](shim-threshold-060/validation.json) |
| 只改初始归一化偏置 `1.15` | **28.59%–29.76%** | [结果](shim-bias-115/validation.json) |
| 只改阈值 `0.45` | 37.51%–39.20% | [结果](shim-threshold/validation.json) |
| 只改初始归一化偏置 `1.45` | 45.49%–47.11% | [结果](shim-bias/validation.json) |

新配置使用 `bench_blind.py s-him --grsn-threshold .60` 或 `--grsn-bias 1.15`，两者互斥。旧检查点采用默认阈值及保存的偏置，调优检查点保存自己的阈值配置。`--grsn-diagnostics` 可用于旧模型，适合把输出发放率与历史门输入统计分开看。10%–35% 是本批短训筛选区间，长训仍需跟踪速度估计和任务表现。

## ES-Parkour 事件响应

三组统一用保留的 ES-Parkour 特权教师、16 个基础通道、4 个环境、每轮 15 步、三轮蒸馏（两轮教师驱动、一轮学生驱动），种子 1。探针使用相同本体输入和 GRU 初始状态，将真实事件帧替换为零帧。日志包含各视觉块的活跃比例及真帧、零帧输出差，以及 Actor 各下游层的输入活跃比例。

| 配置 | GRU 最大差 | 动作最大差 | 记录 |
|---|---:|---:|---|
| 标准 IF、BatchNorm | 0 | 0 | [短训](es-baseline/validation.json) |
| tdBN、阈值 0.5、SEW 加法 | 0.13471 | 0.19135 | [短训](es-tdbn/validation.json) |
| CaRe-BN、阈值 0.5、SEW 加法 | 0.16162 | 0.24521 | [短训](es-care/validation.json) |

两个变体的 blocks.4–7 与 actor.2–3 输入或输出均有活跃位点。以 tdBN 为例，blocks.4–7 非零比例为 48.66%–61.68%，actor.2–3 的输入为 32.45% 和 32.28%。原 64 通道旧检查点仍可直接加载，探针复现 GRU 和动作零差。SEW 加法可能输出大于 1 的整数，接收多值输入的卷积走 dense MAC，分支内二值层仍走 AC。 [旧检查点探针](es-legacy/validation.json) · [SEW 回放计量](es-sew-energy/validation.json)

入口为 `bench_parkour.py distill --student-normalization tdbn --student-threshold .5 --student-residual sew-add --visual-response-probe`。归一化、阈值和残差可独立选择，表中的两个组合属于同预算结构对照，各单项的贡献仍需消融。

## 解析光流与相机时钟

随机光流仍是默认。`--flow-mode analytical` 读取同一次采样的深度、内参、光学位姿。当前像素反投影到三维后重投影至上一视觉帧，得到按当前像素对齐的前向位移。视野外、近远裁剪、被上一帧几何遮挡和跨 reset 的像素不生成位移。事件振幅来自深度伪亮度梯度。零位姿变化精确产生零事件；停顿时的机身微动仍可产生事件。单测覆盖静止边缘、横移符号与量级、纯旋转、异步重置和制动后恢复。

额外使用 `--frame-stride 2 --delay-frames 1`。原生深度每 5 个控制步更新，送入策略的帧每 10 步交付。第 20 步交付给 WMP 的深度采自第 5 步（原策略读取 `-2`），给事件学生的帧采自第 10 步（读取 `-1`）。WMP 的 RSSM 仍每 5 步更新，间隔期间复用上一张已交付深度。时间戳 `-1` 表示初始延迟填充或重置后无效帧。 [WMP 时钟](clock-wmp/validation.json) · [学生时钟](clock-event/validation.json)

## 速度指令型 Stop-and-Go

Isaac Lab 短验使用四条跑酷赛道、4 个环境、种子 1、130 步。触发要求距离第一障碍不超过 1.6 m，且已运行 15 步。保持时长为 15 个连续控制步（0.30 s），平面速度带为 0.15 m/s。策略在制动和停稳时仍输出关节动作。ES-WMP 的实际触发距离约 1.14–1.47 m，这个较宽窗口用于让现有短训检查点触发状态机。

| 检查点／条件 | 进入恢复状态 | 保持阶段最高平面速度 | 接近 → 保持的交付事件密度 | 恢复后净前进范围 |
|---|---:|---:|---:|---:|
| ES-WMP GRSN，解析 | 4/4 | 0.1465 m/s | 7.664% → **3.214%** | −0.0061 至 0.0015 m |
| ES-WMP GRSN，随机加停顿屏蔽 | 4/4 | 0.1400 m/s | 1.196% → **0%** | −0.0048 至 0.0016 m |
| ES-WMP LIF，解析 | 4/4 | 0.1475 m/s | 6.789% → 0.378% | −0.0113 至 0.0034 m |
| ES-WMP GRSN，每步清状态，解析 | 4/4 | 0.1406 m/s | 7.767% → 4.116% | −0.0046 至 −0.0015 m |
| WMP 冻结教师 | 4/4 环境至少一次 | 0.1497 m/s | 深度输入 | 1.1357 至 1.5583 m（已恢复的回合） |

解析 GRSN 组在第 33–48 步恢复原指令，制动期间各环境实际策略动作的最大绝对值为 0.441–0.520。保持期的事件密度相对接近阶段下降 **58.1%**。随机屏蔽组仍记录未屏蔽的保持期原始密度 1.166%。ES-WMP 三种记忆条件和随机屏蔽条件的回合均未在此窗口内到达正式目标，现有短训权重恢复后几乎没有净推进。此轮数据确认了状态机、事件输入控制、清状态和恢复位移记录，正式记忆优势仍需用具备可靠行走能力的检查点测试。WMP 教师回放提供了恢复后实际向前行走的短验。 [解析 GRSN](stop-analytical/validation.json) · [随机屏蔽](stop-masked-random/validation.json) · [LIF](stop-lif/validation.json) · [清状态](stop-zero-state/validation.json) · [WMP 教师](stop-wmp/validation.json)

未指定停顿和光流模式时，ES-WMP 的 20 步随机基线与上一批结果逐项核对。有效事件总数 1,849、逐回合记录和速度跟踪误差一致。15 项测试通过。本批只跑了小批量短训和回放。 [随机基线复核](random-regression/validation.json)

### 复跑解析光流与停顿

```bash
WMP_PY=/home/qyw/miniconda3/envs/wmp_lab/bin/python
"$WMP_PY" scripts/eval_visual.py event-spiking-wmp --headless --terrain parkour \
  --num-envs 4 --seed 1 --steps 130 --energy --flow-mode analytical \
  --pause-at 15 --pause-trigger-distance 1.6 --pause-steps 15 --stop-speed .15 \
  --checkpoint artifacts/verified/es-wmp-fp32/student.pt --out runs/p0/stop-analytical
```

随机屏蔽组改用 `--flow-mode random --mask-events-on-pause`。清状态组在相同 GRSN 检查点上加 `--memory-control zero-state`。LIF 组加载本批用 `train_event_wmp.py --memory lif` 产生的 [短训权重](es-wmp-lif/student.pt)。`--pause-at` 是距离触发的最早控制步，省略 `--pause-trigger-distance` 时单独按步数触发。报告中的每环境恢复位移以恢复指令时的位置为起点。

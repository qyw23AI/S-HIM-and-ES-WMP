# 同场评测协议 v2 短验证

2026-11-26。使用 Isaac Lab 2.3.2、Isaac Sim 5.1、`wmp_lab`、PhysX 和 RTX 5060 Ti。每对运行使用同一冻结检查点、混合地形、4 个环境、种子 1、50 Hz 控制和 10 Hz 原生视觉更新。目录 [before](before) 和 [after](after) 提供逐回合、逐层计量和控制台记录。[同场行级 CSV](summary/runs.csv) 同时包含两版。教师轨迹录制不生成回合行，计量见各自的 JSON。

## S-HIM 观测与成功分母

固定首帧做配对前向，四个环境的示例见 [观测探针](after/s-him/validation.json)。例如，物理前进指令为 0.398825 m/s，旧观测输入为 0.398825，新输入为 0.797651。横摆指令、角速度、关节偏移及关节速度按两个实际 A1 配置转换，经过同名关节顺序和动作缩放核对。对同一初始状态，旧、新观测对应的 12 维策略动作最大绝对差为 0.041909。逐通道单测另检查 0.8 m/s 对应 1.6。

为在短窗口内实际产生回合结局，S-HIM 和 WMP 额外采用 0.2 秒回合、0.02 m 目标和 22 个控制步。目标距离用于验证统计逻辑。

| 方法 | 版本 | 成功 / 已结束 | 超时失败 | 窗口截尾 |
|---|---|---:|---:|---:|
| S-HIM | 改前 | 2 / 8 | 未单列 | 4 |
| S-HIM | 改后 | 2 / 8 | 6 | 0 |
| WMP（短回合） | 改前 | 4 / 8 | 未单列 | 4 |
| WMP（短回合） | 改后 | 4 / 8 | 4 | 0 |

改前多记的 4 行来自最后一步重置后尚无控制步的新回合。正常 20 步窗口中的 WMP、ES-Parkour、ES-WMP 共各有 4 个真实截尾，成功率字段仍为空。新协议按首次到点、失败或超时记一次。到点与失败同一步时算失败，到点与单纯超时同一步时算成功。到点后的物理回合可继续运行，已记结局保持不变。末步位置和速度从自动重置前读取。路径、超时、截尾和瞬时失败的回归断言见项目测试。此表用于验证分母，正式越障性能需要按正式目标与预算采集。

## 45 nm 理论突触能耗

单位为 µJ / 50 Hz 控制决策。标准窗口为 20 步。旧口径因入口而异。新口径仅对原始事件首层分别用 dense MAC 与按输入发放计的 AC，其余层规则一致。输出记载实际层调用、MAC、AC，及 WMP 世界模型的启动、稳态和重置批次。

| 方法和入口 | 改前 | 改后 · 保守首层 MAC | 改后 · 严格事件 AC |
|---|---:|---:|---:|
| WMP 同场 | 34.3940 | 34.3940 | 34.3940 |
| WMP 教师轨迹录制 | 0.8720（仅策略） | 34.3940 | 34.3940 |
| ES-Parkour 同场 | 25.0693（保守） | 25.0695 | 1.4895 |
| ES-WMP 同场 | 3.0190（严格） | 3.2896 | 3.0190 |

WMP 标准窗口的策略、深度 CNN、本体 MLP、RSSM 分别为 **0.8720、23.6349、3.8898、5.9973 µJ/决策**。世界模型合计 **33.5220 µJ/决策（97.46%）**。1 次启动批次和 3 次稳态批次共处理 16 张深度帧，分母为 80 个环境决策。启动批次每帧 37,026,816 MAC，稳态每帧 36,240,384 MAC。短回合验证还有 1 次重置批次，4 个环境都重置，累计 5 次世界模型更新，摊销后的全链路是 39.0886 µJ/决策。该窗口的控制步数和重置频率都不同。教师录制中的动作、RSSM 特征、事件和结束掩码在两版中逐值相等。

能耗范围限于现有推理链执行的仿射突触运算，45 nm 系数为 MAC 4.6 pJ、AC 0.9 pJ。表中的数字是运算模型估计值。ES-Parkour 两版保守值的微小差异来自两次运行的后续脉冲 AC 计数；同一改后轨迹的两列才是首层口径的配对比较。

## 复跑改后短验证

从项目根目录运行。视觉入口的公共参数见下方；其余三个同场方法用表中相应检查点，各自写入新的输出目录。

```bash
WMP_PY=/home/qyw/miniconda3/envs/wmp_lab/bin/python
"$WMP_PY" scripts/eval_visual.py s-him --headless --num-envs 4 --terrain mixed \
  --steps 22 --episode-seconds .2 --goal-distance .02 --seed 1 --energy \
  --checkpoint artifacts/verified/s-him/model.pt --out runs/protocol-v2/s-him
"$WMP_PY" scripts/eval_visual.py wmp --headless --num-envs 4 --terrain mixed \
  --steps 20 --seed 1 --energy --checkpoint artifacts/teachers/wmp-a1-15000.pt \
  --out runs/protocol-v2/wmp
"$WMP_PY" scripts/bench_parkour.py wmp-events --headless --num-envs 4 --terrain mixed \
  --steps 20 --seed 1 --energy --checkpoint artifacts/teachers/wmp-a1-15000.pt \
  --out runs/protocol-v2/wmp-record
```

| 补充入口 | 检查点及变更参数 |
|---|---|
| `eval_visual.py wmp` 短回合 | WMP 教师，采用与 S-HIM 相同的 22 步与短回合参数 |
| `eval_visual.py es-parkour` | `artifacts/verified/es-distill-final/student.pt`，20 步 |
| `eval_visual.py event-spiking-wmp` | `artifacts/verified/es-wmp-fp32/student.pt`，20 步 |

运行了 8 项单元测试。保存的实验记录均为冒烟用途。汇总脚本对同条件、共同种子的 v2 正式记录分别生成两套前沿，配对和截尾规则用合成用例校验。

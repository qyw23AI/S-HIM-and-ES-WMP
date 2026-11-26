# A1 脉冲强化学习毕业设计

项目覆盖 Isaac Lab A1 盲走、视觉跑酷、脉冲策略训练、同环境评估和结果汇总。初始实现面向短轮次功能验证，正式实验仍需训练、筛选检查点并增加随机种子。

- [任务计划](docs/plan.md)
- [代码重构方案、首轮验证与实验改进清单](docs/重构方案与实验改进清单.md)
- [运行手册](docs/runbook.md)
- [文献和 Zotero 条目](docs/papers.md)
- [源码与版本清单](docs/sources.json)
- [18 个上游仓库的实际用途和调用入口](third_party/upstream/README.md)
- [工程总报告](docs/毕业设计工程总报告.docx)

## 项目目录

| 路径 | 内容 |
|---|---|
| `third_party/MyHIMLoco` | 本机 HIMLoco 的 A1 迁移、RSL-RL、资产 |
| `third_party/WMP` | 本机 WMP 的 A1 迁移、AMP 专家数据、Dreamer、深度相机 |
| `third_party/upstream` | 公开源码按项目保留，提交 SHA 见版本清单 |
| `graduation/snn`、`rl`、`sensors` | 共享脉冲单元、归一化、PPO 和事件合成 |
| `graduation/sim` | Isaac Lab 环境工厂、赛道几何、相机缓冲和观测适配 |
| `graduation/methods` | 各方法的网络、教师、动作数据与任务专用实现 |
| `graduation/experiments` | 盲走、AMP、跑酷、蒸馏和视觉同场评测的运行流程 |
| `graduation/evaluation` | 能耗计数、评测汇总和帕累托前沿 |
| `scripts` | 原命令行入口，负责解析参数和启动对应流程 |
| `artifacts` | 验收记录、检查点、冻结的 WMP A1 教师与事件轨迹 |

所有仿真训练与回放都通过本机已安装的 Isaac Lab 2.3.2、Isaac Sim 5.1 和 PhysX。Conda `himloco` 负责盲走，`wmp_lab` 负责 WMP、跑酷与 FullySNN AMP。

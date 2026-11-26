# 毕设上游仓库使用清单

先按要运行的实验找依赖。FullySNN 的两个四足入口直接读取本目录。HIMLoco 和 WMP 的 Isaac Lab 实验从同级的本机迁移目录运行。仅做 S-HIM、ES-Parkour 或 ES-WMP 的现有复现时，无需逐一安装下面 18 个仓库。

下表的“使用”指项目现有训练、回放或评估入口对**这份本地检出**的依赖。算法同源、包同名和开发时参照分别在各行说明。核查依据为项目入口中的 import、路径注入、动态加载、资产读取及迁移配置。已逐仓核对远端 origin 和本地 HEAD。每个短 SHA 链接到该版本，完整提交和本机迁移来源见[源码锁定清单](../../docs/sources.json)。

## 运行时直接加载（1 个）

| 仓库与版本 | 关注内容 | 当前消费者或参考用途 |
|---|---|---|
| [FullySNNforLeggedRobots](FullySNNforLeggedRobots/RMA/quadruped-robot-master/rl-robotics/rsl_rl/rsl_rl/modules/actor_critic.py) · [a57791b](https://github.com/thisisnotahuman/FullySNNforLeggedRobots/tree/a57791b4494452b8836c2a90df6ffe1a828d56f8) | RMA、AMP 四足代码与 AMP 专家动作 | [RMA 适配](../../graduation/methods/fully_snn/rma.py) 按文件加载作者的 ActorCritic。[AMP 入口](../../scripts/bench_fully_snn_amp.py) 从 AMP 子树导入 rsl_rl 并读取 datasets/mocap_motions。项目内 [AMPLoader](../../graduation/methods/fully_snn/motion_loader.py) 是路径和关节顺序适配副本。 |

## 原版对照，训练使用本机迁移（2 个）

| 仓库与版本 | 关注内容 | 当前消费者或参考用途 |
|---|---|---|
| [HIMLoco](HIMLoco/rsl_rl/rsl_rl/modules/him_estimator.py) · [ef289ac](https://github.com/InternRobotics/HIMLoco/tree/ef289acaa62795009363b7b819c9186690630441) | HIMEstimator、特权观测、训练配置 | 盲走入口实际使用 [MyHIMLoco](../MyHIMLoco) 的 himloco_lab 和配套 rsl_rl。对齐历史帧、速度标签和原型分配时查此仓。 |
| [WMP](WMP/rsl_rl/modules/actor_critic_wmp.py) · [c232c11](https://github.com/bytedance/WMP/tree/c232c115ada4517453ebded5019078ba055456de) | ActorCriticWMP、RSSM、动作先验 | 世界模型训练和 [冻结教师](../../graduation/methods/wmp/teacher.py) 使用 [WMP 迁移](../WMP)。可在此核对作者模型和默认配置。 |

## 算法或实现对照（11 个）

| 仓库与版本 | 关注内容 | 当前消费者或参考用途 |
|---|---|---|
| [CaRe-BN](CaRe-BN/CaRe-BN/continuous_control/care_bn.py) · [6be027d](https://github.com/xuzijie32/CaRe-BN/tree/6be027dbf82e5fc064faedf499334fe0db9cf19c) | 统计量更新、重校准与融合 | [CaReBN](../../graduation/snn/normalization.py) 是本项目 Torch 实现。调整阈值和偏置时与连续控制版逐项对照。 |
| [GRSN-SNN](GRSN-SNN/grsn/policies/rlifs/GRSN.py) · [2ca2624](https://github.com/StillWolf/GRSN-SNN/tree/2ca2624a9533145fdc9e9080b39d2e7d8d8ce896) | 循环电流、遗忘门与重置 | [GRSN](../../graduation/snn/neurons.py) 自行实现。单步状态对齐和 LIF 消融可查原单元。 |
| [pop-spiking-deep-rl](pop-spiking-deep-rl/popsan_drl/popsan_ppo/popsan.py) · [5e5b904](https://github.com/combra-lab/pop-spiking-deep-rl/tree/5e5b9047899e9f5282d14c8b8212073aa12742ce) | PopSAN 群体编码和解码 | FullySNN 实际加载其作者树中的 PopSAN；这里供比对编码、解码和代理梯度。 |
| [swav](swav/main_swav.py) · [06b1b7c](https://github.com/facebookresearch/swav/tree/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9) | 原型与 Sinkhorn 分配 | [S-HIM 辅助损失](../../graduation/methods/s_him/policy.py) 内实现分配，原仓供检查温度和归一化方向。 |
| [extreme-parkour](extreme-parkour/legged_gym) · [d2ffe27](https://github.com/chengxuxin/extreme-parkour/tree/d2ffe27ba59a3229fad22a9fc94c38010bb1f519) | 扫描教师、跑酷奖励和地形课程 | 对应项目 [跑酷地形](../../graduation/sim/parkour.py)、[奖励](../../graduation/methods/es_parkour/task.py) 和 [教师/蒸馏](../../scripts/bench_parkour.py)。可查作者的 Isaac Gym 任务。 |
| [EOLO](EOLO/models/backbone/spk_resnet.py) · [9911a76](https://github.com/SageCao1125/EOLO/tree/9911a761c223fe45d3d95de13728c7c02e8c0485) | 随机流向单帧事件、脉冲视觉骨干参考 | 对应自写的 [事件合成](../../graduation/sensors/events.py) 和 [ESStudent](../../graduation/methods/es_parkour/student.py)。EOLO 的任务是事件检测；本项目 ES-Parkour 来自论文重建。 |
| [AMP_for_hardware](AMP_for_hardware/rsl_rl/rsl_rl/datasets/motion_loader.py) · [bfb0dbd](https://github.com/escontra/AMP_for_hardware/tree/bfb0dbdcf32bdf83a916790bddf193fffc7e79b8) | 独立 AMP 原版，动捕文件与加载器对照 | FullySNN AMP 运行时从 FullySNN 仓库的嵌套 AMP 副本读动作；WMP 使用本机迁移的 datasets。 |
| [dreamerv3](dreamerv3/dreamerv3) · [e01491f](https://github.com/danijar/dreamerv3/tree/e01491fad6434b2245a3b8ca201dd7faedcc458c) | RSSM、潜变量和世界模型训练 | [FrozenWMP](../../graduation/methods/wmp/teacher.py) 的世界模型来自 [WMP 自带 dreamer](../WMP/dreamer)。独立 DreamerV3 供原理和实现对照。 |
| [legged_gym](legged_gym/legged_gym) · [8fa29ac](https://github.com/leggedrobotics/legged_gym/tree/8fa29acc6fd1910c3d9659eef6310bdd301cde0a) | 旧版四足环境、地形和奖励 | 需要追溯 Gym 到 Lab 的语义时查此仓。当前执行 PhysX 环境来自两个本机迁移。 |
| [rsl_rl](rsl_rl/rsl_rl) · [4c9900f](https://github.com/leggedrobotics/rsl_rl/tree/4c9900fe242805266b30a91cd75302a6ad376833) | 通用 PPO 实现 | 项目 [简化 PPO](../../graduation/rl/ppo.py) 以及各基线随仓的 rsl_rl 各有自己的调用方。这个独立克隆不能直接替换作者 fork。 |
| [spikingjelly](spikingjelly/spikingjelly) · [ea10208](https://github.com/fangwei123456/spikingjelly/tree/ea10208738a48d905f78589040593d886bacb067) | 神经元、代理梯度与脉冲残差实现参考 | 自研模型在 [共享神经元](../../graduation/snn/neurons.py) 和 [ES-Parkour 学生](../../graduation/methods/es_parkour/student.py) 使用 Torch。可用于核对深层网络变体。 |

## 工具与资产储备（4 个）

| 仓库与版本 | 关注内容 | 当前消费者或参考用途 |
|---|---|---|
| [isaac-sim-event-camera-plugin](isaac-sim-event-camera-plugin/extension) · [d6fe479](https://github.com/spikelab-jhu/isaac-sim-event-camera-plugin/tree/d6fe47923fad09c4bfa2fd2cb13f82520114cfae) | Isaac Sim 事件相机扩展 | 候选高保真事件对照。项目当前跑酷入口采用深度 raycast 和自写合成，未发现启用该扩展的实验记录。 |
| [rpg_esim](rpg_esim/event_camera_simulator) · [4cf0b89](https://github.com/uzh-rpg/rpg_esim/tree/4cf0b8952e9f58f674c3098f1b027a4b6db53427) | 事件相机模拟器 | 候选离线对照，用同一相机轨迹比较极性和事件密度。 |
| [v2e](v2e/v2ecore) · [48f8d0e](https://github.com/SensorsINI/v2e/tree/48f8d0e6fa65657b56f087b1e84b6effd418f55c) | 从视频生成事件 | 候选视频对照与噪声建模，可与同轨迹渲染序列配合。 |
| [unitree_ros](unitree_ros/robots/a1_description/urdf/a1.urdf) · [157ff95](https://github.com/unitreerobotics/unitree_ros/tree/157ff950351e5f535e03c5d85a16b7d594b3dbcc) | Unitree A1 URDF 与模型资产 | 可核对关节命名、几何和惯量。当前实验通过迁移工程配置解析各自的机器人资产。 |

## 维护方式

新增或升级克隆时，一并更新本表与 sources.json。将储备仓改为运行依赖时，补上首次加载的入口、实际使用的子目录或资产，以及验证该入口的命令或结果。复制后修改的源码要保留上游位置和本地改动说明。实验批次按 SHA 固定，避免多个同名 rsl_rl 在同一进程中串用。

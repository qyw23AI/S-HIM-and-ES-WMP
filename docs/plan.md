# 毕业设计前四阶段执行计划

以开题报告「研究工作安排及进度」的周次表为交付顺序。按本次请求对各链路执行小批量冒烟验证。

|阶段|交付物|冒烟验收|
|---|---|---|
|1 底座|A1 Isaac Lab 盲走与跑酷场景、合成事件、45 nm 脉冲评测|共享任务入口可低批量 reset/step；验证关节映射、事件极性、发放率及逐层 MAC/AC|
|2 基线|HIMLoco、PPO-MLP、FullySNN A1 RMA|在线 rollout、权重更新、保存和短回放；FullySNN 检查特权编码与历史适应|
|3 S-HIM|5 帧 GRSN、LIF/群体读出、CaRe-BN、ANN 辅助模块|分别检查估计器与 PPO 更新、归一化融合、盲走短评测和扰动入口|
|4 视觉基线|WMP A1 训练及冻结、事件闭环、ES-Parkour|WMP PPO/RSSM 更新及加载；ES-Parkour 教师、动作/航向热身和学生闭环蒸馏|

本次复用 Conda `himloco`、`wmp_lab` 和已安装的 Isaac Lab 2.3.2/Isaac Sim 5.1。分别使用两环境兼容的项目入口。

## 扩展范围

按后续请求，追加第 5 阶段 Event-Spiking-WMP 网络、动作和隐状态蒸馏，以及第 6 阶段评测、消融和帕累托汇总代码。对完整论文所需的正式结果保留独立配置及结果记录格式。

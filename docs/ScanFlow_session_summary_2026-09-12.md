# ScanFlow 会话研究总结

日期：2026-09-12  
范围：本会话对 ScanFlow 当前项目的代码、设计、隔离实验和相关研究的整理。分析没有读取 `baseline/`；该目录仅作为用户指定的参考项目保留。

## 1. 当前项目要解决的问题

ScanFlow 的目标是使用最近一段 2D LiDAR 和机器人里程计，在去除机器人自运动后估计局部动态运动，并将结果交给动态感知 NMPC：

```text
历史 2D LiDAR + 里程计
        ↓
SE(2) 自运动补偿
        ↓
当前坐标系下的几何/时序表示
        ↓
动态区域识别 + 障碍物运动估计
        ↓
当前 LiDAR 占据 + 动态预测
        ↓
动态感知 NMPC
        ↓
控制量 (v, ω)
```

项目采用职责分离：当前 LiDAR 几何负责说明“哪里有占据”，学习或几何运动模块负责说明“占据如何移动”，NMPC 负责动力学约束、碰撞约束和控制决策。网络输出曾定义为每个角度 patch 的 `[x, y, vx, vy, dynamic_confidence]`，并额外输出 `current_valid`。

这个问题仍有研究价值，尤其适合研究稀疏 2D LiDAR、遮挡、自运动和动态避障之间的接口。但“Transformer + 历史 LiDAR + MPC”的组合本身不足以构成创新；需要证明一个具体表示或不确定性处理方法在独立场景和闭环规划上带来可重复收益。

## 2. 已确认的关键问题

### 2.1 patch 的几何位置和运动标签不是同一个物理对象

当前模型中的 `PatchAnchorExtractor` 用一个 patch 内所有有效回波的平均位置作为 anchor；训练标签则将同一 patch 内动态 beam 的速度求平均，confidence 只要包含一个动态 beam 就为 1。

在近处运动物体和远处静态墙面同时落入同一 patch 时，会出现：

- anchor 落在两个表面之间的空地；
- velocity 属于动态物体，但位置不属于它；
- planner 将该位置按动态速度外推；
- 两个相向运动物体还可能被平均成接近零速度。

在 512 个样本的表示审计中，36 patch、180 beams 的结果如下：

| 指标 | 固定 patch 表示 | patch 内几何分段 |
|---|---:|---:|
| 每帧平均分量数 | 30.67 | 93.36 |
| 动态分量中静动混合比例 | 64.04% | 2.07% |
| 动态分量中多个动态物体比例 | 2.63% | 0.20% |
| anchor 到动态点中心平均偏差 | 1.209 m | 0.003 m |
| 当前点重建平均误差 | 0.654 m | 0.051 m |

这证明旧 patch 输出存在米级位置—运动错配。几何分段能减轻问题，但会增加分量数、预算压力和遮挡边界错误，不能直接视为完整方案。

### 2.2 自运动补偿不等于动态运动完全可观测

SE(2) 对齐只消除坐标系变化，仍然会受到遮挡、显露、扫描采样变化、量测噪声和重投影空洞影响。2D LiDAR 只能看到表面；对于没有端点的长直表面，沿表面方向的运动可能没有足够观测。

因此至少要区分：

1. 有足够几何证据，可以估计速度；
2. 只约束了某一个方向，速度存在歧义；
3. 历史中没有可靠对应，不能把它当作静态。

单一 `dynamic_confidence` 同时承担动态存在概率和速度可信度是不够的。后续接口应保留 `match/support/confidence` 等证据状态，并明确其是否经过概率校准。

### 2.3 当前 Transformer 没有显式建立“对应—位移—速度”链路

现有网络的局部 cross-frame attention 输出经过残差、FFN 和时序 Transformer 后直接回归速度；attention 权重没有被转成匹配位置，输入也没有显式的 XY 位移候选或每个历史帧的真实 `dt`。模型必须从特征差中自行推导米制位移和时间关系。

这可以在固定数据分布上拟合，但容易学习场景或扫描模式先验，难以泛化到新的运动、遮挡和时间间隔。更稳妥的结构是先构造带真实单位的几何匹配代价或位移候选，再让学习模块筛选、重排和修正。

### 2.4 planner 中的 confidence 缩放速度不是一般的不确定性处理

当前 NMPC 使用近似形式：

```text
obstacle_position(τ) = position + confidence × velocity × τ
```

它可解释为特定假设下的位置期望，但“以一定概率静止、以一定概率运动”通常对应多个可能位置或一片未来占据区域，而不是一条按概率缩小的轨迹。未来应考虑运动假设、占据走廊、速度协方差或随时间扩张的安全边界。

## 3. NMPC 数值稳定性诊断与修复

原始隔离测试中，重复规划会在目标进入预测 horizon 后出现求解失败。根因是目标点处 `atan2(0, 0)` 的导数奇异，而不是 GPU、CasADi 或 obstacle 数量本身。

已在 [planner.py](../planner.py) 中完成以下修复：

- 用平滑的目标方向 cross-product 误差替代直接 `atan2` 航向误差，并以目标距离平方归一化；
- 对所有安全 slack 加小的二次正则，避免 padding slack 形成无代价的秩亏自由度；
- 每次机器人中心重规划前重置 IPOPT 初始对偶变量；
- 保留当前障碍占据，不再让 dynamic confidence 直接过滤障碍物。

最终 GT 运动场隔离结果：

| 模式 | 碰撞次数 | 最小净距 | 目标距离 | 求解失败 |
|---|---:|---:|---:|---:|
| 静态运动假设 | 7 | -0.360 m | 0.214 m | 0 |
| GT 障碍物运动 | 0 | +0.122 m | 0.221 m | 0 |

planner smoke test 和闭环隔离测试均达到 `Solve_Succeeded`，求解失败为零。这个结果只说明 planner 在受控场景中数值稳定，并说明真实运动信息有规划价值；它不证明当前学习网络能提供可靠运动。

## 4. 原网络训练与 A2/A4 结果

### 4.1 小数据过拟合

使用 32 个干净合成样本、6 帧、180 beams、36 patches、无 LiDAR 噪声，在 GPU 上训练 1,200 epochs：

- dynamic precision：0.959；recall：0.952；F1：0.955；
- velocity EPE：0.115 m/s；零速度基线：0.721 m/s；改善 84.1%；
- static speed：0.004 m/s。

结论是网络可以记住这类样本，但不能据此证明它学习了可泛化的运动对应。

### 4.2 512 样本泛化和 A2

使用另一个 512 样本数据集，噪声标准差 0.01 m，动态/有效 beam 比例约 7.62%，GPU 训练 150 epochs；最佳 checkpoint 为 epoch 49。

整体 motion learning：

- F1：0.519；
- velocity EPE：0.697 m/s；
- 零速度基线：0.702 m/s；改善只有 0.6%；
- static speed：0.012 m/s。

A2（静态世界、机器人直行或转弯）通过：

- slow straight：static speed 0.006 m/s，false dynamic 0.083，alignment 2.05×；
- fast straight：0.006，0.056，9.60×；
- turning：0.007，0.083，27.98×；
- fast-turning：0.006，0.083，36.43×。

A2 主要验证“不把机器人自运动误认为外部运动”，零速度模型也可能通过，因此不足以证明动态估计有效。

### 4.3 A4

A4 为移动机器人加固定世界速度的移动圆形障碍物。原网络结果：aggregate precision 0.333、recall 0.200、F1 0.250，平均 EPE 0.643 m/s，零速度基线 0.650 m/s，改善 1.1%。turning 和 fast-turning 场景动态召回为零。

这个结果把瓶颈从 NMPC 明确移到了运动表示、跨帧匹配和泛化，而不是继续增加时序层数即可解决的问题。

## 5. 几何基线实验

### 5.1 表面质心关联

`geometric_motion.py` 提供不训练的基线：SE(2) 对齐、距离相关的相邻点分段、表面质心一对一关联、按真实时间间隔拟合速度，并在缺少证据时标记 unsupported。

在新生成的 128 个场景上，seed 20260920 的结果：

- dynamic EPE：0.156 m/s；
- 零速度基线：0.690 m/s；改善 77.4%；
- dynamic recall：0.897；F1：0.303；
- static false dynamic rate：47.2%。

seed 20260921 的复核：

- dynamic EPE：0.172 m/s；
- 零速度基线：0.733 m/s；改善 76.5%；
- dynamic recall：0.927；F1：0.315；
- static false dynamic rate：45.2%。

这说明输入几何通常包含可恢复的运动信息，但表面质心漂移使静态误报不可接受。

### 5.2 局部点到表面匹配

`surface_motion.py` 保留当前每个 beam 的 XY 点，不再给一个混合 patch 只分配一个速度。对每个当前点，按历史时间间隔回推恒速候选，并与历史有限线段计算距离；允许不匹配，用残差和静态假设改善程度判断是否支持输出。

在相同 180 beams、相同 seed 的配对复核中：

| seed / 方法 | dynamic EPE | dynamic F1 | static false dynamic rate | static mean speed |
|---|---:|---:|---:|---:|
| 20260920 / 质心 | 0.156 | 0.303 | 47.2% | 0.237 |
| 20260920 / 点到表面 | 0.156 | 0.668 | 8.1% | 0.073 |
| 20260921 / 质心 | 0.172 | 0.315 | 45.2% | 0.226 |
| 20260921 / 点到表面 | 0.153 | 0.685 | 7.8% | 0.072 |

受控 A2/A4 中，点到表面方法的 A2 静态速度约 0.006–0.017 m/s、误报率约 1.1%–2.8%；五种 A4 自运动条件动态 recall 均为 1.0，EPE 约 0.050–0.109 m/s。这个结果支持几何匹配作为新模型的输入基础，但不是完整闭环安全验证。

计算成本是明显代价：CPU 上平均约 117.8 ms、P95 212.7 ms/序列；质心基线约 3.5/7.5 ms。当前还不能宣称满足 10 Hz 实时要求。

## 6. 候选运动评分网络实验

`motion_cost_model.py` 是一个小型实验模型，不替换生产 `model.py`。它枚举 113 个速度候选，将真实单位的历史表面匹配代价、局部代价、静态改善和径向/切向信息输入 8→48→48→1 MLP。网络学习选择候选，而不是直接从压缩 patch 回归自由空间速度。

数据按场景独立划分：

| 集合 | 数量 | seed |
|---|---:|---:|
| train | 128 | 20260930 |
| validation | 32 | 20261001 |
| test | 128 | 20261002 |

使用 GTX 1650、AdamW、lr=0.003、weight decay=1e-4、batch=16。模型只按 validation loss 选 checkpoint，test 不参与选择。

16 个训练样本过拟合 2,000 epochs：

- F1：0.972；
- dynamic EPE：0.103 m/s；
- static false dynamic rate：0.41%。

独立 test 的对比如下：

| 方法 | dynamic F1 | precision | recall | dynamic EPE | static false dynamic rate |
|---|---:|---:|---:|---:|---:|
| 点到表面基线 | 0.662 | 0.536 | 0.866 | 0.158 | 8.16% |
| 评分网络未训练 | 0.314 | 0.189 | 0.940 | 0.192 | 44.02% |
| 评分网络训练后 | 0.761 | 0.668 | 0.883 | 0.200 | 4.77% |

训练评分网络明显减少静态误报、提高 F1，但速度 EPE 变差，A4 recall 也下降。因此当前最合理的方向是“学习筛选可靠候选 + 几何连续细化”，而不是让 MLP 完全覆盖几何速度。

## 7. 现有研究可直接借鉴的设计

### 7.1 2D LiDAR 直接相关

- **LiDAR-FlowNet，2019**：使用 GRU 从历史 2D LiDAR 地图估计运动流并预测下一时刻地图，采用自监督训练。它直接对应传感器类型，但目标偏向地图预测，不等于当前项目的逐点外部障碍速度。
  - [arXiv: 2D LiDAR Map Prediction via Estimating Motion Flow with GRU](https://arxiv.org/abs/1902.06919)
- **DR-SPAAM，2020**：2D range data 的空间注意力和时序模板更新，主要用于行人检测。可以参考跨帧特征融合和轻量时序设计，但不能直接提供速度场。
  - [arXiv: DR-SPAAM](https://arxiv.org/abs/2004.14079)
- **Wang, Posner, Newman，IJRR 2015**：使用非参数表面采样点、Bayes filter 和联合状态做 2D LiDAR 动态目标检测与跟踪。它提醒我们，detection-free 不意味着必须把所有回波压缩成固定角度 patch。
  - [Model-free detection and tracking of dynamic objects with 2D lidar](https://journals.sagepub.com/doi/10.1177/0278364914562237)

### 7.2 3D scene flow 中最值得迁移的结构

- **FLOT，ECCV 2020**：学习点特征，用 optimal transport 得到软对应，再估计 scene flow；FLOT₀ 在没有 Sinkhorn 迭代时近似 attention。最适合用作“成熟对应模块”参考。
  - [论文](https://arxiv.org/abs/2007.11142) · [官方代码](https://github.com/valeoai/FLOT)
- **SCOOP，CVPR 2023**：训练阶段专注于 correspondence，推理阶段直接优化 flow refinement；强调小数据、自监督、软匹配和置信度。这与当前“对应关系学习 + 几何细化”的需求最接近。
  - [CVPR 论文](https://openaccess.thecvf.com/content/CVPR2023/html/Lang_SCOOP_Self-Supervised_Correspondence_and_Optimization-Based_Scene_Flow_CVPR_2023_paper.html) · [官方代码](https://github.com/itailang/SCOOP)
- **PointPWC-Net，ECCV 2020**：点云 cost volume、粗到细和 flow refinement。适合参考连续细化和多尺度代价体，但官方实现需要编译点云算子，移植成本较高。
  - [官方代码](https://github.com/DylanWusee/PointPWC)
- **ICP-Flow，CVPR 2024**：利用局部刚体运动和 ICP 生成伪标签，再训练实时网络。可参考“几何优化作为教师/伪标签”的路线，尤其适用于当前合成数据标签之外的真实数据扩展。
  - [论文](https://arxiv.org/abs/2402.17351)
- **PWC-Net，CVPR 2018**：金字塔、warping、cost volume 和 refinement 是光流中的成熟范式。当前 2D LiDAR 可借用“粗搜索 → warp → 局部细化”的思想，但不能假定 LiDAR 回波具有图像亮度恒常性。
  - [官方代码与论文说明](https://github.com/NVlabs/PWC-Net)

FLOT、SCOOP、PointPWC 和 ICP-Flow 主要在 3D 点云上验证；直接加一个零 z 坐标并不构成 2D 适配。必须重新处理稀疏回波、无效 beam、遮挡、SE(2) 对齐和实际时间间隔。

## 8. 推荐的模型路线

为了减少从零设计，建议采用以下最小路线：

```text
原始 LiDAR → XY 点与有效性
          → SE(2) 对齐
          → 局部几何候选 / cost volume
          → 参考 FLOT/SCOOP 的对应评分
          → 几何连续速度细化
          → motion support + dynamic presence + uncertainty
          → 未来占据走廊
          → NMPC
```

具体约束：

1. 输出位置、速度和支撑证据必须绑定同一当前点或表面分量；
2. 显式保留真实 XY、历史 lag 和 `dt`，速度单位直接为 m/s；
3. 允许无匹配，不把无证据等同于静态；
4. 将动态存在概率与运动估计可信度拆开；
5. 优先学习匹配代价或候选排序，保留可靠的几何速度；
6. planner 使用多假设或占据走廊表达不确定性，而不是简单缩放速度；
7. 只有在轻量 cost scorer、FLOT 风格对应和几何 refinement 的消融结果明确后，才决定是否需要大型 temporal Transformer。

这条路线仍然可以形成研究贡献，但贡献应表述为“针对稀疏 2D LiDAR、自运动和遮挡的运动表示/不确定性接口及其对闭环规划的影响”，而不是把通用 Transformer 或已有 scene-flow 模块本身作为创新。

## 9. 后续验证协议

### 9.1 输入与时间干预

对同一当前扫描分别：

- 改变历史运动方向；
- 移除历史帧；
- 改变历史时间间隔；
- 加入里程计偏差；
- 改变扫描噪声和动态物体可见比例。

输出应随真实运动证据变化，且在无历史证据时给出 unsupported 或高不确定性。

### 9.2 表示与运动场指标

- beam/point/surface 级 dynamic precision、recall、F1；
- dynamic velocity EPE、速度方向误差和零速度基线改善；
- static speed 与 static false dynamic rate；
- support coverage、未匹配率和遮挡条件下的误差；
- calibration：confidence 与实际正确率的可靠性曲线或 Brier/ECE；
- 区分当前可见点重建误差和未来完整占据预测误差。

### 9.3 闭环指标

在相同 NMPC 参数和计算预算下比较：

- 碰撞率、最小净距、到达率和到达时间；
- 求解失败率、控制抖动和平均规划耗时；
- 静态障碍、单个横穿物体、多个相向物体、遮挡显露和高速转弯；
- 静态运动假设、GT 运动、几何基线、学习方法和 oracle 表示。

A2/A4 是诊断场景，不足以单独支撑论文结论。测试必须使用独立生成的场景，不能把训练文件前 N 个样本当作泛化测试。

## 10. 当前代码与实验产物

本会话新增或修改的主要文件：

- [planner.py](../planner.py)：NMPC 数值稳定性修复；
- [train.py](../train.py)：confidence/static/focal loss 默认权重修正；
- [geometric_motion.py](../geometric_motion.py)：表面质心几何基线；
- [surface_motion.py](../surface_motion.py)：局部点到表面匹配基线；
- [motion_cost_model.py](../motion_cost_model.py)：候选运动评分网络；
- [test/audit_motion_representation.py](../test/audit_motion_representation.py)：patch 表示审计；
- [test/evaluate_geometric_motion.py](../test/evaluate_geometric_motion.py)：几何方法独立评估；
- [test/experiment_candidate_scorer.py](../test/experiment_candidate_scorer.py)：候选评分网络数据、GPU 训练与独立测试；
- `test/test_*_contract.py`：几何和候选评分的契约测试；
- `docs/*_2026-09-12.md`：分阶段实验记录。

训练和数据产物保存在被 `.gitignore` 忽略的 `artifacts/candidate_scorer/`，包括独立的 train/validation/test 数据、checkpoint 和报告。历史的部分诊断数据在 `/tmp`，可能被系统清理。

当前使用的 conda 环境为 `scanflow`。GPU 已确认：NVIDIA GeForce GTX 1650，PyTorch CUDA 可用；SciPy 已安装到该环境中，用于线性分配和空间查询。

常用复现命令：

```bash
# 生产模型和 planner 基础检查
conda run -n scanflow python model.py
conda run -n scanflow python planner.py
conda run --no-capture-output -n scanflow python test/test_gt_motion_field_nmpc.py

# 表示审计与几何基线
conda run -n scanflow python -m unittest test.test_geometric_motion_contract test.test_surface_motion_contract
conda run --no-capture-output -n scanflow python test/evaluate_geometric_motion.py --method surface --seed 20260921

# 候选评分网络
conda run --no-capture-output -n scanflow python test/experiment_candidate_scorer.py --prepare
conda run --no-capture-output -n scanflow python test/experiment_candidate_scorer.py --overfit --epochs 2000
conda run --no-capture-output -n scanflow python test/experiment_candidate_scorer.py --epochs 150
conda run -n scanflow python -m unittest test.test_motion_cost_contract
```

截至本记录，契约测试通过，NMPC 隔离测试求解失败为零；候选评分网络尚未接入生产 `DynamicLiDARNetwork` 或 NMPC。

## 11. 当前决策

继续研究是合理的，但研究对象应从“自制 patch Transformer 是否能学会速度”收紧为“如何让稀疏 2D LiDAR 的对应、运动和不确定性绑定到同一可见几何实体，并改善闭环动态避障”。

优先级应为：

1. 以 FLOT/SCOOP 的对应学习思想和当前点到表面几何代价为参考，建立可复现的成熟架构基线；
2. 训练一个只筛选/修正几何候选的轻量模块；
3. 加入多运动、遮挡、里程计扰动和不同 `dt` 的独立测试；
4. 只有在运动精度不退化、静态误报降低、计算预算可接受时，才接入 NMPC 做闭环比较；
5. 若成熟对应基线已经达到目标，则把贡献转向 2D LiDAR 特有的稀疏性、不确定性和规划接口，而不是继续增加模型复杂度。

## 12. v2 prototype 执行结果

根据 [ScanFlow v2 Motion Estimator 执行计划](ScanFlow%20v2%20Motion%20Estimator%20执行计划.md)，已完成公共几何层、统一 cost volume、candidate scorer 接口、top-k 连续几何细化、统一独立评测和 J1–J4 hard-case 诊断。详细结果见 [v2 refinement experiment](v2_refinement_experiment.md) 及 Phase A–D 文档。

在 128 个新 test scenes（seed 20261010、CUDA GTX 1650）上，surface baseline 为 F1=0.663、EPE=0.170 m/s、static FP=7.95%；scorer coarse MAP 为 F1=0.762、EPE=0.198 m/s、static FP=4.84%；scorer top-1/top-3 + refinement 分别为 F1=0.681/0.665、EPE=0.173/0.169 m/s、static FP=7.03%/7.95%。所以 Gate 2 的相对条件通过，Gate 3 的绝对目标尚未通过。A2 全部低于 3% false dynamic；J1–J4 诊断级行为通过，但还不能接入 NMPC。

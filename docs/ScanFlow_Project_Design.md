# ScanFlow 项目设计文档

> **暂定项目名：ScanFlow**  
> **核心目标：从历史 2D LiDAR 中学习 ego-motion-compensated 动态运动场，并将其直接用于 dynamic-aware NMPC，实现无需显式检测/跟踪的动态机器人导航。**

---

## 1. 项目一句话定义

ScanFlow 的核心思想是：

```text
Historical 2D LiDAR + Odometry
↓
Ego-Motion Compensation
↓
2D LiDAR Patch Tokenization
↓
Cross-frame Dynamic Modeling
↓
Detection-free Token Motion Field
↓
Dynamic-aware NMPC
↓
(v_cmd, ω_cmd)
```

神经网络只负责回答：

> **“周围哪里在动，以及怎么动？”**

NMPC 负责回答：

> **“根据这些运动状态，我接下来怎么走才安全？”**

因此本项目不是：

```text
LiDAR → End-to-End Control
```

也不是：

```text
LiDAR → Neural Trajectory Decoder
```

而是：

```text
Learning for Dynamic Perception
+
Model-based Optimization for Planning
```

---

# 2. 为什么最终选择这条路线

最初考虑过：

```text
Historical LiDAR
→ Transformer
→ trajectory
→ MPC
```

但这条路线需要：

- expert trajectory label
- dynamic-aware teacher
- 大量导航 episode
- 处理 trajectory 多模态
- 学习模块承担 perception + prediction + planning 三个任务

工程复杂度和训练数据成本都比较高。

最终改成 NeuPAN 风格的：

```text
Learning
→ Dynamic Representation

Optimization
→ Trajectory / Control
```

主要优势：

1. **训练标签更容易自动生成**
2. 不需要 expert planner trajectory
3. 网络任务更单纯
4. perception 与 planning 解耦
5. 可解释性更强
6. NMPC 可显式处理动力学和安全约束
7. 后续真实机器人部署更容易调试

---

# 3. 核心研究问题

ScanFlow 实际研究的问题是：

> 给定最近一小段历史 2D LiDAR 和机器人 odometry，能否在去除机器人自身运动后，直接学习一个 detection-free 的局部二维运动场，并利用该运动场进行动态避障？

核心变量：

```math
L_{t-K+1:t}
```

表示历史 LiDAR。

经过 ego-motion compensation 后：

```math
\tilde L_{t-K+1:t}
```

网络最终输出：

```math
\mathcal M_t
=
\{(q_j, v_j, c_j)\}_{j=1}^{P}
```

其中：

```math
q_j=(x_j,y_j)
```

为第 `j` 个 spatial token 的几何 anchor；

```math
v_j=(v_{x,j},v_{y,j})
```

为对应局部区域估计的运动速度；

```math
c_j
```

为动态置信度。

最终 Motion Field：

```text
[x, y, vx, vy, confidence]
```

shape：

```text
[B, P, 5]
```

默认：

```text
P = 36
```

---

# 4. 项目创新点定位

必须避免把以下内容本身当作主创新：

- Historical 2D LiDAR
- Transformer
- Ego-motion compensation
- Learning + MPC
- Dynamic obstacle velocity + MPC
- Future occupancy prediction

这些方向都已有相关研究。

ScanFlow 真正需要坚持的创新点是：

## 4.1 Ego-aligned Cross-frame Dynamic Tokens

利用 odometry 将历史 LiDAR 对齐到当前机器人坐标系：

```text
Raw temporal variation
=
Robot ego-motion
+
External object motion
```

通过 SE(2) compensation：

```text
Residual temporal variation
≈
External object motion
```

让网络重点学习环境真实动态。

---

## 4.2 Local Cross-frame Attention

不直接做：

```text
token_t[j] - token_t-1[j]
```

因为：

- 障碍物可能跨 patch
- occlusion 会改变 beam correspondence
- ego alignment 后仍然有 reprojection discretization
- LiDAR scan 存在噪声

因此采用：

```text
Current Patch j
↓
Attend to historical patches
[j-r, ..., j+r]
↓
Matched Historical Feature
```

第一版：

```text
r = 2
```

即每个当前 patch 只关注历史帧附近 5 个 angular patches。

---

## 4.3 Residual Motion Token

对当前 token 和 cross-attention 得到的 matched token 构造：

```math
[
e_t,
\bar e_{t-k},
e_t-\bar e_{t-k},
|e_t-\bar e_{t-k}|
]
```

再通过 MLP：

```text
512 → 256 → 128
```

得到：

```text
Residual Motion Token
```

它是整个网络最重要的动态表征之一。

---

## 4.4 Detection-free Token Motion Field

不采用传统：

```text
LiDAR
→ DBSCAN
→ Object Detection
→ Tracking
→ Kalman Filter
→ Velocity
```

而是：

```text
Historical LiDAR
↓
Token-level Dynamic Modeling
↓
Sparse Motion Field
```

即每个 angular patch 直接输出：

```text
(vx, vy, confidence)
```

不要求显式 object ID。

训练阶段允许使用 object ID 产生 GT；

**推理阶段完全不需要 object ID、检测或跟踪。**

---

# 5. 最终系统架构

```text
Historical 2D LiDAR
[B, K, H]

+
Historical Odometry
[B, K, 3]

↓

Odometry-based Scan Warping
(range → xy → SE(2) → polar reprojection)

↓

Aligned Range History
[B, K, H]

+
Validity Mask
[B, K, H]

↓

LiCS-style Patch Tokenization

↓

Spatial + Temporal Embedding

↓

Local Cross-frame Attention

↓

Residual Motion Token Encoder

↓

Temporal Transformer

↓

Patch-wise Dynamic Features

↓

Motion Field Head

↓

Token Motion Field
[x, y, vx, vy, confidence]

↓

Dynamic-aware NMPC

↓

Local Trajectory

↓

(v_cmd, ω_cmd)
```

---

# 6. 默认网络配置

第一版统一采用：

```text
K = 6
H = 720
P = 36
patch_size = 20
D = 128

Cross-Attention Heads = 4
Temporal Transformer Layers = 3
Temporal Heads = 4
FFN Dim = 256
```

历史窗口若 LiDAR 为 10 Hz：

```text
6 frames ≈ 0.5 s history
```

---

# 7. Ego-Motion Compensation

## 7.1 为什么需要 range → xy

LiDAR 原始数据：

```math
(r_i,\theta_i)
```

转 Cartesian：

```math
x_i=r_i\cos\theta_i
```

```math
y_i=r_i\sin\theta_i
```

因为 SE(2) rigid transform 在 Cartesian 中最自然：

```math
p'_i
=
R(\Delta\theta)p_i
+
t
```

即：

```math
p'_i
=
R(\Delta\theta)p_i
+
[\Delta x,\Delta y]^T
```

然后重新投影：

```math
r'_i=\sqrt{x'^2+y'^2}
```

```math
\theta'_i=\operatorname{atan2}(y',x')
```

最终重新落入当前 LiDAR angular grid。

---

## 7.2 为什么需要重新回到 range

LiCS-style tokenizer 假设：

> 连续 beam index 对应连续 angular sector。

如果 SE(2) 后直接保留原 beam index：

```text
原 beam j
≠
当前 frame angular bin j
```

会破坏 patch 的空间意义。

因此必须：

```text
Historical range
↓
Cartesian
↓
SE(2)
↓
Current-frame polar reprojection
↓
Aligned fixed-angle range scan
```

---

## 7.3 Reprojection 规则

若多个历史点落入同一当前 angular bin：

```text
取 minimum range
```

若某 bin 无点：

```text
validity mask = 0
```

而不是直接认为：

```text
free space
```

---

# 8. LiCS-style 2D LiDAR Tokenization

LiCS 的核心思想：

```text
720 beams
↓
36 patches
↓
20 beams / patch
↓
Linear Embedding
↓
Transformer Token
```

ScanFlow 做了一点扩展：

每个 patch 输入：

```text
20 range values
+
20 validity values
```

因此：

```text
input dim = 40
```

第一版：

```text
Linear(40 → 64)
GELU
Linear(64 → 128)
LayerNorm
```

输出：

```text
[B, 6, 36, 128]
```

---

# 9. Cross-frame Dynamic Modeling

当前 token：

```text
current
[B, 36, 128]
```

历史：

```text
history
[B, 5, 36, 128]
```

Local Cross Attention：

```text
Query:
current patch j

Key / Value:
historical patch
j-2 ... j+2
```

输出：

```text
matched_history
[B, 5, 36, 128]
```

然后构造 residual：

```text
current
matched
current - matched
|current - matched|
```

得到：

```text
[B, 5, 36, 512]
```

经过：

```text
512 → 256 → 128
```

得到：

```text
Residual Motion Tokens
[B, 5, 36, 128]
```

---

# 10. Temporal Transformer

不建议一开始做：

```text
6 × 36 = 216 tokens
→ global attention
```

第一版采用：

> **每个 spatial patch 独立进行 temporal attention**

即：

```text
[B, 6, 36, 128]
↓
[B, 36, 6, 128]
↓
[B×36, 6, 128]
↓
Temporal Transformer
↓
[B, 6, 36, 128]
```

这样：

- 更轻量
- temporal semantics 更明确
- 不会过早混合空间位置
- 局部空间 matching 已由 cross-attention 完成

---

# 11. Motion Field Head

Temporal pooling：

```text
[B, 6, 36, 128]
↓
[B, 36, 128]
```

然后：

```text
MLP
128 → 64 → 64
```

输出：

```text
velocity:
[B, 36, 2]

confidence:
[B, 36, 1]
```

每个 token：

```text
vx
vy
confidence
```

---

# 12. Patch Anchor

Anchor 不通过网络预测。

直接从当前 LiDAR patch 的 valid points 中计算：

```text
20 beams
↓
Cartesian points
↓
centroid / median
↓
(x, y)
```

输出：

```text
anchors
[B, 36, 2]
```

最终：

```text
motion_field
[B, 36, 5]
```

即：

```text
[x, y, vx, vy, confidence]
```

---

# 13. 为什么不预测 Future Occupancy

最开始考虑过：

```text
Dynamic Transformer
↓
Future Occupancy
↓
MPC
```

但最终没有作为第一版。

原因：

1. 网络任务明显更重
2. 需要预测整个未来空间
3. 与已有 occupancy forecasting 方法更接近
4. MPC 本身已经具有未来预测能力
5. 当前 motion state 更容易监督

所以最终分工：

```text
Network:
在哪里 + 怎么动

MPC:
未来在哪 + 我怎么走
```

---

# 14. NMPC

网络给出：

```math
(q_j,v_j,c_j)
```

NMPC 内部采用第一版 constant velocity prediction：

```math
\hat q_j(k)
=
q_j
+
v_j k\Delta t
```

机器人使用 unicycle model：

```math
x_{k+1}
=
x_k
+
v_k\cos\theta_k\Delta t
```

```math
y_{k+1}
=
y_k
+
v_k\sin\theta_k\Delta t
```

```math
\theta_{k+1}
=
\theta_k
+
\omega_k\Delta t
```

控制：

```math
u_k=[v_k,\omega_k]
```

---

# 15. NMPC Cost

第一版：

```math
J=
J_{goal}
+
J_{heading}
+
J_{control}
+
J_{smooth}
+
J_{dynamic-collision}
```

其中动态碰撞项：

```math
J_{collision}
=
\sum_{k,j}
c_j
\phi(
\|p_k-\hat q_j(k)\|
)
```

`confidence` 直接作为风险权重。

控制约束：

```text
v_min ≤ v ≤ v_max
ω_min ≤ ω ≤ ω_max
```

同时限制：

```text
Δv
Δω
```

---

# 16. NMPC 当前实现

当前 `planner.py` 使用：

```text
CasADi + IPOPT
```

默认：

```text
Horizon = 15
dt = 0.15 s
```

未来如果需要更高频实机：

```text
CasADi/IPOPT
↓
acados
```

---

# 17. 训练数据设计

## 17.1 一个训练样本

保存：

```text
lidar_history
[K, 720]

odom_history
[K, 3]

beam_velocity
[720, 2]

beam_dynamic
[720]

beam_valid
[720]

beam_object_id
[720]
```

`beam_object_id`：

```text
仅用于生成 GT / debug
```

推理时不需要。

---

# 18. 关键 GT 定义

## 18.1 beam_dynamic

当前 beam 命中的最近表面：

```text
static object
→ 0

dynamic object
→ 1
```

---

## 18.2 beam_velocity

非常重要：

GT 不是：

```math
v_{obs}-v_{robot}
```

而是：

```math
v^{GT}_{beam}
=
R(-\theta_t)
v^{world}_{object}
```

即：

> **障碍物自身 world velocity，表达在当前机器人坐标系。**

因为 ego-motion 已由几何对齐模块消除。

所以：

```text
机器人快速经过一面静态墙
```

墙仍然应该：

```math
v_{GT}=0
```

---

# 19. 为什么训练数据可以完全自动生成

借鉴 NeuPAN 的思路：

> 不一定要生成大量完整 navigation episode，而是针对学习模块真正需要的数学任务，直接程序化构造训练样本。

因此：

```text
随机静态几何
+
随机动态障碍
+
随机 robot ego-motion
↓
Ray Casting
↓
Historical LiDAR
+
Object ID
+
GT Object Velocity
↓
beam labels
```

无需：

- 人工标注
- expert trajectory
- teacher planner
- navigation demonstration

---

# 20. Synthetic Generator

当前 `generate_dataset.py` 已实现：

```text
随机 static line segments
+
随机 static circles
+
随机 moving circles
+
随机 differential-drive ego-motion
↓
Exact 2D ray casting
↓
K-frame historical LiDAR
```

动态 obstacle 第一版：

```math
p(t+\tau)=p(t)+v\tau
```

即 constant velocity。

---

# 21. 遮挡处理

Ray casting 取：

```text
nearest intersection
```

所以：

```text
Robot
→ static wall
→ dynamic obstacle
```

beam 的 GT 是：

```text
static
```

而不是墙后动态物体。

这使 synthetic 数据符合真实 LiDAR observation semantics。

---

# 22. Class Imbalance

动态 beam 在真实/随机场景中天然偏少。

因此 generator 设置：

```text
min_dynamic_beams
```

若一个 sample 动态可见 beam 太少：

```text
regenerate scene
```

避免训练集几乎全是：

```text
static
```

否则网络容易坍缩：

```text
confidence = 0
velocity = 0
```

---

# 23. Beam Label → Patch Label

网络输出是 36 patches。

训练时：

```text
720 beam labels
↓
36 × 20
```

每个 patch 的 GT confidence：

```math
c_j^*
=
\frac{
N_{dynamic}
}{
N_{valid}
}
```

是 soft label，而不是简单 0/1。

---

## 23.1 Patch velocity

只聚合 dynamic beam：

```math
v_j^*
=
\frac{
\sum_i d_i v_i
}{
\sum_i d_i
}
```

其中：

```math
d_i
```

是 dynamic indicator / weight。

---

# 24. Training Loss

最终：

```math
L=
\lambda_c L_{conf}
+
\lambda_v L_{vel}
+
\lambda_s L_{static}
+
\lambda_{sm} L_{smooth}
```

---

## 24.1 Confidence Loss

采用：

```text
Focal BCE
```

目的：

> 抵抗大量静态 patch 带来的类别不平衡。

---

## 24.2 Velocity Loss

采用：

```text
confidence-weighted Huber Loss
```

主要只在真实动态区域监督 velocity。

避免大量静态 patch 把网络推成：

```text
v = 0 everywhere
```

---

## 24.3 Static Velocity Regularization

对 GT 静态 patch：

```math
\|\hat v_j\|
```

施加弱约束。

防止网络在静态墙上随意预测运动。

---

## 24.4 Spatial Smoothness

对于相邻的高动态 confidence patches：

```text
weak velocity smoothness
```

但权重不能过强。

因为相邻 patch 可能属于两个不同动态障碍。

---

# 25. 数据可视化检查

训练前必须运行：

```text
visualize_sample.py
```

三联图：

```text
左：
Raw LiDAR History

中：
Ego-aligned History

右：
Current Scan + GT Motion
```

右图：

```text
静态点 = 空心圆
动态点 = 实心圆
GT velocity = 箭头
```

肉眼必须确认：

1. raw history 中 static scene 会因 robot ego-motion 漂移
2. ego alignment 后 static background 重合
3. dynamic obstacles 仍留下 temporal displacement
4. velocity arrow 方向合理
5. occluded dynamic object 不会被错误标注

如果这一关不过：

```text
不要开始正式训练
```

---

# 26. 当前代码结构

```text
ScanFlow/
│
├── model.py
│   ├── ScanWarpingSE2
│   ├── LiDARPatchTokenizer
│   ├── SpatioTemporalEmbedding
│   ├── LocalCrossFrameAttention
│   ├── ResidualMotionEncoder
│   ├── DynamicTemporalTransformer
│   ├── TemporalTokenPooling
│   ├── MotionFieldHead
│   └── DynamicLiDARNetwork
│
├── planner.py
│   └── DynamicNMPC
│
├── train.py
│   ├── Dataset
│   ├── Beam → Patch Label
│   ├── Motion Field Loss
│   ├── Training Loop
│   └── Checkpoint
│
├── generate_dataset.py
│   ├── Procedural Scene
│   ├── Dynamic Agents
│   ├── Robot Ego-motion
│   ├── Ray Casting
│   └── Automatic GT Labels
│
└── visualize_sample.py
    └── Raw / Aligned / GT Visualization
```

---

# 27. 推荐源码参考

## 27.1 LiCS

主要参考：

```text
2D LiDAR range scan
→ angular patches
→ embedding
→ Transformer
```

ScanFlow 使用它作为：

```text
Spatial Tokenizer Reference
```

不使用其 direct control head。

---

## 27.2 DR-SPAAM

主要参考：

```text
Temporal 2D LiDAR
+
Spatial Similarity / Attention
```

尤其值得参考：

```text
cross-frame local matching
```

但 ScanFlow 不做 person detector。

---

## 27.3 SCOPE

主要参考：

```text
Historical LiDAR
+
Robot motion
+
Dynamic environment prediction
```

重点借鉴：

- 时序数据组织
- ego-motion 处理
- training pipeline

避免直接变成：

```text
future occupancy prediction
```

---

## 27.4 NeuPAN

主要借鉴思想：

```text
Learning
↓
Environment Representation

Optimization
↓
Motion Planning
```

ScanFlow 对应：

```text
Motion Field Learning
+
NMPC
```

---

## 27.5 IR-SIM

后续非常适合替代当前简化 procedural simulator。

可用于：

- 2D LiDAR
- dynamic obstacles
- differential drive
- polygon / circle / rectangle
- social/dynamic motion
- collision evaluation

---

## 27.6 Arena-Rosnav

论文后期使用：

```text
复杂 dynamic navigation benchmark
```

而不是第一阶段开发工具。

---

## 27.7 TEB

可作为重要 baseline：

```text
TEB static

TEB + tracked dynamic obstacles

ScanFlow Motion Field + NMPC
```

---

## 27.8 CasADi / acados

当前：

```text
CasADi + IPOPT
```

后续实时性不足时：

```text
acados
```

---

# 28. 论文创新 Claim 建议

不要写：

> We are the first to use historical LiDAR.

不要写：

> We are the first to combine learning and MPC.

不要写：

> We are the first to estimate obstacle velocity from LiDAR.

更安全的表达：

> **We propose an ego-motion-compensated, detection-free token motion representation that estimates local scene dynamics directly from historical 2D LiDAR scans and integrates the learned motion field with model predictive control for dynamic robot navigation.**

核心关键词：

```text
Historical 2D LiDAR

Ego-Motion Compensation

Cross-frame Dynamic Tokens

Residual Motion Representation

Detection-free Motion Field

Dynamic-aware NMPC
```

---

# 29. 核心 Ablation

论文必须至少做：

## 29.1 Ego Alignment

```text
Raw History
vs
Ego-aligned History
```

证明 robot ego-motion compensation 有效。

---

## 29.2 Residual Motion Tokens

```text
Vanilla Temporal Transformer
vs
Residual Motion Token
```

---

## 29.3 Local Cross Attention

```text
Same-index subtraction
vs
Local Cross Attention
```

---

## 29.4 Temporal History Length

```text
K = 1
K = 3
K = 6
K = 10
```

---

## 29.5 Robot Ego-speed Stress Test

保持 dynamic obstacle motion 不变。

改变 robot：

```text
slow
medium
fast
```

理想结果：

> motion estimation 对 robot ego-speed 更不敏感。

---

# 30. Motion Field Evaluation Metrics

不能只评 navigation success。

需要单独证明 representation 本身有效。

建议：

```text
Dynamic Classification
- Precision
- Recall
- F1

Velocity
- EPE / velocity L2 error
- angular error
- speed error

Static False Motion
- predicted speed on static patches

Ego-motion Robustness
- velocity error vs robot speed
```

---

# 31. Navigation Evaluation

最终比较：

```text
Success Rate

Collision Rate

Dynamic Collision Rate

Time to Goal

Path Length

Minimum Clearance

Average Speed

Control Smoothness

Computation Time
```

---

# 32. 推荐 Baselines

## Perception / Representation

```text
No history

Raw history Transformer

Ego-aligned history Transformer

Same-index residual

Cross-attention residual

ScanFlow full
```

---

## Planning

```text
Static NMPC

TEB

Dynamic TEB / tracked obstacle baseline

GT Motion Field + NMPC

Predicted ScanFlow Motion Field + NMPC
```

其中：

```text
GT Motion Field + NMPC
```

非常重要。

它可以区分：

```text
Planner 上限
vs
Perception 误差
```

---

# 33. 数据阶段路线

## Stage 1 — Pure Synthetic

当前：

```text
generate_dataset.py
```

目的：

> 验证 motion field learning 是否成立。

---

## Stage 2 — Better Simulation

迁移到：

```text
IR-SIM
```

加入：

- acceleration
- turning
- stop-and-go
- crossing
- different shapes
- noise
- occlusion
- odometry noise

---

## Stage 3 — Navigation Benchmark

使用：

```text
Arena-Rosnav / DynaBARN / custom dynamic benchmark
```

评测完整：

```text
perception → NMPC → navigation
```

---

## Stage 4 — Real Robot

输入：

```text
LaserScan
Odometry
Goal
```

实时输出：

```text
cmd_vel
```

需要重点处理：

- timestamp synchronization
- odometry interpolation
- TF
- LiDAR noise
- scan deskew
- sensor latency
- inference latency

---

# 34. 必须特别注意的工程问题

## 34.1 Timestamp Synchronization

历史 scan 和 odom 必须时间对齐。

不要：

```text
scan_t-k
+
latest odom
```

应该：

```text
scan timestamp
↓
interpolate robot pose
```

---

## 34.2 LiDAR Deskew

第一版低速仿真可以不做。

高速实机时：

```text
LaserScan 本身也有采样时间跨度
```

可能需要根据：

```text
time_increment
```

做 deskew。

---

## 34.3 Odometry Noise

第一阶段 GT odometry。

之后必须测试：

```text
Gaussian pose noise

yaw drift

timestamp error
```

证明模型不会过度依赖完美对齐。

---

## 34.4 Constant Velocity Assumption

当前 NMPC：

```math
p(t+\tau)=p(t)+v\tau
```

只是第一版。

后续可扩展：

```text
Acceleration

Motion uncertainty

Multi-modal prediction
```

但不要第一版全部做。

---

# 35. 目前最推荐的开发顺序

```text
① visualize_sample.py
确认 geometry / GT 完全正确

↓

② 小数据过拟合
100~1000 samples
确认网络能学会

↓

③ Static-vs-Dynamic sanity test
固定简单场景

↓

④ 10k~100k synthetic training

↓

⑤ Motion Field quantitative evaluation

↓

⑥ Plug into NMPC

↓

⑦ Closed-loop navigation simulation

↓

⑧ IR-SIM

↓

⑨ Strong baselines + ablation

↓

⑩ ROS / real robot
```

---

# 36. 第一阶段成功标准

在进入复杂 simulator 之前，必须达到：

### Geometry

```text
Ego-aligned static background visually stable
```

### Dynamic Classification

```text
动态 patch F1 明显高于 naive baseline
```

### Velocity

```text
dynamic velocity EPE 明显低于 zero prediction
```

### Static Region

```text
static predicted speed 接近 0
```

### NMPC

使用 GT motion field 时：

```text
明显优于 static NMPC
```

使用 predicted field 时：

```text
接近 GT field planner 性能
```

如果这五步成立：

> 这个研究方向就真正站住了。

---

# 37. 最终项目定义

ScanFlow 可以最终定义为：

> **A detection-free dynamic navigation framework that learns an ego-motion-compensated local motion field from historical 2D LiDAR scans and integrates the learned motion representation with nonlinear model predictive control.**

中文：

> **ScanFlow 是一种面向动态机器人导航的检测无关框架，通过历史 2D LiDAR 和里程计学习经 ego-motion 补偿的局部运动场，并将该动态表示直接嵌入非线性模型预测控制进行实时导航。**

---

# 38. 最终核心 Pipeline

```text
Historical 2D LiDAR + Odometry
↓
SE(2) Ego-Motion Compensation
↓
Aligned Range History
↓
LiCS-style Patch Tokens
↓
Local Cross-frame Attention
↓
Residual Motion Tokens
↓
Temporal Transformer
↓
Detection-free Token Motion Field
[x, y, vx, vy, confidence]
↓
Dynamic-aware NMPC
↓
(v_cmd, ω_cmd)
```

---

# 39. 当前项目状态

目前已经形成第一版代码骨架：

```text
model.py
✓ 网络主体
✓ SE(2) scan warping
✓ patch tokenizer
✓ cross-frame attention
✓ residual motion encoder
✓ temporal transformer
✓ motion head

planner.py
✓ CasADi NMPC
✓ dynamic token prediction
✓ collision penalty
✓ control constraints
✓ warm start

train.py
✓ beam → patch GT
✓ Focal confidence loss
✓ Huber velocity loss
✓ static regularization
✓ smoothness
✓ training loop

generate_dataset.py
✓ procedural scene
✓ moving agents
✓ robot ego-motion
✓ ray casting
✓ automatic motion GT

visualize_sample.py
✓ raw history
✓ aligned history
✓ static/dynamic GT
✓ velocity arrows
```

下一阶段的重点已经不是继续堆模块，而是：

```text
验证 GT
↓
验证 motion field 是否真正可学
↓
验证 predicted field 是否能改善 NMPC 动态导航
```

这三个实验结果会决定 ScanFlow 是否值得继续扩展成完整论文。

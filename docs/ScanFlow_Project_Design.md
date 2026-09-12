# ScanFlow 项目设计文档

> **暂定项目名：ScanFlow**  
> **核心目标：从历史 2D LiDAR 中学习 ego-motion-compensated、detection-free 的局部运动场，并将当前 LiDAR 几何占据与学习到的动态状态共同用于 dynamic-aware NMPC。**

---

## 1. 项目一句话定义

ScanFlow 的核心思想是：

```text
Historical 2D LiDAR + Odometry
↓
SE(2) Ego-Motion Compensation
↓
LiDAR Angular Patch Tokenization
↓
Validity-aware Local Cross-frame Matching
↓
Residual Motion Tokens
↓
Per-patch Temporal Modeling
↓
Detection-free Token Motion Field
[x, y, vx, vy, dynamic_confidence]
+
Current LiDAR Occupancy / Anchor Validity
↓
Dynamic-aware NMPC
↓
(v_cmd, ω_cmd)
```

神经网络主要回答：

> **“当前可见区域中哪里存在动态运动，以及这些区域怎么动？”**

当前 LiDAR 几何回答：

> **“哪里存在需要避让的实际占据？”**

NMPC 负责回答：

> **“结合静态占据与动态运动预测，我接下来怎么走才安全？”**

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
Geometry for Obstacle Existence
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
- 大量 navigation episode
- 处理 trajectory 多模态
- 学习模块同时承担 perception + prediction + planning

工程复杂度和训练数据成本都较高。

最终采用 NeuPAN 风格的职责分离：

```text
Learning
→ Dynamic Representation

Current LiDAR Geometry
→ Obstacle Existence

Optimization
→ Trajectory / Control
```

主要优势：

1. **训练标签容易自动生成**
2. 不需要 expert planner trajectory
3. 网络任务更单纯
4. perception 与 planning 解耦
5. 静态障碍不会依赖动态分类器才能被 planner 看见
6. NMPC 可显式处理动力学、安全距离和控制约束
7. 可解释性更强
8. 后续真实机器人部署更容易调试

---

# 3. 核心研究问题

ScanFlow 实际研究的问题是：

> 给定最近一小段历史 2D LiDAR 和机器人 odometry，能否在去除机器人自身运动后，直接学习一个 detection-free 的局部二维运动场，并利用该运动场增强基于当前 LiDAR 占据的动态避障？

历史 LiDAR：

```math
L_{t-K+1:t}
```

经过 ego-motion compensation 后：

```math
\tilde L_{t-K+1:t}
```

网络输出：

```math
\mathcal M_t
=
\{(q_j, v_j, c_j)\}_{j=1}^{P}
```

其中：

```math
q_j=(x_j,y_j)
```

为第 `j` 个当前 spatial token 的几何 anchor；

```math
v_j=(v_{x,j},v_{y,j})
```

为对应区域估计的障碍物自身运动速度，表达在当前机器人坐标系；

```math
c_j\in[0,1]
```

表示：

> **当前 patch 存在动态运动的概率（dynamic presence probability）。**

注意：

```text
c_j 不是 obstacle-existence probability
c_j 也不是 dynamic beam fraction
```

障碍物是否存在由当前 LiDAR 几何 / anchor validity 决定。

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

模型同时输出：

```text
current_valid
[B, P]
```

用于表示当前 patch 是否具有有效 LiDAR 几何 anchor。

---

# 4. 项目创新点定位

必须避免把以下内容本身当作主创新：

- Historical 2D LiDAR
- Transformer
- Ego-motion compensation
- Learning + MPC
- Dynamic obstacle velocity + MPC
- Future occupancy prediction

这些方向均已有相关研究。

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

## 4.2 Validity-aware Local Cross-frame Matching

不直接做：

```text
token_t[j] - token_t-1[j]
```

因为：

- 障碍物可能跨 patch
- occlusion 会改变 beam correspondence
- ego alignment 后仍存在 reprojection discretization
- LiDAR scan 存在噪声
- 历史 scan warping 后部分 angular bin 可能没有有效点

因此采用：

```text
Current Patch j
↓
Attend only to valid historical patches
[j-r, ..., j+r]
↓
Matched Historical Feature
+
Matched Validity
```

第一版：

```text
r = 2
```

即每个当前 patch 只关注历史帧附近 5 个 angular patches。

历史无效 patch 会在 attention 中被 mask 掉；若整个局部窗口均不可匹配，则该 residual time step 标记为无效，不参与后续 temporal pooling。

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

再通过：

```text
4D → 2D → D
```

默认：

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

每个 angular patch 输出：

```text
(vx, vy, dynamic_confidence)
```

不要求显式 object ID。

训练阶段允许使用 object ID 生成 GT / debug；

**推理阶段不需要 object ID、检测或跟踪。**

---

## 4.5 Geometry / Dynamics Decoupling for Planning

当前实现明确区分：

```text
Obstacle existence
← current LiDAR geometry

Obstacle motion
← learned motion field
```

因此：

```text
static wall
→ current anchor valid
→ confidence ≈ 0
→ effective velocity ≈ 0
→ 仍然必须参与 collision avoidance
```

而不是：

```text
confidence ≈ 0
→ obstacle disappears
```

这是当前 planner 语义中非常重要的一条原则。

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
LiDAR Patch Tokenization
(range + validity)
↓
Spatial Patch Embedding
↓
Validity-aware Local Cross-frame Attention
↓
Residual Motion Token Encoder
↓
Temporal Lag Embedding
↓
Per-patch Temporal Transformer
↓
Validity-aware Temporal Pooling
↓
Motion Evidence
+
Current Spatial Token
↓
Dynamic Feature Fusion
↓
Motion Field Head
↓
Token Motion Field
[x, y, vx, vy, confidence]
+
Current Anchor Validity
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
Local Patch Radius = 2
Temporal Transformer Layers = 3
Temporal Heads = 4
FFN Dim = 256
```

历史窗口若 LiDAR 为 10 Hz：

```text
6 frames ≈ 0.5 s history
```

其中 temporal motion sequence 长度为：

```text
T_motion = K - 1 = 5
```

因为 Temporal Transformer 处理的是 5 个 historical-vs-current residual motion tokens，而不是把 raw current token 混入同一 temporal sequence。

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

SE(2) rigid transform 在 Cartesian 中最自然：

```math
p'_i=R(\Delta\theta)p_i+t
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

LiDAR angular patch tokenizer 假设：

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

而不是直接认为 free space。

---

# 8. LiDAR Patch Tokenization

LiCS 提供的核心启发是：

```text
720 beams
↓
36 angular patches
↓
20 beams / patch
↓
Linear Embedding
↓
Transformer-compatible Token
```

ScanFlow 扩展为每个 patch 输入：

```text
20 normalized range values
+
20 validity bits
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

在 cross-frame matching 前只加入：

```text
Spatial Patch Embedding
```

不提前加入 temporal embedding，避免 frame identity 干扰 spatial similarity matching。

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
valid historical patches
j-2 ... j+2
```

输出：

```text
matched_history
[B, 5, 36, 128]

matched_valid
[B, 5, 36]
```

`matched_valid` 表示对应 history lag 是否在局部 angular window 内存在可用匹配，并且当前 patch 本身有效。

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

第一版采用：

> **每个 spatial patch 独立进行 temporal motion attention。**

输入只包含 residual motion tokens：

```text
[B, K-1, P, D]
=
[B, 5, 36, 128]
```

先加入 learned Temporal Lag Embedding：

```text
Residual Motion Tokens
+
Lag Embedding
```

再变换：

```text
[B, 5, 36, 128]
↓
[B, 36, 5, 128]
↓
[B×36, 5, 128]
↓
Temporal Transformer
↓
[B, 5, 36, 128]
```

Temporal Transformer 使用 `matched_valid` 作为 padding mask，避免无效历史匹配参与 temporal attention。

随后进行 validity-aware learned pooling：

```text
[B, 5, 36, 128]
↓
[B, 36, 128]
```

这样做的原因：

- temporal sequence 中所有 token 都具有相同的 residual-motion 语义
- current spatial appearance 不再和 residual token 混在同一序列
- invalid reprojection / unmatched history 不会被当成有效动态证据
- 空间 matching 已由 local cross-attention 处理

---

# 11. Dynamic Feature Fusion 与 Motion Field Head

Temporal pooling 得到：

```text
motion_features
[B, 36, 128]
```

当前 spatial token：

```text
current
[B, 36, 128]
```

二者在 temporal modeling 后进行 fusion：

```text
[motion_features, current]
↓
Linear(256 → 128)
GELU
LayerNorm
↓
dynamic_features
[B, 36, 128]
```

再进入 Motion Field Head：

```text
128 → 64 → 64
```

输出：

```text
velocity
[B, 36, 2]

confidence
[B, 36, 1]
```

其中：

```text
confidence = P(dynamic motion exists in this current patch)
```

对无 current anchor 的 patch：

```text
velocity = 0
confidence = 0
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
centroid
↓
(x, y)
```

输出：

```text
anchors
[B, 36, 2]

current_valid
[B, 36]
```

最终：

```text
motion_field
[B, 36, 5]
=
[x, y, vx, vy, confidence]
```

Anchor 的几何有效性与 dynamic confidence 是两个不同概念：

```text
current_valid → 障碍物几何是否存在
confidence    → 该几何区域是否具有动态运动
```

---

# 13. 为什么不预测 Future Occupancy

第一版不预测完整 Future Occupancy。

原因：

1. 网络任务明显更重
2. 需要预测整个未来空间
3. 与已有 occupancy forecasting 方法更接近
4. NMPC 本身已具有有限时域预测能力
5. 当前 motion state 更容易监督和解释

所以分工为：

```text
Current LiDAR:
哪里有东西

Network:
哪些可见区域在动 + 怎么动

NMPC:
它们未来可能在哪 + 机器人怎么走
```

---

# 14. NMPC Motion Semantics

网络给出：

```math
(q_j,v_j,c_j)
```

规划器定义 effective token velocity：

```math
v_j^{eff}=c_jv_j
```

第一版 constant-velocity prediction：

```math
\hat q_j(k)
=
q_j
+
v_j^{eff}k\Delta t
```

即：

```math
\hat q_j(k)
=
q_j
+
c_jv_jk\Delta t
```

这意味着：

```text
static obstacle:
current geometry valid
confidence ≈ 0
→ future position ≈ current position
→ 始终参与避障

high-confidence dynamic obstacle:
confidence ≈ 1
→ 使用完整预测 velocity
```

**confidence 只调制运动，不决定障碍物是否存在。**

机器人采用 unicycle model：

```math
x_{k+1}=x_k+v_k\cos\theta_k\Delta t
```

```math
y_{k+1}=y_k+v_k\sin\theta_k\Delta t
```

```math
\theta_{k+1}=\theta_k+\omega_k\Delta t
```

控制：

```math
u_k=[v_k,\omega_k]
```

---

# 15. NMPC Cost 与 Safety Constraint

第一版 cost：

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
J_{soft-collision}
+
J_{safety-slack}
```

所有有效 obstacle token 都参与 collision term：

```math
J_{soft-collision}
=
\sum_{k,j}
\phi(\|p_k-\hat q_j(k)\|)
```

注意这里**不再用 `c_j` 作为 obstacle validity / collision weight**。

同时加入软化安全距离约束：

```math
\|p_k-\hat q_j(k)\|+s_{j,k}
\ge r_{safe}
```

```math
s_{j,k}\ge0
```

并对 slack 加较强惩罚：

```math
J_{safety-slack}
=
\lambda_s\sum_{j,k}s_{j,k}
```

其中：

```math
r_safe
=
r_robot+r_obstacle+safety_margin
```

控制约束：

```text
v_min ≤ v ≤ v_max
ω_min ≤ ω ≤ ω_max
```

同时限制：

```text
|Δv| / Δt ≤ a_v_max
|Δω| / Δt ≤ a_ω_max
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
max_obstacles = 24
```

Obstacle selection 原则：

```text
先保留当前几何有效、有限值、范围内 token
↓
若超过 max_obstacles
↓
优先近距离 occupancy
+
较弱 dynamic-confidence tie-break
```

不会因为：

```text
confidence ≈ 0
```

而删除静态障碍。

Warm start：

```text
shift previous U
↓
在当前 robot-centric planning frame 中重新 rollout X
```

不直接复用上一周期的 state trajectory 坐标。

求解失败时：

```text
IPOPT failure / non-finite control
↓
(v_cmd, ω_cmd) = (0, 0)
```

不会执行未经验证的 `Opti.debug` 控制量。

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
[720]   # generator/debug 可选
```

`beam_object_id`：

```text
仅用于生成 GT / debug / 后续 multi-object 分析
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

GT 不是：

```math
v_{obs}-v_{robot}
```

而是：

```math
v^{GT}_{beam}
=
R(-\theta_t)v^{world}_{object}
```

即：

> **障碍物自身 world velocity，表达在当前机器人坐标系。**

因为 ego-motion 已由几何对齐模块消除。

所以机器人快速经过一面静态墙时：

```math
v_{GT}=0
```

---

# 19. 为什么训练数据可以完全自动生成

借鉴 NeuPAN 的任务分解思路：

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

beam GT 是：

```text
static
```

而不是墙后动态物体。

这使 synthetic 数据符合真实 LiDAR observation semantics。

---

# 22. Class Imbalance

动态 beam 在真实/随机场景中天然偏少。

因此 generator 可设置：

```text
min_dynamic_beams
```

若一个 sample 动态可见 beam 太少：

```text
regenerate scene
```

避免训练集几乎全是 static。

否则网络容易坍缩：

```text
confidence = 0
velocity = 0
```

训练损失仍使用 focal confidence loss 缓解 patch-level dynamic presence 的类别不平衡。

---

# 23. Beam Label → Patch Label

网络输出是 36 patches。

训练时：

```text
720 beam labels
↓
36 × 20
```

## 23.1 Dynamic Presence Confidence

当前 confidence GT 定义为：

```math
c_j^*
=
\mathbb{1}[N_{dynamic,j}>0]
```

也就是：

> **只要当前有效 patch 内至少有一个 dynamic beam，该 patch 的 dynamic-presence target 就为 1。**

这样可以避免小目标只占少量 beam 时，完美模型仍输出低 confidence 的语义问题。

例如：

```text
20 valid beams
4 dynamic beams
```

当前定义：

```text
confidence_target = 1
```

而不是：

```text
confidence_target = 4/20 = 0.2
```

---

## 23.2 Dynamic Fraction

仍保留：

```math
f_j
=
\frac{N_{dynamic,j}}{N_{valid,j}}
```

但它只是：

```text
dynamic_fraction
```

用途：

- dataset diagnostics
- 可选 velocity reliability gating
- 分析小目标 / patch 混合程度

它**不是 motion-field confidence 的监督目标**。

---

## 23.3 Patch Velocity

只聚合 dynamic beam：

```math
v_j^*
=
\frac{\sum_i d_i v_i}{\sum_i d_i}
```

其中：

```math
d_i
```

是 dynamic indicator / weight。

如果同一个 patch 内同时包含多个不同运动物体，当前第一版仍会对其 dynamic beam velocity 做平均。这是明确已知的 representation limitation，后续可利用 `beam_object_id` 做 dominant / nearest / risk-aware motion aggregation。

---

# 24. Training Loss

当前：

```math
L=
\lambda_cL_{conf}
+
\lambda_vL_{vel}
+
\lambda_sL_{static}
+
\lambda_{sm}L_{smooth}
```

---

## 24.1 Confidence Loss

采用：

```text
Focal BCE
```

监督目标：

```text
binary dynamic presence
```

目的：

> 抵抗大量 static patch 带来的类别不平衡，同时让 confidence 与 planner 中“动态运动可信度”的语义一致。

---

## 24.2 Velocity Loss

采用：

```text
weighted Huber Loss
```

默认：

```text
velocity_weight = dynamic_presence
```

即主要在真实动态 patch 上监督 velocity。

可选设置：

```text
min_dynamic_fraction_for_velocity > 0
```

用于只对 dynamic fraction 足够高的 patch 做 velocity regression，从而过滤非常混杂、速度标签可靠性较弱的 patch。

---

## 24.3 Static Velocity Regularization

对 GT static patch：

```math
\|\hat v_j\|
```

施加弱约束，防止静态墙上出现任意预测运动。

---

## 24.4 Spatial Smoothness

对于相邻且 GT dynamic 的 patches：

```text
weak velocity smoothness
```

但权重不能过强，因为相邻 patch 可能属于两个不同动态障碍。

---

# 25. 数据可视化检查

训练前必须运行：

```text
visualize_sample.py
```

三联图：

```text
左：Raw LiDAR History
中：Ego-aligned History
右：Current Scan + GT Motion
```

肉眼必须确认：

1. raw history 中 static scene 会因 robot ego-motion 漂移
2. ego alignment 后 static background 重合
3. dynamic obstacles 仍留下 temporal displacement
4. velocity arrow 方向合理
5. occluded dynamic object 不会被错误标注
6. 小目标即使只占少量 beams，也能形成正确 patch-level dynamic presence

如果这一关不过，不要开始正式训练。

---

# 26. 当前代码结构

```text
ScanFlow/
│
├── model.py
│   ├── ScanWarpingSE2
│   ├── LiDARPatchTokenizer
│   ├── SpatialPatchEmbedding
│   ├── LocalCrossFrameAttention
│   ├── ResidualMotionEncoder
│   ├── TemporalLagEmbedding
│   ├── DynamicTemporalTransformer
│   ├── TemporalTokenPooling
│   ├── DynamicFeatureFusion
│   ├── MotionFieldHead
│   ├── PatchAnchorExtractor
│   └── DynamicLiDARNetwork
│
├── planner.py
│   └── DynamicNMPC
│       ├── current-geometry obstacle selection
│       ├── confidence-gated motion prediction
│       ├── soft collision cost
│       ├── safety slack constraints
│       ├── robot-centric warm start
│       └── safe-stop fallback
│
├── train.py
│   ├── Dataset
│   ├── Beam → Patch Targets
│   ├── Dynamic Presence Confidence
│   ├── Motion Field Loss
│   ├── Epoch-level Metrics
│   └── Checkpoint
│
├── generate_dataset.py
│   ├── Procedural Scene
│   ├── Dynamic Agents
│   ├── Robot Ego-motion
│   ├── Ray Casting
│   └── Automatic Motion GT
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
→ Transformer-compatible token
```

ScanFlow 使用它作为 spatial tokenizer reference，不使用其 direct control head。

---

## 27.2 DR-SPAAM

主要参考：

```text
Temporal 2D LiDAR
+
Spatial Similarity / Attention
```

重点借鉴：

```text
local cross-frame matching
```

ScanFlow 进一步加入 ego alignment、reprojection validity mask 和 residual motion representation。

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

避免直接变成 future occupancy prediction。

---

## 27.4 NeuPAN

主要借鉴思想：

```text
Learning
→ Environment / Dynamic Representation

Optimization
→ Motion Planning
```

同时保留一个重要原则：

> obstacle geometry 本身应直接进入 planning，而不能因为 learned dynamic confidence 较低就消失。

---

## 27.5 IR-SIM

后续适合替代当前简化 procedural simulator，用于：

- 2D LiDAR
- dynamic obstacles
- differential drive
- polygon / circle / rectangle
- social/dynamic motion
- collision evaluation

---

## 27.6 Arena-Rosnav

论文后期用于复杂 dynamic navigation benchmark，而不是第一阶段开发工具。

---

## 27.7 TEB

重要 baseline：

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

> **We propose an ego-motion-compensated, detection-free token motion representation that estimates local scene dynamics directly from historical 2D LiDAR scans and integrates the learned motion field with geometry-preserving nonlinear model predictive control for dynamic robot navigation.**

核心关键词：

```text
Historical 2D LiDAR
Ego-Motion Compensation
Validity-aware Cross-frame Matching
Residual Motion Representation
Detection-free Motion Field
Geometry / Dynamics Decoupling
Dynamic-aware NMPC
```

---

# 29. 核心 Ablation

论文至少做：

## 29.1 Ego Alignment

```text
Raw History
vs
Ego-aligned History
```

## 29.2 Residual Motion Tokens

```text
Vanilla Temporal Modeling
vs
Residual Motion Token
```

## 29.3 Local Cross Attention

```text
Same-index subtraction
vs
Local Cross Attention
```

## 29.4 Validity-aware Matching

```text
No validity mask
vs
Validity-aware local matching
```

## 29.5 Temporal History Length

```text
K = 2 / 3 / 6 / 10
```

注意当前 motion model 至少需要：

```text
K >= 2
```

## 29.6 Robot Ego-speed Stress Test

保持 dynamic obstacle motion 不变，改变 robot：

```text
slow
medium
fast
```

理想结果：

> motion estimation 对 robot ego-speed 更不敏感。

## 29.7 Planner Semantics

建议加入：

```text
Dynamic-confidence-as-obstacle-filter
vs
Geometry-preserving planner
```

证明静态障碍不会因 dynamic confidence 低而从规划器中消失。

---

# 30. Motion Field Evaluation Metrics

不能只评 navigation success。

建议：

```text
Dynamic Presence Classification
- Precision
- Recall
- F1

Velocity
- EPE / velocity L2 error
- angular error
- speed error

Static False Motion
- predicted speed on static patches

Confidence Calibration
- optional reliability / calibration curve

Patch Mixing
- error vs dynamic_fraction

Ego-motion Robustness
- velocity error vs robot speed
```

训练/验证统计应优先在整个 epoch 累计 TP / FP / FN 后计算 F1，而不是简单平均 batch F1。

---

# 31. Navigation Evaluation

最终比较：

```text
Success Rate
Collision Rate
Dynamic Collision Rate
Static Collision Rate
Time to Goal
Path Length
Minimum Clearance
Average Speed
Control Smoothness
Computation Time
Solver Failure Rate
Safety Slack Usage
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

非常重要，它可以区分：

```text
Planner 上限
vs
Perception 误差
```

另外建议保留：

```text
GT geometry + GT motion
GT geometry + predicted motion
```

进一步隔离 motion estimation 对 planning 的影响。

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
- solver timeout / safe fallback

---

# 34. 必须特别注意的工程问题

## 34.1 Timestamp Synchronization

历史 scan 和 odom 必须时间对齐。

不要：

```text
scan_t-k + latest odom
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

高速实机时 LaserScan 本身有采样时间跨度，可能需要根据：

```text
time_increment
```

做 deskew。

---

## 34.3 Odometry Noise

第一阶段使用 GT odometry。

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
q(t+\tau)=q(t)+c\,v\tau
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

## 34.5 Planner Frame Consistency

当前推荐 robot-centric planning：

```text
robot_state = [0, 0, 0]
anchors / velocity / goal
均表达在同一当前 planning frame
```

每个周期 frame 都会变化，所以 warm start 只 shift 控制序列，并从当前 state 重新 rollout trajectory。

---

## 34.6 Solver Failure

求解器失败不是“仍然输出某个 debug iterate”的理由。

当前第一版策略：

```text
solver failure
→ safe stop
```

后续可升级为：

```text
validated braking controller
or
certified previous-safe command
```

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
↓
④ Dynamic Presence sanity test
特别检查小目标 / 少 beam 目标
↓
⑤ 10k~100k synthetic training
↓
⑥ Motion Field quantitative evaluation
↓
⑦ GT Motion Field + NMPC
先验证 planner 上限和静态障碍安全性
↓
⑧ Predicted Motion Field + NMPC
↓
⑨ Closed-loop navigation simulation
↓
⑩ IR-SIM + strong baselines + ablation
↓
⑪ ROS / real robot
```

---

# 36. 第一阶段成功标准

在进入复杂 simulator 之前，必须达到：

### Geometry

```text
Ego-aligned static background visually stable
```

### Dynamic Presence

```text
动态 patch F1 明显高于 naive baseline
小型动态目标不会因占 beam 少而被系统性压低 confidence
```

### Velocity

```text
dynamic velocity EPE 明显低于 zero prediction
```

### Static Region

```text
static predicted speed 接近 0
```

### Planner Static Safety

```text
confidence ≈ 0 的静态障碍仍然能够被避让
```

### NMPC Dynamic Benefit

使用 GT motion field 时：

```text
明显优于 static-motion assumption baseline
```

使用 predicted field 时：

```text
接近 GT field planner 性能
```

如果这些条件成立：

> 这个研究方向才真正站住。

---

# 37. 最终项目定义

ScanFlow 可以最终定义为：

> **A detection-free dynamic navigation framework that learns an ego-motion-compensated local motion field from historical 2D LiDAR scans and integrates the learned dynamics with geometry-preserving nonlinear model predictive control.**

中文：

> **ScanFlow 是一种面向动态机器人导航的检测无关框架，通过历史 2D LiDAR 与里程计学习经 ego-motion 补偿的局部运动场，同时保留当前 LiDAR 几何占据，并将二者共同嵌入非线性模型预测控制进行动态导航。**

---

# 38. 最终核心 Pipeline

```text
Historical 2D LiDAR + Odometry
↓
SE(2) Ego-Motion Compensation
↓
Aligned Range + Validity History
↓
LiDAR Angular Patch Tokens
↓
Spatial Patch Embedding
↓
Validity-aware Local Cross-frame Attention
↓
Residual Motion Tokens
↓
Temporal Lag Embedding
↓
Per-patch Temporal Transformer
↓
Validity-aware Temporal Pooling
↓
Fuse with Current Spatial Token
↓
Detection-free Token Motion Field
[x, y, vx, vy, dynamic_confidence]
+
Current LiDAR Geometry / Anchor Validity
↓
confidence-gated motion prediction
+
geometry-preserving collision avoidance
↓
Dynamic-aware NMPC
↓
(v_cmd, ω_cmd)
```

---

# 39. 当前项目状态

目前第一版代码已经形成语义闭环：

```text
model.py
✓ SE(2) scan warping
✓ range reprojection + validity
✓ patch tokenizer
✓ spatial-only embedding before matching
✓ validity-aware local cross-frame attention
✓ matched validity propagation
✓ residual motion encoder
✓ temporal lag embedding
✓ residual-only temporal transformer
✓ validity-aware temporal pooling
✓ current spatial feature fusion
✓ motion head
✓ geometric anchor / current_valid

planner.py
✓ CasADi + IPOPT NMPC
✓ current geometry determines obstacle existence
✓ dynamic confidence gates predicted motion only
✓ static obstacles preserved
✓ soft collision penalty
✓ safety slack constraints
✓ control / rate constraints
✓ robot-centric warm start re-rollout
✓ solver failure safe-stop fallback

train.py
✓ beam → patch GT
✓ binary dynamic-presence confidence target
✓ dynamic_fraction retained as auxiliary diagnostic
✓ Focal confidence loss
✓ Huber velocity loss
✓ static regularization
✓ weak spatial smoothness
✓ epoch-level classification metrics
✓ static-speed diagnostic
✓ training loop / checkpoint

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

下一阶段重点不是继续堆模型模块，而是：

```text
验证 geometry / GT
↓
验证 dynamic presence 与 velocity 是否真正可学
↓
验证 GT motion field 下 planner 的静态安全和动态收益
↓
验证 predicted field 是否能保持这些收益
```

这几组实验结果会决定 ScanFlow 是否值得继续扩展成完整论文。
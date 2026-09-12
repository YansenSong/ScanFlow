# ScanFlow v2 Motion Estimator 执行计划

## 0. 文档目的

本阶段只推进一条主线：

> **构建一个几何一致的 ScanFlow v2 motion estimator，使当前可见几何支撑、跨帧对应、连续速度和运动可信度绑定到同一实体。**

本阶段**不以闭环 NMPC 为主要目标**，也不继续优化旧 patch Transformer。

当前已有实验已经说明：

- GT motion 输入 NMPC 可以显著改善动态避障，因此 planner 接口本身具有价值。
- 原固定 angular patch 表示存在严重的 position-motion mismatch。
- 原网络在小数据上可过拟合，但独立场景中 velocity 几乎不优于 zero-velocity baseline。
- point-to-surface 几何方法已经证明：历史 2D LiDAR 中存在可恢复的动态运动信息。
- 几何方法速度精度较好，但静态误报仍偏高。
- 轻量 learned candidate scorer 能降低误报、提升 F1，但会损失 velocity EPE。

因此本阶段研究问题被限定为：

> **能否让 learning 只负责 correspondence / candidate reliability，而由 geometry 保留和细化连续速度，从而同时获得较好的动态分类和速度精度？**

---

# 1. 本阶段最终成功标准

完成本阶段后，应当有一个独立于旧 `DynamicLiDARNetwork` 的 v2 estimator，并在独立测试集上同时达到：

```text
dynamic F1              >= 0.75
dynamic EPE             <= 0.16 m/s
static false dynamic    <= 5%
static mean speed       <= 0.05 m/s

A2:
static false dynamic    <= 3%

A4:
dynamic recall          >= 0.95
dynamic EPE             <= 0.15 m/s
```

以上阈值是研发 Go/No-Go 门槛，不是论文最终指标。

此外必须满足：

```text
velocity EPE <= 当前 point-to-surface baseline + 0.02 m/s
```

也就是说：

> **分类改善不能通过牺牲明显的速度精度获得。**

如果分类指标改善而 velocity EPE 明显恶化，本阶段视为未完成。

---

# 2. 明确禁止事项

Codex 在本阶段不要做以下事情：

```text
1. 不增加旧 model.py 的 Transformer 层数。
2. 不继续围绕固定 angular patch velocity head 调参。
3. 不把更多 epoch / 更多 hidden dimension 当作主要方案。
4. 不直接让神经网络从 latent feature 自由回归完整 vx, vy。
5. 不优先修改 NMPC cost。
6. 不用 test set 选择 checkpoint、阈值或超参数。
7. 不把 unsupported 自动解释为 static。
8. 不使用 object_id 作为 estimator 输入。
9. 不让 beam index embedding 成为模型记忆固定场景位置的捷径。
10. 不删除现有 geometric / surface baseline；它们必须一直作为对照组。
```

旧 `model.py` 暂时保留，用于历史 baseline，不作为 v2 的实现基础。

---

# 3. v2 的目标接口

建议新增：

```text
motion_estimator_v2.py
```

主接口建议：

```python
result = estimator(
    lidar_history,      # [B, K, H]
    odom_history,       # [B, K, 3]
    timestamps,         # [B, K]
)
```

输出不再是固定 36 patch：

```python
{
    "points": ...,              # [B, H, 2]
    "valid": ...,               # [B, H]

    "velocity": ...,            # [B, H, 2]
    "dynamic_probability": ..., # [B, H]

    "motion_supported": ...,    # [B, H]
    "motion_confidence": ...,   # [B, H]

    "fit_residual": ...,        # [B, H]
    "candidate_entropy": ...,   # [B, H]

    # debug / research outputs
    "coarse_velocity": ...,
    "refined_velocity": ...,
    "candidate_scores": ...,
}
```

这里必须严格区分：

```text
valid
= 当前 LiDAR 是否有真实回波

dynamic_probability
= 当前可见点是否属于运动表面

motion_supported
= 历史扫描是否提供足够运动估计证据

motion_confidence
= 当前速度估计是否可信
```

不能再让一个 `confidence` 同时承担以上所有语义。

---

# 4. 表示单位

## 4.1 当前输出单位

v2 默认以：

> **当前 scan 中的每个有效 beam / point**

作为 motion 支撑单位。

即：

```text
current beam i
→ point q_i = (x_i, y_i)
→ velocity v_i
```

必须满足：

```text
q_i 与 v_i 属于同一个当前可见点。
```

不要再采用：

```text
patch centroid
+
patch dynamic average velocity
```

这种可能来自不同物理表面的组合。

---

## 4.2 后续可选压缩

第一版 v2 不需要立即解决 planner obstacle budget。

先保证 beam-level motion 是正确的。

之后才允许增加：

```text
beam-level motion
↓
surface / component clustering
↓
planner token compression
```

压缩是单独问题，不要和 motion estimation 混在第一阶段。

---

# 5. 总体架构

推荐 v2 pipeline：

```text
Raw LiDAR history
        ↓
Current-frame XY conversion
        ↓
SE(2) ego-motion alignment
        ↓
Historical finite-surface representation
        ↓
Per-current-point local context
        ↓
Coarse metric velocity hypotheses
        ↓
Geometric matching cost volume
        ↓
Learned candidate reliability / scoring
        ↓
Top-k candidate selection
        ↓
Continuous geometric velocity refinement
        ↓
Dynamic probability + support + confidence
        ↓
Beam-level motion field
```

核心思想：

> **geometry 提供真实单位的 motion hypothesis 和连续 refinement；learning 负责挑选哪些 correspondence / hypotheses 可信。**

---

# 6. Phase A — 整理公共几何模块

## 6.1 目标

把目前：

```text
geometric_motion.py
surface_motion.py
motion_cost_model.py
```

中重复的几何代码整理成明确的公共层。

建议新增：

```text
motion_geometry.py
```

至少包含：

```python
scan_to_xy()
align_history_to_current()
extract_finite_surfaces()
build_local_neighborhood()
surface_distance()
velocity_candidates()
compute_motion_cost_volume()
```

---

## 6.2 必须保持的几何语义

所有历史点：

```text
historical robot frame
→ odom/world frame
→ current robot frame
```

速度候选使用：

```math
q_i(t-\Delta t)
=
q_i(t)-v_i\Delta t
```

与历史有限 surface 比较。

必须使用真实：

```text
timestamps
dt
metric XY
```

不要只用 lag index。

---

## 6.3 验收

迁移后现有：

```text
test_surface_motion_contract
test_geometric_motion_contract
test_motion_cost_contract
```

全部继续通过。

并增加：

```text
same scan + different dt
same current scan + reversed history
missing history
SE(2) ego-motion invariance
```

契约测试。

Phase A 不改变现有 benchmark 数字。

---

# 7. Phase B — 建立统一的 coarse motion cost volume

## 7.1 速度候选

第一版 coarse grid：

```text
vx, vy ∈ [-1.5, 1.5] m/s
coarse step = 0.25 m/s
speed <= 1.5 m/s
```

目前约 113 candidates，可以继续沿用。

但代码设计不能假设永远固定这个分辨率。

---

## 7.2 每个 candidate 的基础几何特征

每个：

```text
current beam i
candidate c
```

至少产生：

```text
median historical surface distance
upper-quartile distance
mean local-neighborhood distance
static candidate distance
improvement over static
candidate speed
radial velocity
tangential velocity
number of supporting history frames
local geometry spread
```

建议特征从当前 8 维扩到约 10–14 维即可。

不要一开始堆大量 engineered feature。

---

## 7.3 支持状态

必须有显式：

```text
supported / unsupported
```

unsupported 示例：

```text
历史帧缺失
没有足够 surface
local context 少于最低点数
候选之间无法形成有效区分
```

unsupported：

```text
velocity 可以暂时输出 0
```

但：

```text
motion_supported = false
```

不能被解释成：

```text
确定静态
```

---

# 8. Phase C — Candidate Scoring v2

## 8.1 模型职责

模型只学习：

```text
P(candidate | geometric evidence)
```

或者更保守地理解为：

```text
candidate ranking score
```

不要宣称是校准概率，除非后面完成 calibration。

---

## 8.2 网络规模

第一版继续保持很小：

```text
Feature Dim ~ 10–14

MLP:
F
→ 64
→ 64
→ 1
```

参数量目标：

```text
< 10k parameters
```

不要上 Transformer。

目标是验证：

> correspondence scoring 本身是否值得学习。

---

## 8.3 几何 prior 必须保留

建议：

```math
score
=
score_geometry
+
\Delta score_{learned}
```

而不是完全替换几何 score。

初始化：

```text
learned correction ≈ 0
```

保证未训练模型接近 geometry baseline。

---

## 8.4 输出 top-k

不要只保留 argmax。

保存：

```text
top-1 velocity
top-k candidates
top-k logits
softmax entropy
score margin
```

例如：

```text
k = 3 or 5
```

这些后面用于：

```text
continuous refinement
uncertainty
ambiguity analysis
```

---

# 9. Phase D — Continuous Velocity Refinement

这是本阶段最关键的新增模块。

当前 scorer 最大问题之一：

```text
0.25 m/s coarse candidate
→ MAP
→ velocity 被离散化
```

因此学习后的 F1 提高，但 velocity EPE 反而下降。

这一阶段必须解决。

---

## 9.1 推荐方案：geometry refinement

对 scorer 选出的 top-k candidates：

```text
v_c
```

在其附近继续搜索：

```text
±0.25 m/s
```

fine step：

```text
0.05 m/s
```

或采用局部连续优化。

最终：

```math
v_{final}
=
\arg\min_v
E_{surface}(v)
```

其中初始化来自 learned top-k candidate。

---

## 9.2 学习模块不能完全覆盖 refinement

不要：

```text
MLP → arbitrary residual vx, vy
```

第一版建议：

```text
learning:
选择 coarse basin

geometry:
负责 continuous optimum
```

后续 ablation 再比较：

```text
geometry-only refinement
vs
small learned residual
```

---

## 9.3 如果加入 residual head

只能允许：

```math
v_{final}
=
v_{geo}
+
\Delta v
```

并限制：

```text
|Δvx| <= 0.10~0.15 m/s
|Δvy| <= 0.10~0.15 m/s
```

目的是修正：

```text
surface discretization
noise
local approximation bias
```

而不是重新预测 velocity。

---

# 10. Phase E — Dynamic Presence 与 Motion Confidence 分离

## 10.1 dynamic_probability

目标：

```text
当前 point 是否属于真正移动物体
```

训练标签来自 synthetic：

```text
beam_dynamic
```

可继续使用 GT。

---

## 10.2 motion_confidence

目标不是动态分类。

应该反映：

```text
当前 velocity estimate 是否可靠
```

可以先由可解释证据构造：

```text
candidate margin
entropy
fit residual
history support count
top-k agreement
```

第一版无需监督成概率。

可定义 heuristic confidence：

```text
low residual
+
large top1-top2 margin
+
low entropy
+
enough history support
```

后续再做 calibration。

---

## 10.3 motion_supported

这是 hard evidence flag：

```text
history evidence sufficient
```

必须和：

```text
dynamic_probability
```

解耦。

可能出现：

```text
dynamic_probability high
motion_supported false
```

例如：

```text
刚刚从遮挡中出现的运动物体
```

这种点不能被简单外推完整速度。

---

# 11. Phase F — 训练协议

## 11.1 数据分割

必须使用：

```text
train seed group
validation seed group
test seed group
```

严格场景独立。

例如：

```text
train:
512 scenes
seed 21001...

validation:
128 scenes
different seed range

test:
256 scenes
completely unseen seeds
```

禁止：

```text
把训练文件前 N 个样本当 test
```

---

## 11.2 第一阶段数据

仍用 synthetic，但必须增加难度分层。

Level 1：

```text
single moving circle
static robot
clean odom
clean scan
```

Level 2：

```text
moving robot
moving circle
static geometry
```

Level 3：

```text
multiple movers
near/far mixed surfaces
occlusion
partial appearance
```

Level 4：

```text
odom noise
timestamp perturbation
scan noise
different dt
```

---

## 11.3 Curriculum

不要直接训练最复杂数据。

建议：

```text
Stage 1:
single mover
no odom noise

Stage 2:
multi mover
occlusion

Stage 3:
noise + odom disturbance + dt variation
```

每一级必须在 independent validation 上保持：

```text
velocity EPE
classification
support
```

不明显退化后才能继续。

---

# 12. Phase G — Loss 设计

Candidate scorer 主 loss：

```text
candidate classification / ranking
```

推荐两种可实验选择。

方案 A：

```text
cross entropy to nearest GT candidate
```

方案 B：

使用 soft target：

```math
p(c)
\propto
\exp(-||v_c-v_{GT}||^2/\sigma^2)
```

方案 B 更适合避免：

```text
GT 位于两个 coarse candidates 中间时
只奖励一个格点
```

建议优先尝试 soft target。

---

## 12.1 dynamic loss

另外训练：

```text
dynamic presence BCE / focal BCE
```

输入可以是：

```text
top-k score statistics
fit residual
support
refined speed
```

不要从 beam position 直接分类。

---

## 12.2 可选 ranking regularization

鼓励：

```text
GT 附近 candidate
>
明显错误 candidate
```

例如 margin ranking loss。

第一版不要堆太多 loss。

推荐总损失：

```text
L
=
L_candidate
+
λ_dyn L_dynamic
```

如果有 residual head：

```text
+
λ_res L_residual
```

---

# 13. Phase H — 统一评估脚本

新增：

```text
test/evaluate_motion_v2.py
```

同一脚本同时评估：

```text
zero velocity baseline
surface_motion baseline
candidate scorer v1
motion estimator v2
```

不能让不同方法使用不同 test scenes。

---

# 14. 必须输出的指标

## 14.1 Dynamic detection

```text
precision
recall
F1
```

---

## 14.2 Velocity

```text
dynamic EPE
speed error
angular error
zero-velocity EPE
relative improvement over zero
```

---

## 14.3 Static behavior

```text
static mean predicted speed
static false dynamic rate
```

---

## 14.4 Support / uncertainty

```text
supported coverage
dynamic supported recall
unsupported fraction
candidate entropy
top1-top2 score margin
```

---

## 14.5 分桶指标

至少按照：

```text
object distance:
near / medium / far

visible dynamic beams:
1–3
4–8
9+

ego translational speed:
low / medium / high

ego angular speed:
low / medium / high

occlusion:
low / medium / high
```

分别统计。

---

# 15. Phase I — 必须保留的诊断测试

## A2

```text
Static world + robot ego motion
```

期望：

```text
static speed ~ 0
false dynamic <= 3%
```

但注意：

> A2 通过不代表 motion estimation 成功。

---

## A4

```text
moving robot + moving obstacle
```

至少包含：

```text
robot static
slow straight
fast straight
slow turn
fast turn
```

期望：

```text
recall >= 0.95
EPE <= 0.15 m/s
```

---

## History intervention

固定 current scan：

```text
history upward motion
history downward motion
history static
```

输出 velocity 必须对应变化。

---

## Time intervention

固定位移，改变：

```text
dt
```

估计速度必须按物理比例变化。

---

## No-history test

移除大部分历史：

```text
motion_supported
```

应降低。

不能：

```text
无历史
→ 高置信度速度
```

---

# 16. Phase J — 多目标和遮挡压力测试

必须新增明确的 hard cases。

## J1. 两个相向 mover

例如：

```text
object A:
vy = +0.6

object B:
vy = -0.6
```

角度上靠近。

目的：

> 确认不会再次被平均成接近零速度。

---

## J2. Static + dynamic same angular neighborhood

近动态物体：

```text
range ≈ 2 m
```

远静态墙：

```text
range ≈ 5 m
```

目的：

> 检查 velocity 是否仍绑定当前 point，而不是局部区域平均。

---

## J3. Occlusion emergence

动态物体：

```text
前几帧不可见
当前刚出现
```

期望：

```text
dynamic_probability 可以高
motion_supported 应低
```

不要伪造一个确定 velocity。

---

## J4. Tangential ambiguity

长表面做切向运动。

目的不是要求恢复不可观测速度，而是检查：

```text
motion_confidence ↓
candidate entropy ↑
```

---

# 17. Phase K — 计算性能

算法正确之前不作为第一优化目标。

但必须记录：

```text
mean latency
P50
P95
```

分别统计：

```text
geometry feature extraction
candidate scorer
refinement
total
```

目标阶段：

```text
第一版:
< 100 ms acceptable for research

优化版:
target <= 50 ms

最终部署目标:
~10 Hz or better
```

---

# 18. 推荐文件结构

建议增加：

```text
motion_geometry.py
motion_estimator_v2.py
train_motion_v2.py

test/
├── evaluate_motion_v2.py
├── test_motion_geometry_contract.py
├── test_motion_v2_contract.py
├── test_motion_v2_a2.py
├── test_motion_v2_a4.py
├── test_motion_v2_occlusion.py
├── test_motion_v2_multi_object.py
└── benchmark_motion_v2.py
```

现有：

```text
geometric_motion.py
surface_motion.py
motion_cost_model.py
```

保留作为 baseline，不要删除。

---

# 19. 实验顺序

必须严格按以下顺序推进。

## Gate 1 — Geometry refactor

要求：

```text
旧 surface baseline 数字基本复现
contract tests 全过
```

失败：

```text
先修 geometry
```

---

## Gate 2 — Candidate scorer + continuous refinement

要求：

```text
refined velocity EPE
<= surface baseline + 0.02
```

并且：

```text
F1 >= surface baseline
static FP <= surface baseline
```

失败：

> 不进入复杂训练。

---

## Gate 3 — Independent generalization

新 test seed 上要求：

```text
F1 >= 0.75
EPE <= 0.16
static FP <= 5%
```

失败：

> 分析 correspondence / support，不扩模型规模。

---

## Gate 4 — A2 / A4

要求：

```text
A2 false dynamic <= 3%

A4 recall >= 0.95
A4 EPE <= 0.15
```

失败：

> 定位 ego-motion、matching 或 support 问题。

---

## Gate 5 — Hard cases

要求：

```text
multi-object 不平均
occlusion 能标 unsupported
dt scaling 正确
history reversal 正确
```

失败：

> 不接 NMPC。

---

## Gate 6 — Planner integration

只有前面全部通过后，才做：

```text
v2 motion
↓
planner token compression
↓
NMPC
```

---

# 20. Planner integration 暂定接口

本阶段最后才处理。

不要直接把所有 beams 喂 NMPC。

建议后续：

```text
beam-level motion
↓
surface grouping / clustering
↓
risk-aware component selection
↓
planner obstacles
```

planner token 至少包含：

```text
anchor
velocity
geometry validity
dynamic probability
motion support
motion confidence
```

未来预测不要继续长期依赖：

```text
position + confidence * velocity * t
```

后续单独研究：

```text
motion hypotheses
occupancy corridor
uncertainty inflation
```

但不属于当前主线。

---

# 21. Codex 每次提交要求

每完成一个 Phase，Codex 必须：

```text
1. 修改代码。
2. 增加或更新测试。
3. 运行 contract tests。
4. 运行对应实验。
5. 在 docs/ 新增阶段总结。
6. 记录：
   - command
   - seed
   - dataset config
   - checkpoint
   - metrics
   - known limitations
7. 再提交 main。
```

建议阶段文档：

```text
docs/v2_phase_a_geometry.md
docs/v2_phase_b_cost_volume.md
docs/v2_phase_c_scorer.md
docs/v2_phase_d_refinement.md
docs/v2_phase_e_generalization.md
```

不要只在聊天会话里保留实验结果。

---

# 22. 每份阶段报告必须包含

```text
Hypothesis
Method
Code changes
Dataset
Train/val/test split
Hyperparameters
Metrics
Baseline comparison
Failure cases
Timing
Decision
Next step
```

特别是：

```text
Decision
```

必须明确写：

```text
PASS
PARTIAL PASS
FAIL
```

以及依据。

---

# 23. 第一轮最小实现建议

如果希望 Codex 先快速验证整个 v2 方向，不要一次实现所有功能。

第一轮只做：

```text
现有 surface geometry
        ↓
现有 coarse candidate grid
        ↓
learned scorer
        ↓
top-3 candidates
        ↓
每个 candidate 周围 ±0.25 m/s
0.05 m/s fine geometry search
        ↓
选择 fine geometry minimum
```

先不做：

```text
learned residual
calibration
uncertainty planner
large model
Transformer
```

这一个版本已经可以验证最关键假设：

> **learning 能否改善 candidate selection，而 continuous geometry refinement 能否把丢掉的速度精度拿回来。**

---

# 24. 第一轮关键 ablation

第一轮至少比较：

```text
A. surface geometry baseline

B. coarse scorer MAP
当前已有方法

C. geometry coarse candidate
+ geometry fine refinement

D. learned scorer top-1
+ geometry fine refinement

E. learned scorer top-3
+ geometry fine refinement
```

重点看：

```text
F1
dynamic EPE
static false positive
A4
runtime
```

如果：

```text
D/E
```

同时做到：

```text
F1 > A
EPE ≈ A 或更好
static FP < A
```

那么 ScanFlow v2 的核心方向就基本成立。

---

# 25. 预期最理想的第一轮结果

当前大致 baseline：

```text
surface:
F1 ≈ 0.66
EPE ≈ 0.15–0.16
static FP ≈ 8%

learned coarse scorer:
F1 ≈ 0.76
EPE ≈ 0.20
static FP ≈ 4.8%
```

最理想的 v2 第一阶段结果：

```text
F1 ≈ 0.75–0.80
EPE ≈ 0.13–0.16
static FP ≈ 3–5%
```

如果得到这个结果：

> 表明 learning 解决了几何 baseline 的可靠性问题，而 geometry refinement 保留了连续速度精度。

这将成为非常清楚的研究故事。

---

# 26. 如果失败，如何解释

## 情况 A

```text
F1 提高
EPE 仍然很差
```

说明：

```text
candidate selection 好
refinement basin 错
```

检查：

```text
top-k 是否包含 GT 附近 candidate
```

---

## 情况 B

```text
top-k recall 很高
final EPE 仍差
```

说明 refinement 有问题。

优化 geometry refinement。

---

## 情况 C

```text
top-k 根本经常不包含 GT
```

说明：

```text
cost volume / correspondence feature
```

信息不足。

优先改 matching feature，不增加网络规模。

---

## 情况 D

```text
static FP 高
dynamic EPE 好
```

说明 motion evidence 有，但：

```text
dynamic/support discrimination
```

不足。

加强：

```text
static hypothesis
support evidence
candidate margin
```

---

## 情况 E

```text
synthetic 很好
odom noise 后崩
```

不要直接堆网络。

需要：

```text
alignment uncertainty
odom augmentation
robust geometric cost
```

---

# 27. 研究层面的最终判断标准

完成 v2 后，必须能够回答三个问题：

## Q1

历史 2D LiDAR 中是否存在足够运动信息？

当前几何 baseline 已基本回答：

```text
YES
```

---

## Q2

learning 是否能比手工 geometry 更可靠地判断 correspondence / motion support？

当前 scorer 已显示初步：

```text
YES
```

---

## Q3

这种 learning gain 是否可以在不牺牲连续速度精度的情况下实现？

这正是本阶段唯一需要重点回答的问题。

如果答案是：

```text
YES
```

再继续：

```text
uncertainty
planner interface
closed-loop navigation
```

如果答案是：

```text
NO
```

则应重新评估 learning 在 ScanFlow 中的角色，而不是继续扩网络。

---

# 28. Codex 的第一条具体任务

建议直接从下面这个任务开始：

```text
读取：
surface_motion.py
motion_cost_model.py
geometric_motion.py
docs/surface_matching_2026-09-12.md
docs/candidate_scorer_2026-09-12.md
docs/ScanFlow_session_summary_2026-09-12.md

实现 ScanFlow v2 第一阶段 prototype：

1. 保留 beam-level current XY support。
2. 复用现有 coarse velocity candidates。
3. 使用现有 learned MotionCandidateScorer 对 coarse candidates 排序。
4. 对 top-k candidates 做连续/细网格 geometry refinement。
5. 输出 refined velocity、dynamic probability、motion_supported、
   candidate entropy、top1-top2 margin。
6. 新建统一 evaluation script。
7. 在严格独立 train/val/test seeds 下比较：
   surface baseline
   coarse scorer
   scorer + top1 refinement
   scorer + top3 refinement。
8. 不修改 planner.py。
9. 不修改旧 model.py 的结构。
10. 将完整实验结果写入 docs/v2_refinement_experiment.md。
```

第一阶段的核心验收条件：

```text
相比 surface baseline：

dynamic F1 不下降
static false dynamic 更低
dynamic EPE 不超过 baseline + 0.02 m/s

并优先争取：
F1 >= 0.75
EPE <= 0.16 m/s
static FP <= 5%
```

达到后再继续下一阶段。

---

# 29. 本阶段一句话原则

> **不要让网络重新猜速度；让几何提出和细化速度，让学习负责判断哪些运动证据值得相信。**

---

# 2026-09-12 执行记录

本计划的第一轮可运行原型已经完成。实现没有读取或修改 `baseline/`；旧 `model.py`、`geometric_motion.py`、`surface_motion.py` 和 planner baseline 继续保留作为对照。v2 本轮没有接入 planner/NMPC。

## 已完成的代码

- `motion_geometry.py`：统一 scan→XY、SE(2) 对齐、有限 surface、局部邻域、候选速度和 cost volume；
- `motion_estimator_v2.py`：beam-level batch/single API、top-k candidate、连续 `±0.25 m/s`、`0.05 m/s` 几何细化、support/confidence/evidence 输出；
- `geometric_motion.py`、`surface_motion.py`、`motion_cost_model.py`：复用公共几何层并保留旧接口；
- `test/evaluate_motion_v2.py`：zero、surface、coarse scorer、geometry refine、scorer top-1/top-3 refine 的同场景独立比较；
- `test/evaluate_motion_v2_controls.py`：A2/A4 和 history/time intervention；
- `test/test_motion_geometry_contract.py`、`test/test_motion_estimator_v2_contract.py`：新增契约覆盖。

## Gate 判定

| Gate | 当前判定 | 依据 |
|---|---|---|
| Gate 1 Geometry refactor | **PASS** | 26 项相关契约测试通过；旧 baseline 数字保持同量级；SE(2)、dt、反向 history、缺历史均有检查 |
| Gate 2 scorer + refinement | **PASS（相对）** | C/D/E 相对 A 的 F1 不降、static FP 不升、EPE 均不超过 A `+0.02 m/s` |
| Gate 3 independent generalization | **FAIL** | refined 方法 F1 `0.665–0.681`、EPE `0.169–0.173`、static FP `7.03–7.95%`，未同时达到绝对门槛 |
| Gate 4 A2/A4 | **PARTIAL PASS** | A2 全部 false dynamic `≤2.78%`；C/E A4 recall=1，但 D 部分 EPE `>0.15`，B recall `0.625–0.875` |
| Gate 5 hard cases | **PASS（诊断级）** | J1 相向目标不平均、J2 静动态邻域分离、J3 显露点 unsupported、J4 切向表面高 entropy/低 confidence 均通过；仍不是闭环安全证明 |
| Gate 6 planner integration | **BLOCKED BY GATE 3–5** | 按计划暂不接 NMPC |

## 统一独立 test（seed 20261010）

配置为 128 scenes、180 beams、6 frames、`dt=0.1 s`、LiDAR noise `0.01 m`，使用 CUDA GTX 1650。动态 detection 主指标以预测 speed `≥0.15 m/s` 判定，表中数值为 scene 指标均值。

| 方法 | F1 | dynamic EPE (m/s) | static FP | static mean speed (m/s) | mean/P95 ms |
|---|---:|---:|---:|---:|---:|
| zero velocity | 0.000 | 0.725 | 0.00% | 0.000 | 0.004 / 0.004 |
| A surface geometry | 0.663 | 0.170 | 7.95% | 0.072 | 114.0 / 195.0 |
| B coarse scorer MAP | 0.762 | 0.198 | 4.84% | 0.034 | 159.3 / 169.4 |
| C geometry coarse + refine | 0.665 | 0.169 | 7.94% | 0.075 | 187.8 / 231.3 |
| D scorer top-1 + refine | 0.681 | 0.173 | 7.03% | 0.041 | 190.6 / 232.4 |
| E scorer top-3 + refine | 0.665 | 0.169 | 7.95% | 0.061 | 240.1 / 327.8 |

B 的结果说明“学习候选排序”确实降低误报，但 coarse MAP 牺牲速度；C/D/E 说明局部几何细化可以恢复速度精度，但当前 refined estimator 尚未达到绝对研究门槛。`dynamic_probability` 还是 logit-gap / geometry proxy，未校准，不能替代正式 probability 结论。

## 受控测试

`artifacts/motion_v2/controls.json` 已保存 A2/A4 结果。A2 五种自运动条件下所有方法 false dynamic 为 `0–2.78%`；C/D/E 的 A4 recall 全部为 `1.0`，但 D 的 stationary/fast-turning EPE 约 `0.218/0.224 m/s`，因此 Gate 4 只能判定部分通过。固定当前 scan 改变 history 方向时 C/E 的 `vy` 约为 `+0.656/-0.613 m/s`；加倍 dt 的半速最大误差约 `0.025 m/s`。

## 结论与下一步

当前最有研究价值且已得到实验支持的主线是：

```text
beam-level metric geometry
→ candidate/correspondence scoring
→ finite-surface continuous refinement
→ presence / support / confidence 分离
```

它支持继续做 v2，但不支持扩大旧 Transformer 或直接接入 NMPC。下一步按顺序为：

1. 用 cost volume 的 `extended_features [H,C,11]` 训练真正的 v2 scorer（soft candidate target + 独立 dynamic-presence head），严格保持 train/validation/test 独立；
2. 完成距离/遮挡/odom noise 分层和 confidence calibration；
3. 只有 Gate 3–5 通过后，才做 beam→surface/component compression 和 planner integration。

详细结果见：

- [Phase A 公共几何层](v2_phase_a_geometry.md)；
- [Phase B cost volume](v2_phase_b_cost_volume.md)；
- [Phase C candidate scorer](v2_phase_c_scorer.md)；
- [Phase D 连续细化](v2_phase_d_refinement.md)；
- [Phase E 独立泛化](v2_phase_e_generalization.md)；
- [refinement experiment 完整记录](v2_refinement_experiment.md)。

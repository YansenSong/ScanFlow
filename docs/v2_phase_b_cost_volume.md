# v2 Phase B：统一 coarse motion cost volume

日期：2026-09-12

## 设计

每个当前有效 beam `i` 和每个速度候选 `c` 形成一个 metric hypothesis。默认候选为：

```text
vx, vy ∈ [-1.5, 1.5] m/s
速度圆盘约束
coarse step = 0.25 m/s
候选数 = 113
```

候选点被按每个真实时间间隔回推到历史时刻，并与对齐后的有限 surface 比较。代价体保留 legacy scorer 的 8 维输入：

```text
median distance
upper-quartile distance
local-neighborhood distance
static distance
static improvement
candidate speed
radial velocity
tangential velocity
```

同时暴露 v2 诊断证据：`support_frames`、`frame_spread`、`neighborhood_count`、`median_cost`、`local_cost` 和 `static_cost`，并组装成可直接供后续训练使用的 `extended_features [H,C,11]`。

## 支撑语义

`supported` 只有在当前点有效、局部邻域足够、历史中至少有两个可查询 finite-surface frame 时才为真。一个没有历史对应的点可以输出零速度作为安全 fallback，但不能被报告为“已证明静态”。

## 代码与验证

- 实现：`motion_geometry.compute_motion_cost_volume`；
- 兼容入口：`motion_cost_model.matching_features`；
- 当前 scorer 仍能直接加载 `artifacts/candidate_scorer/best.pt`；
- 空历史、无效 scan、不同 `dt` 和候选边界均有契约覆盖。

## 局限

当前训练 checkpoint 仍消费 8 维 legacy 特征；11 维扩展证据已经在 `extended_features` 中准备好，但尚未用独立的 v2 scorer 重新训练。因此本阶段证明的是“统一几何证据接口成立”，不是“扩展特征已经带来学习收益”。

## 判定

**Phase B：PASS。** cost volume 已成为 surface baseline、geometry-only v2 和 learned scorer v2 的共同输入；下一步重点是候选 ranking 与连续速度细化的消融，而不是扩展网络规模。

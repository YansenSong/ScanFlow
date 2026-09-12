# v2 Phase C：Candidate Scoring

日期：2026-09-12

## 模型职责

当前实验复用已有的轻量 `MotionCandidateScorer`，参数量 2,833：

```text
8 → 48 → 48 → 1
```

它只对每个 beam 的几何 candidate ranking 做 learned correction，不接收 GT、object id、beam position embedding，也不直接回归自由的 `(vx, vy)`。`dynamic_probability` 目前是 non-static-vs-static 的 logit-gap proxy，尚未做 calibration，不能当作校准概率使用。

## 训练数据与 checkpoint

沿用此前已经按场景独立划分并只用 validation 选 checkpoint 的实验：

| split | 场景 | seed |
|---|---:|---:|
| train | 128 | 20260930 |
| validation | 32 | 20261001 |
| test | 128 | 20261002 |

训练使用 CUDA GTX 1650、AdamW、learning rate `0.003`、weight decay `1e-4`、batch size `16`。v2 评测只加载 `artifacts/candidate_scorer/best.pt`，不在 test 上更新或选阈值。

## v2 中的使用方式

- B：trained scorer coarse MAP，直接取 coarse top-1；
- D：trained scorer top-1 作为 geometry refinement basin；
- E：trained scorer top-3 作为多个 refinement basin；
- C：不使用 learned scorer，以 geometry local cost 作为同一 cost volume 的 coarse score。

因此 C–E 能区分“geometry refinement 的收益”和“learned candidate selection 的收益”。

## 独立 test 观察

在统一的 128 个新场景（seed `20261010`）上，B 的 speed-threshold 指标为 F1 `0.762`、dynamic EPE `0.198 m/s`、static false dynamic `4.84%`。它改善了分类，却比 surface baseline 的 EPE `0.170 m/s` 高 `0.027 m/s`，违反“不以明显速度退化换分类”的约束。

这正是连续 refinement 必须存在的原因：学习模块可以挑 basin，但不应成为离散速度输出的终点。

## 判定

**Phase C：集成 PASS，研究结论 PARTIAL。** scorer 接口和独立 split 已复用成功，但本轮还没有用 `extended_features [H,C,11]` 重新训练 v2 scorer；不能把现有 checkpoint 的收益表述成最终 v2 模型收益。

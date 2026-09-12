# v2 Phase E：Independent Generalization

日期：2026-09-12

## Hypothesis

如果 v2 的几何约束确实改善了 correspondence，而不是只记住 beam/scene pattern，那么在训练和验证 seed 之外的新场景上，refinement 至少应不低于 surface baseline，并逐步接近绝对研发门槛。

## Protocol

所有 A–E 方法和 zero baseline 使用完全相同的 128 个独立场景：seed `20261010`、180 beams、6 frames、`dt=0.1 s`、LiDAR noise `0.01 m`。训练好的 scorer checkpoint 只来自 train/validation seed `20260930/20261001`；没有用本 test 选择 checkpoint、阈值或 refinement 参数。

评估记录 speed-threshold dynamic F1/EPE/static FP、support、距离分桶和 latency；`dynamic_probability` 作为未校准 diagnostic 单独报告。

## Results

| method | F1 | EPE (m/s) | static FP | far-bin F1 |
|---|---:|---:|---:|---:|
| A surface geometry | 0.663 | 0.170 | 7.95% | 0.205 |
| B coarse scorer MAP | 0.762 | 0.198 | 4.84% | 0.295 |
| C geometry + refine | 0.665 | 0.169 | 7.94% | 0.210 |
| D scorer top-1 + refine | 0.681 | 0.173 | 7.03% | 0.220 |
| E scorer top-3 + refine | 0.665 | 0.169 | 7.95% | 0.210 |

C/D/E 均满足相对 Gate 2；没有 refined 方法同时满足 `F1>=0.75`、`EPE<=0.16 m/s` 和 `static FP<=5%`。B 的 F1/static FP 达到绝对值，但 EPE 高于 A `0.027 m/s`，不能作为最终方法。

## Failure analysis

`far>=6 m` 的 A/C/D/E F1 只有约 `0.205–0.220`，EPE 约 `0.419–0.426 m/s`；这说明当前瓶颈集中在稀疏远距离回波、有限 surface overlap 和 support，而非大模型容量。独立测试也没有覆盖 odometry disturbance、遮挡等级、多种 `dt` 的系统分层，因此不能把这一轮当作最终泛化结论。

## Decision

**Gate 3：FAIL。** Gate 2 的相对不退化条件已通过，但绝对研发目标未通过。按照计划不扩大 Transformer，也不接入 NMPC。

## Next step

在 train/validation 上训练真正的 v2 scorer：使用 cost volume 扩展 evidence、soft candidate target 和独立 dynamic-presence head；然后在新的 seed group 上复核距离、可见 beam 数、odom noise、timestamp perturbation 和遮挡等级。完成这些分层前，planner integration 保持冻结。

## Reproduction

```bash
conda run --no-capture-output -n scanflow python test/evaluate_motion_v2.py \
  --samples 128 --seed 20261010 --beams 180 --device cuda \
  --checkpoint artifacts/candidate_scorer/best.pt \
  --save artifacts/motion_v2/evaluation.json
```

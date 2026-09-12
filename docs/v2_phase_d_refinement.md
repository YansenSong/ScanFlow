# v2 Phase D：Continuous Velocity Refinement

日期：2026-09-12

## 方法

对 coarse scorer 的 top-k candidate，在局部 `±0.25 m/s` 范围内以 `0.05 m/s` fine grid 做有限 surface matching。最终速度由 geometry cost 最小值决定，并带很小的 speed regularizer；没有新增 unconstrained residual head。

```text
learned/geometry score → coarse basin
                         ↓
              local finite-surface search
                         ↓
                 beam-level velocity
```

v2 输出同时保留：`coarse_velocity`、`refined_velocity`、`topk_velocity`、`candidate_scores`、`candidate_entropy`、`candidate_margin`、`fit_residual`、`motion_supported` 和 `motion_confidence`。

## 当前 test 结果

统一评测配置：128 scenes、180 beams、6 frames、`dt=0.1 s`、LiDAR noise `0.01 m`、test seed `20261010`、CUDA GTX 1650。表中 dynamic 指标使用预测 speed `>=0.15 m/s`；数值是 128 个 scene 指标的均值。

| 方法 | F1 | dynamic EPE (m/s) | static FP | static mean speed (m/s) | mean/P95 ms |
|---|---:|---:|---:|---:|---:|
| zero velocity | 0.000 | 0.725 | 0.00% | 0.000 | 0.004 / 0.004 |
| A surface baseline | 0.663 | 0.170 | 7.95% | 0.072 | 114.0 / 195.0 |
| B scorer coarse MAP | 0.762 | 0.198 | 4.84% | 0.034 | 159.3 / 169.4 |
| C geometry + refine | 0.665 | 0.169 | 7.94% | 0.075 | 187.8 / 231.3 |
| D scorer top-1 + refine | 0.681 | 0.173 | 7.03% | 0.041 | 190.6 / 232.4 |
| E scorer top-3 + refine | 0.665 | 0.169 | 7.95% | 0.061 | 240.1 / 327.8 |

相对 A，C/D/E 的 EPE 分别变化 `-0.0017`、`+0.0028`、`-0.0019 m/s`，且 F1 没有下降、静态 FP 没有升高（D 的相对 F1 提升最大）。因此 Gate 2 的相对条件通过；但绝对目标 `F1>=0.75、EPE<=0.16、static FP<=5%` 在 refinement 方法上仍未同时达到。B 达到 F1 和 static FP 的绝对值，却以 EPE `+0.027 m/s` 为代价。

v2 的 probability proxy（不是 calibration）在 B/D/E 上为 F1 `0.695`、static FP `7.39%`；C 的 geometry proxy 为 F1 `0.592`、static FP `11.22%`。这说明 presence score 仍需要独立的 calibration/dynamic head，不能直接替代速度阈值指标。

## 受控 A2/A4

复用 beam-level controlled helper；它以 speed threshold 统计，尚未将 v2 probability 作为唯一 gate。

- A2 五种自运动：所有方法 false dynamic `0%–2.78%`，均不超过 `3%`；
- C/D/E A4 五种自运动：dynamic recall 均为 `1.0`；
- C/E A4 EPE 为 `0.050–0.109 m/s`；D 为 `0.062–0.224 m/s`，快速 turning 和 stationary case 仍较差；
- B coarse scorer A4 recall 为 `0.625–0.875`，说明离散 scorer 不能直接接管速度输出；
- 固定当前 scan 的 history intervention 中，C/E 的 `vy` 随历史方向变为约 `+0.656/-0.613 m/s`，加倍 `dt` 的半速最大误差约 `0.025 m/s`。

## 性能分解

v2 的平均耗时（ms）如下：

| 方法 | geometry | scorer | refinement/post | total |
|---|---:|---:|---:|---:|
| B | 113.55 | 1.17 | 43.68 | 158.40 |
| C | 113.63 | 0.01 | 73.16 | 186.80 |
| D | 113.49 | 0.87 | 75.29 | 189.66 |
| E | 113.55 | 0.91 | 124.69 | 239.15 |

当前 CPU/SciPy 几何查询仍是主要开销；这只是研究原型，尚未满足最终 10 Hz 部署预算。

## Hard cases

`test/evaluate_motion_v2_hard_cases.py` 在 CPU 上运行四个确定性诊断：

- J1 两个相向 mover 的平均 `vy` 为约 `-0.596/+0.596 m/s`，没有被平均到零；
- J2 近处动态圆平均速度约 `0.591 m/s`，同角邻域远处静态线段平均速度为 `0`；
- J3 当前刚显露、历史五帧均无回波的点为 `motion_supported=False` 且速度为零；
- J4 切向移动长表面平均速度约 `0.014 m/s`，candidate entropy `4.722`（接近 `log(113)=4.727`），motion confidence `0.0006`。

这四项验证的是表示和证据状态，不是 noisy multi-scene accuracy 或闭环安全门。

## 判定

**Phase D：PARTIAL PASS。** “候选选择 + 几何连续细化”成功消除了粗网格带来的明显速度退化，并通过 Gate 2 相对门禁；Gate 3 绝对目标和 Gate 4 的所有 A4 EPE 仍未通过。J1–J4 诊断级 hard cases 已通过，但还需要距离、odom noise、遮挡等级和 calibration 的独立统计。下一步应优先处理 correspondence/support、远距离和遮挡，而不是增加 Transformer 容量。

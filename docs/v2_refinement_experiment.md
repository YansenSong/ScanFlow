# ScanFlow v2 refinement experiment

日期：2026-09-12

## Hypothesis

已有 learned candidate scorer 的分类收益伴随速度 EPE 退化，主要原因是它把 `0.25 m/s` coarse MAP 当作最终速度。若让 learning 只选择 coarse correspondence basin，再在有限历史 surface 上做连续几何细化，应该可以保留低静态误报，同时恢复 surface baseline 的连续速度精度。

## Method

实现了独立于旧 `DynamicLiDARNetwork` 的 `MotionEstimatorV2`：

```text
current beam XY
→ SE(2)-aligned finite surfaces
→ 113 metric velocity candidates
→ geometry / learned candidate ranking
→ top-1 or top-3 local ±0.25 m/s, 0.05 m/s refinement
→ velocity + presence/support/confidence evidence
```

每个输出速度都绑定当前有效 beam。`motion_supported` 是历史几何证据的 hard flag；`dynamic_probability` 是当前的 ranking-derived diagnostic proxy，未 calibration；`motion_confidence` 综合 residual、margin、entropy 和 support，未作为监督概率。

## Code changes

- `motion_geometry.py`：公共 scan-to-XY、SE(2)、finite-surface、邻域、candidate 和 cost volume；
- `motion_estimator_v2.py`：batch/single NumPy/Torch API、CUDA scorer、top-k、连续 geometry refinement、证据输出和耗时分解；
- `geometric_motion.py`、`surface_motion.py`、`motion_cost_model.py`：改为复用公共几何层，保留旧 public API；
- `test/evaluate_motion_v2.py`：同一独立 scene 同时比较 zero、surface、coarse scorer、geometry refine、scorer top-1/top-3 refine；
- `test/evaluate_motion_v2_controls.py`：统一 A2/A4 和 history/time intervention；
- `test/evaluate_motion_v2_hard_cases.py`、`test/test_motion_v2_hard_cases.py`：J1–J4 多目标、邻域分离、显露和切向歧义诊断；
- `test/test_motion_geometry_contract.py`、`test/test_motion_estimator_v2_contract.py`：接口、形状、SE(2)、缺历史、反向运动、细化契约；
- 没有修改 `model.py` 或 `planner.py` 的 v2 逻辑，也没有删除旧几何 baseline。

## Dataset and split

统一 test 使用 `generate_dataset.py` 的独立合成场景：

```text
scenes = 128
seed = 20261010
beams = 180
frames = 6
scan_dt = 0.1 s
LiDAR noise = 0.01 m
```

每个方法读取同一批场景；原 learned scorer checkpoint 在更早的独立划分上训练/选取：train seed `20260930`（128）、validation seed `20261001`（32）、test seed `20261002`（128）。本轮 refinement test 没有用 test 结果调参或挑 checkpoint。原始 test arrays 保存到 `artifacts/motion_v2/test_seed_20261010.npz`（该目录被 `.gitignore` 忽略）。

## Hyperparameters

```text
candidate speed max = 1.5 m/s
coarse step = 0.25 m/s (113 candidates)
neighborhood radius = 0.35 m
neighborhood beam radius = 3
refinement radius = 0.25 m/s
refinement step = 0.05 m/s
noise tolerance = 0.025 m
max residual = 0.06 m
minimum static improvement = 0.015 m
top-k = 1 or 3
```

Candidate scorer 是已有的 `8→48→48→1`、2,833 参数 checkpoint；没有新增大模型或直接 residual velocity head。

## Metrics and baselines

主要 detection signal 是预测 speed `>=0.15 m/s`。记录 dynamic precision/recall/F1、dynamic EPE、speed/angular error、zero-EPE gain、static mean speed、static false dynamic、support coverage、dynamic support recall、unsupported fraction、candidate entropy、top1-top2 margin、距离分桶和 latency。

关键结果（128 scene 均值）：

| method | F1 | EPE (m/s) | static FP | support | mean/P95 ms |
|---|---:|---:|---:|---:|---:|
| zero velocity | 0.000 | 0.725 | 0.00% | 0.0% | 0.004 / 0.004 |
| A surface geometry | 0.663 | 0.170 | 7.95% | 80.70% | 114.0 / 195.0 |
| B coarse scorer MAP | 0.762 | 0.198 | 4.84% | 84.80% | 159.3 / 169.4 |
| C geometry coarse + refine | 0.665 | 0.169 | 7.94% | 80.67% | 187.8 / 231.3 |
| D scorer top-1 + refine | 0.681 | 0.173 | 7.03% | 79.82% | 190.6 / 232.4 |
| E scorer top-3 + refine | 0.665 | 0.169 | 7.95% | 80.71% | 240.1 / 327.8 |

其中 D 的 dynamic F1 相对 A 增加 `0.018`，E 的 EPE 相对 A 降低 `0.0019 m/s`；三种 refinement 都满足：F1 不低于 A、static FP 不高于 A、EPE 不超过 A `+0.02 m/s`。这构成 Gate 2 的相对 PASS。

但绝对门槛仍未达到：refined 方法 F1 `0.665–0.681`、EPE `0.169–0.173 m/s`、static FP `7.03–7.95%`。B 虽有 F1 `0.762` 和 static FP `4.84%`，EPE `0.198 m/s`，不能作为最终方法。Probability proxy 的 B/D/E F1 为 `0.695`、static FP `7.39%`，尚未校准。

## Failures and diagnostics

1. **粗候选离散化仍是 B 的主要损失。** 学习 scorer 能判断静态/动态，但 coarse MAP 的 EPE 比 A 高 `0.027 m/s`；refinement 恢复了这一差距，却没有解决远距离动态 beams 的漏检。
2. **远距离退化。** A 的 `far>=6 m` 分桶 F1 约 `0.205`、EPE 约 `0.426 m/s`；C/D/E 的远距离 F1 仍约 `0.210–0.220`。主要瓶颈是稀疏回波、遮挡和有限 surface overlap，而非 MLP 容量。
3. **A4 受控 case 并不等同于独立 test。** C/E 五种自运动下 recall 为 1.0，但 D 的 stationary/fast-turning EPE 分别约 `0.218/0.224 m/s`；B recall 只有 `0.625–0.875`。因此不能把 A4 的单一 helper 结果外推为完整泛化。
4. **support 与 presence 仍需独立 calibration。** 无历史时 v2 保证 `motion_supported=False`、速度为零；当前 probability proxy 在这种场景不承担“确定静态”的语义。
5. **J1–J4 诊断通过但范围有限。** 两个相向 mover 保持相反方向（约 `-0.596/+0.596 m/s`）；近动态+远静态邻域分别保持约 `0.591/0 m/s`；显露点被标 unsupported；切向长表面 entropy `4.722`、confidence `0.0006`。这些是确定性 CPU case，不能替代 noisy multi-scene 统计。
6. **性能未达部署目标。** 几何 cost volume 占 v2 平均约 `114 ms`，E 的 top-3 refinement 总耗时约 `242 ms`，当前只适合研究验证。

## Timing

平均 component timing（ms）：

| method | geometry | scorer | refinement/post | total |
|---|---:|---:|---:|---:|
| B | 113.55 | 1.17 | 43.68 | 158.40 |
| C | 113.63 | 0.01 | 73.16 | 186.80 |
| D | 113.49 | 0.87 | 75.29 | 189.66 |
| E | 113.55 | 0.91 | 124.69 | 239.15 |

GPU 用于 scorer tensor 前向和 checkpoint 加载；有限 surface 查询、分段和 refinement 仍在 NumPy/SciPy CPU 上，因此“有 GPU”不等于当前 pipeline 已 GPU 化。

## Decision

**相对 Gate 2：PASS。Gate 3 绝对目标：FAIL。Phase D：PARTIAL PASS。**

研究方向获得了正面但有限的证据：把学习限制为 correspondence/candidate reliability，再用 geometry 保留连续速度，确实消除了 coarse scorer 的明显 EPE 退化；但是当前 support 和远距离/遮挡处理还不足以满足最终门槛，也没有资格接 planner/NMPC。

## Next step

按优先级：

1. 在 train/validation 上训练真正的 v2 scorer：使用已有 `extended_features [H,C,11]`、soft candidate target 和独立 dynamic-presence head；test 仍只在最后使用；
2. 以距离、可见 beam 数、odom noise、`dt` 和遮挡等级分层，建立 calibration/Brier/ECE；
3. 在所有 Gate 3–5 通过前不接 NMPC。之后再做 beam→surface/component compression 和闭环规划。

## Reproduction

```bash
conda run --no-capture-output -n scanflow python -m unittest \
  test.test_motion_cost_contract \
  test.test_motion_geometry_contract \
  test.test_motion_estimator_v2_contract \
  test.test_geometric_motion_contract \
  test.test_surface_motion_contract

conda run --no-capture-output -n scanflow python test/evaluate_motion_v2.py \
  --samples 128 --seed 20261010 --beams 180 --device cuda \
  --checkpoint artifacts/candidate_scorer/best.pt \
  --save artifacts/motion_v2/evaluation.json \
  --save-data artifacts/motion_v2/test_seed_20261010.npz

conda run --no-capture-output -n scanflow python test/evaluate_motion_v2_controls.py \
  --device cuda --checkpoint artifacts/candidate_scorer/best.pt \
  --save artifacts/motion_v2/controls.json

conda run --no-capture-output -n scanflow python test/evaluate_motion_v2_hard_cases.py \
  --save artifacts/motion_v2/hard_cases.json
```

完整原始 JSON 结果在 `artifacts/motion_v2/evaluation.json` 和 `artifacts/motion_v2/controls.json`；两者均为本地实验产物并被 `.gitignore` 忽略。

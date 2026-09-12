# v2 Phase A：公共几何层

日期：2026-09-12

## 目标

将旧 `geometric_motion.py`、`surface_motion.py` 和 `motion_cost_model.py` 中重复的 LiDAR 几何语义收拢到 `motion_geometry.py`，让所有方法共享同一套：

```text
polar scan → metric XY
历史 robot frame → world/odom → current robot frame
相邻有效回波 → finite surfaces
当前点局部邻域 → 速度候选代价
```

公共层不读取 `object_id`，也不依赖旧 patch Transformer。

## 已实现

`motion_geometry.py` 现在提供：

- `scan_to_xy`：显式 range validity、beam angle 和 XY 单位；
- `align_history_to_current`：完整 SE(2) 对齐；
- `extract_finite_surfaces`：保留间隙的有限线段分段；
- `build_local_neighborhood`：当前 beam 的局部几何上下文；
- `FiniteSurfaceDistance` / `surface_distance`：有限端点距离；
- `velocity_candidates`：可配置的速度圆盘网格；
- `motion_cost_for_points`、`compute_motion_cost_volume`：统一的历史匹配代价和证据。

旧模块仍保留原有 public name：`geometric_motion.extract_surfaces` 和 `surface_motion.extract_surfaces` 都是公共实现的兼容别名；旧 baseline 算法本身没有被删除。

## 语义约束

- 速度回推使用 `q(t-Δt)=q(t)-vΔt`；`Δt` 来自真实 `timestamps`，不是 frame index；
- 当前输出仍以有效 beam/point 为单位；
- `valid`、`motion_supported`、`dynamic_probability`、`motion_confidence` 不合并；
- 历史不足时允许有限 debug cost，但 `motion_supported=False`，v2 速度回退为零；
- cost volume 的前 8 个特征保持现有 `MotionCandidateScorer` checkpoint 兼容，扩展证据单独返回。

## 验收

```bash
conda run --no-capture-output -n scanflow python -m unittest \
  test.test_motion_geometry_contract \
  test.test_motion_cost_contract \
  test.test_geometric_motion_contract \
  test.test_surface_motion_contract
```

当前公共几何、候选代价和两个旧 baseline 的相关契约测试均通过；v2 总契约测试合计 26 项通过。新增检查覆盖：

- 相同 scan 在不同绝对 pose 下的 SE(2) 一致性；
- 有限线段端点和 gap 不被跨越；
- 相同当前 scan、反向历史运动得到反向速度；
- 历史缺失不会变成“确定静态”；
- scan、candidate、evidence 的形状和有限性。

## 判定

**Phase A：PASS。** 迁移只改变代码复用位置，没有修改旧 baseline 的算法接口或 planner/model。下一阶段可以在统一 cost volume 上比较 scorer 和 refinement。

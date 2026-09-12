# ScanFlow early validation tests

These are **research validation experiments**, not conventional unit tests. A non-zero exit code means the current research hypothesis did not meet the configured gate; it does not necessarily mean the Python program is broken.

Run commands from the repository root.

## 1. Planner isolation: GT Motion Field + NMPC

```bash
python test/test_gt_motion_field_nmpc.py
```

This bypasses the neural network and compares a crossing scenario under:

- static-motion assumption (`confidence=0`, moving obstacle treated as stationary in prediction)
- GT motion field (`confidence=1`, true obstacle velocity)

**Interpretation:** if GT motion does not reduce collision exposure or increase minimum clearance, fix/tune the planner before spending time on the network.

Optional trajectory dump:

```bash
python test/test_gt_motion_field_nmpc.py --save /tmp/gt_nmpc.npz
```

## 2. Does a checkpoint learn motion?

```bash
python test/test_motion_learning.py \
  --checkpoint checkpoints/best.pt \
  --data val_data.npz
```

Reports:

- dynamic precision / recall / F1
- dynamic velocity EPE
- zero-velocity baseline EPE
- EPE improvement over the zero baseline
- predicted speed on static patches

Default gate:

- F1 >= 0.80
- static predicted speed <= 0.15 m/s
- velocity EPE improves on the zero baseline by >= 20%

For the very first clean overfit experiment, stricter targets such as F1 > 0.95 are reasonable.

## 3. A2: static world + robot ego-motion

```bash
python test/test_a2_static_ego_motion.py \
  --checkpoint checkpoints/best.pt
```

The world is fully static while robot translation/rotation changes. It checks:

- static predicted speed
- false dynamic rate
- raw-history range MAD versus ego-aligned range MAD

A good model should not convert robot ego-motion into external obstacle velocity.

## 4. A4: moving robot + moving obstacle

```bash
python test/test_a4_moving_robot_moving_obstacle.py \
  --checkpoint checkpoints/best.pt
```

A visible circle moves at a fixed world velocity while robot speed and turn rate change. It checks whether:

- the moving patch remains detected
- velocity EPE remains useful
- predicted velocity beats the zero-velocity baseline

## Recommended order

1. `test_gt_motion_field_nmpc.py`
2. overfit a tiny clean dataset (128-512 samples)
3. `test_motion_learning.py`
4. `test_a2_static_ego_motion.py`
5. `test_a4_moving_robot_moving_obstacle.py`

Do not tune thresholds to make a weak model pass. The threshold flags exist so experiments can deliberately define stricter or looser research gates.

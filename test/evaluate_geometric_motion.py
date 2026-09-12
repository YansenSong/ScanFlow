"""Independent synthetic evaluation and temporal intervention diagnostics."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_dataset import GeneratorConfig, Circle, generate_one_sample
from geometric_motion import estimate_motion
from test.common import constant_control_odom, default_static_scene, render_history, patch_targets_from_scene
from model import ModelConfig


def metrics(pred, gt, dynamic, valid, supported):
    dynamic = dynamic.astype(bool) & valid
    static = valid & ~dynamic
    detected = np.linalg.norm(pred, axis=-1) >= .15
    tp = int((detected & dynamic).sum())
    fp = int((detected & static).sum())
    fn = int((~detected & dynamic).sum())
    epe = float(np.linalg.norm(pred[dynamic] - gt[dynamic], axis=-1).mean()) if dynamic.any() else None
    zero = float(np.linalg.norm(gt[dynamic], axis=-1).mean()) if dynamic.any() else None
    return dict(dynamic_beams=int(dynamic.sum()), precision=tp/max(tp+fp, 1),
                recall=tp/max(tp+fn, 1), f1=2*tp/max(2*tp+fp+fn, 1),
                dynamic_epe_mps=epe, zero_epe_mps=zero,
                gain=1-epe/max(zero, 1e-8) if zero is not None else None,
                static_speed_mps=float(np.linalg.norm(pred[static], axis=-1).mean()) if static.any() else None,
                static_false_dynamic_rate=float(detected[static].mean()) if static.any() else None,
                supported_fraction=float(supported[valid].mean()),
                dynamic_supported_fraction=float(supported[dynamic].mean()) if dynamic.any() else None)


def controlled(estimator=estimate_motion):
    results = {}
    cfg = GeneratorConfig(num_beams=180, lidar_noise_std=0.)
    segments, static_circles = default_static_scene()
    for moving in (False, True):
        circles = static_circles + ([Circle(np.array([3., .35]), .4, np.array([0., .65]), True, 100)] if moving else [])
        for name, v, w in [('stationary', 0., 0.), ('slow', .25, 0.), ('straight', .8, 0.),
                           ('slow_turning', .45, .6), ('turning', .8, .9)]:
            odom = constant_control_odom(6, .1, v, w)
            lidar, ids = render_history(cfg, odom, segments, circles)
            _, gt, dyn, valid = patch_targets_from_scene(ModelConfig(num_beams=180), ids[-1], circles)
            times = np.arange(6) * .1
            out = estimator(lidar, odom, times)
            key = ('A4' if moving else 'A2') + '_' + name
            results[key] = metrics(out['beam_velocity'], gt, dyn, valid > .5, out['beam_supported'])
    # Isolated visible object: ensure the time-scaling check is not vacuous
    # because the challenging A4 object happened to remain unmatched.
    times = np.arange(6)*.1
    odom = np.zeros((6, 3))
    variants = {}
    for label, speed in [('forward', .65), ('backward', -.65), ('still', 0.)]:
        circle = Circle(np.array([3., .35]), .4, np.array([0., speed]), True, 100)
        lidar, ids = render_history(cfg, odom, [], [circle])
        out = estimator(lidar, odom, times)
        mask = ids[-1] >= 0
        variants[label] = dict(mean_velocity=out['beam_velocity'][mask].mean(0).tolist(),
                               supported_fraction=float(out['beam_supported'][mask].mean()))
        if label == 'forward':
            doubled = estimator(lidar, odom, 2*times)
            variants['doubled_dt_half_velocity_max_error'] = float(np.abs(doubled['beam_velocity'] - out['beam_velocity']/2).max())
    results['temporal_interventions_isolated_object'] = variants
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20260920)
    parser.add_argument('--beams', type=int, default=180)
    parser.add_argument('--method', choices=['centroid', 'surface'], default='centroid')
    parser.add_argument('--save', default='/tmp/scanflow_geometry_evaluation.json')
    args = parser.parse_args()
    estimator = estimate_motion
    if args.method == 'surface':
        from surface_motion import estimate_surface_motion
        estimator = estimate_surface_motion
    cfg = GeneratorConfig(num_samples=args.samples, num_beams=args.beams, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    arrays = [[] for _ in range(5)]
    durations = []
    for i in range(args.samples):
        sample = generate_one_sample(rng, cfg)
        started = time.perf_counter()
        out = estimator(sample['lidar_history'], sample['odom_history'], np.arange(6)*cfg.scan_dt)
        durations.append(time.perf_counter()-started)
        values = (out['beam_velocity'], sample['beam_velocity'], sample['beam_dynamic'], sample['beam_valid'] > .5,
                  out['beam_supported'])
        for array, value in zip(arrays, values):
            array.append(value)
        if (i+1) % 32 == 0:
            print(f'evaluated {i+1}/{args.samples}', flush=True)
    report = dict(seed=args.seed, samples=args.samples, beams=args.beams, method=args.method, controlled_beams=180,
                  runtime_ms=dict(mean=float(np.mean(durations)*1000), p95=float(np.percentile(durations, 95)*1000)),
                  independent=metrics(*(np.concatenate(a) for a in arrays)),
                  controlled=controlled(estimator))
    Path(args.save).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

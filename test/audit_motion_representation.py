"""Oracle representation audit, not a learned-model accuracy benchmark.

Compare the existing angular patch contract with geometry-only range segments.
GT velocity is deliberately supplied to both; object IDs are diagnostics only.
Future error measures advection of currently visible samples, not future visibility.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def groups_for_scan(points, valid, patches, segmented):
    if len(valid) % patches:
        raise ValueError('Beam count must be divisible by patch count.')
    width = len(valid) // patches
    groups = []
    for start in range(0, len(valid), width):
        current = []
        for i in range(start, start + width):
            if not valid[i]:
                if current:
                    groups.append(np.array(current))
                    current = []
                continue
            # Fixed metric threshold is an audit baseline, not a tuned segmenter.
            if segmented and current and np.linalg.norm(points[i] - points[current[-1]]) > 0.25:
                groups.append(np.array(current))
                current = []
            current.append(i)
        if current:
            groups.append(np.array(current))
    if not segmented:
        # Match production: all valid returns within a patch share one anchor.
        groups = [np.flatnonzero(valid[start:start + width]) + start
                  for start in range(0, len(valid), width)]
        groups = [g for g in groups if len(g)]
    return groups


def audit(path, patches=36):
    data = np.load(path)
    ranges = data['lidar_history'][:, -1]
    velocities = data['beam_velocity']
    dynamic = data['beam_dynamic'] > 0.5
    valid = data['beam_valid'] > 0.5
    ids = data['beam_object_id'] if 'beam_object_id' in data else None
    # Existing generated experiments use full-circle scans starting at -pi.
    angles = -np.pi + np.arange(ranges.shape[1]) * 2 * np.pi / ranges.shape[1]
    unit = np.stack([np.cos(angles), np.sin(angles)], -1)
    report = {}
    for segmented in (False, True):
        errors = {str(t): [] for t in (0, 0.5, 1.0)}
        dynamic_anchor_error = []
        mixed = multi = total = moving = 0
        counts = []
        for n, scan in enumerate(ranges):
            points = np.where(valid[n], scan, 0)[:, None] * unit
            groups = groups_for_scan(points, valid[n], patches, segmented)
            counts.append(len(groups))
            for g in groups:
                total += 1
                dg = dynamic[n, g]
                anchor = points[g].mean(0)
                velocity = velocities[n, g[dg]].mean(0) if dg.any() else np.zeros(2)
                if dg.any():
                    moving += 1
                    mixed += int(not dg.all())
                    dynamic_anchor_error.append(float(np.linalg.norm(anchor - points[g[dg]].mean(0))))
                    if ids is not None:
                        multi += int(len(np.unique(ids[n, g[dg]])) > 1)
                for t in (0, 0.5, 1.0):
                    target = points[g] + t * velocities[n, g]
                    error = np.linalg.norm(target - (anchor + t * velocity), axis=-1)
                    errors[str(t)].extend(error.tolist())
        report['segments' if segmented else 'patches'] = {
            'samples': len(ranges), 'groups': total, 'dynamic_groups': moving,
            'mean_groups_per_scan': float(np.mean(counts)),
            'mixed_static_dynamic_fraction_of_dynamic_groups': mixed / max(moving, 1),
            'multiple_dynamic_object_fraction': multi / max(moving, 1) if ids is not None else None,
            'dynamic_anchor_offset_mean_m': float(np.mean(dynamic_anchor_error)),
            'dynamic_anchor_offset_p95_m': float(np.percentile(dynamic_anchor_error, 95)),
            'visible_point_reconstruction': {
                t: {'mean_m': float(np.mean(e)), 'p95_m': float(np.percentile(e, 95))}
                for t, e in errors.items()
            },
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--save', required=True)
    args = parser.parse_args()
    report = audit(args.data)
    Path(args.save).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

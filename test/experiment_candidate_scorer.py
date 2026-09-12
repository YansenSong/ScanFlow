"""Separate-seed feasibility experiment; labels only enter supervision/metrics."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_dataset import GeneratorConfig, generate_one_sample
from motion_cost_model import MotionCandidateScorer, matching_features
from surface_motion import estimate_surface_motion
from test.evaluate_geometric_motion import metrics, controlled


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for split, count, seed in [('train', 128, 20260930), ('val', 32, 20261001), ('test', 128, 20261002)]:
        cfg = GeneratorConfig(num_beams=180, seed=seed)
        rng = np.random.default_rng(seed)
        rows = {key: [] for key in ['features', 'supported', 'velocity', 'dynamic', 'valid', 'surface_velocity', 'surface_supported']}
        for i in range(count):
            sample = generate_one_sample(rng, cfg)
            times = np.arange(6)*.1
            features, supported = matching_features(sample['lidar_history'], sample['odom_history'], times)
            baseline = estimate_surface_motion(sample['lidar_history'], sample['odom_history'], times)
            values = [features, supported, sample['beam_velocity'], sample['beam_dynamic'], sample['beam_valid']>.5,
                      baseline['beam_velocity'], baseline['beam_supported']]
            for key, value in zip(rows, values):
                rows[key].append(value)
            if (i+1)%16 == 0:
                print(f'{split}: {i+1}/{count}', flush=True)
        np.savez_compressed(directory/f'{split}.npz', seed=seed, **{k: np.asarray(v) for k, v in rows.items()})


def load(path, device, n=None):
    with np.load(path) as data:
        return {k: torch.as_tensor(data[k][:n], device=device) for k in
                ['features', 'supported', 'velocity', 'dynamic', 'valid', 'surface_velocity', 'surface_supported']}


def loss_fn(model, batch):
    logits = model(batch['features'])
    distance = (batch['velocity'][..., None, :] - model.candidates).square().sum(-1)
    target = distance.argmin(-1)
    loss = F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction='none').reshape(target.shape)
    dyn = batch['dynamic'].bool() & batch['valid'] & batch['supported']
    static = ~batch['dynamic'].bool() & batch['valid'] & batch['supported']
    return sum(loss[mask].mean() if mask.any() else logits.sum()*0 for mask in (dyn, static))


@torch.no_grad()
def evaluate(model, data):
    velocity, _ = model.predict(data['features'], data['supported'])
    def report(pred, supported):
        arrays = [pred, data['velocity'], data['dynamic'], data['valid'], supported]
        arrays = [x.cpu().numpy().reshape(-1, 2) if x.ndim == 3 else x.cpu().numpy().reshape(-1) for x in arrays]
        return metrics(*arrays)
    nearest = (data['velocity'][..., None, :] - model.candidates).square().sum(-1).argmin(-1)
    oracle = model.candidates[nearest] * data['supported'][..., None]
    return dict(learned=report(velocity, data['supported']),
                surface=report(data['surface_velocity'], data['surface_supported']),
                oracle_candidate_epe_mps=report(oracle, data['supported'])['dynamic_epe_mps'])


def train(directory, overfit, epochs):
    if not torch.cuda.is_available():
        raise RuntimeError('This experiment requires the requested CUDA device.')
    torch.manual_seed(42)
    device = torch.device('cuda')
    model = MotionCandidateScorer().to(device)
    train_data = load(directory/'train.npz', device, 16 if overfit else None)
    val_data = train_data if overfit else load(directory/'val.npz', device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=1e-4)
    initial_train = evaluate(model, train_data)
    initial_val = evaluate(model, val_data)
    best = float('inf')
    path = directory/('overfit.pt' if overfit else 'best.pt')
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for ids in torch.randperm(len(train_data['features']), device=device).split(16):
            batch = {k: v[ids] for k, v in train_data.items()}
            loss = loss_fn(model, batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model, val_data))
        if val_loss < best:
            best = val_loss
            torch.save(dict(model=model.state_dict(), epoch=epoch+1, val_loss=best), path)
        if (epoch+1)%25 == 0:
            print(f'epoch={epoch+1} validation_loss={val_loss:.4f} best={best:.4f}', flush=True)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    report = dict(device=torch.cuda.get_device_name(), epochs=epochs, best_epoch=checkpoint['epoch'],
                  train_seconds=time.perf_counter()-started, overfit=overfit,
                  parameters=sum(p.numel() for p in model.parameters()),
                  optimizer=dict(name='AdamW', learning_rate=.003, weight_decay=1e-4, batch_size=16),
                  initial_train=initial_train, initial_validation=initial_val, train=evaluate(model, train_data))
    report['data_seeds'] = {}
    for split in ['train', 'val', 'test']:
        with np.load(directory/f'{split}.npz') as source:
            report['data_seeds'][split] = int(source['seed'])
    if not overfit:
        report['validation'] = evaluate(model, val_data)
        test_data = load(directory/'test.npz', device)
        report['test'] = evaluate(model, test_data)
        report['initial_test'] = evaluate(MotionCandidateScorer().to(device).eval(), test_data)
        def estimator(lidar, odom, times):
            features, supported = matching_features(lidar, odom, times)
            with torch.no_grad():
                velocity, _ = model.predict(torch.from_numpy(features).to(device), torch.from_numpy(supported).to(device))
            return dict(beam_velocity=velocity.cpu().numpy(), beam_supported=supported)
        report['controlled'] = controlled(estimator)
    (directory/('overfit_report.json' if overfit else 'report.json')).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=Path('artifacts/candidate_scorer'))
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--overfit', action='store_true')
    parser.add_argument('--epochs', type=int, default=150)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.directory)
    else:
        train(args.directory, args.overfit, args.epochs)

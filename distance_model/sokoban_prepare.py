"""Freeze paired CRTR banks, rejecting corrupted frames and board leakage.

Exact BFS distances are reused, never replaced by trajectory gaps. Split geometry
is canonical under D4; test/solver geometries are excluded from training entirely.
The optional gap bank is a separately named, matched noisy-supervision control.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def solved_goals(frames):
    x = np.asarray(frames).reshape(-1, 144)
    out = np.ones_like(x, dtype=np.uint8)
    out[x == 0] = 0
    out[np.isin(x, (2, 3, 6))] = 3
    return out


def board_keys(frames):
    goals = solved_goals(frames).reshape(-1, 12, 12)
    transforms = [np.rot90(goals, k, (1, 2)) for k in range(4)]
    transforms += [np.flip(x, 2) for x in transforms]
    return [hashlib.sha256(min(x[i].tobytes() for x in transforms)).hexdigest()
            for i in range(len(goals))]


def split_boards(keys, blocked, validation_fraction=.05):
    excluded = np.array([k in blocked for k in keys], dtype=bool)
    val = np.array([int(k[:16], 16) / 2**64 < validation_fraction for k in keys])
    return ~val & ~excluded, val & ~excluded, excluded


def valid_rows(states, frames0):
    s = np.asarray(states).reshape(-1, 144)
    f = np.asarray(frames0).reshape(-1, 144)
    target = np.isin(f, (2, 3, 6))
    nt = target.sum(1)
    boxes = np.isin(s, (3, 4))
    agent = np.isin(s, (5, 6)).sum(1)
    return ((s <= 6).all(1) & (nt > 0) & ((s == 0) == (f == 0)).all(1)
            & (np.isin(s, (2, 3, 6)) == target).all(1)
            & (boxes.sum(1) == nt)
            & ((agent == 1) | ((agent == 0) & (boxes == target).all(1))))


def validate_lut(traj, idx, dist, shape):
    if not (traj.ndim == idx.ndim == dist.ndim == 1 and len(traj) == len(idx) == len(dist)):
        raise ValueError('LUT shape mismatch')
    if np.any(traj < 0) or np.any(traj >= shape[0]) or np.any(idx < 0) or np.any(idx >= shape[1]):
        raise ValueError('LUT indices outside trajectory tensor')
    if not np.isfinite(dist).all() or np.any(dist < 0) or np.any(dist != np.floor(dist)):
        raise ValueError('Invalid BFS distance')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def load_trajectories(directory):
    import cloudpickle
    ts, ls = [], []
    for p in sorted(Path(directory).glob('*trajectories.pkl')):
        with p.open('rb') as f:
            x = cloudpickle.load(f)
        with p.with_name(p.name.replace('trajectories', 'lens')).open('rb') as f:
            lengths = cloudpickle.load(f)
        x = np.asarray(x, dtype=np.uint8)
        ts.append(x.reshape(x.shape[0], x.shape[1], 144))
        ls.append(np.asarray(lengths, dtype=np.int64))
    if not ts:
        raise ValueError(f'No trajectory shards in {directory}')
    return (ts[0] if len(ts) == 1 else np.concatenate(ts)), np.concatenate(ls)


def clean_lut(T, path):
    with np.load(path) as z:
        tr, ix, ds = z['traj'], z['idx'], z['dist']
    validate_lut(tr, ix, ds, T.shape)
    pair_ids = tr.astype(np.int64) * T.shape[1] + ix
    if len(np.unique(pair_ids)) != len(pair_ids):
        raise ValueError('Duplicate LUT trajectory/index keys; check shard numbering')
    keep = np.zeros(len(ds), dtype=bool)
    for lo in range(0, len(ds), 65536):
        sl = slice(lo, lo + 65536)
        states = T[tr[sl], ix[sl]]
        keep[sl] = valid_rows(states, T[tr[sl], 0])
        solved = (np.isin(states, (3, 4)) == np.isin(states, (2, 3, 6))).all(1)
        mismatch = keep[sl] & ((ds[sl] == 0) != solved)
        if mismatch.any():
            row = lo + int(np.flatnonzero(mismatch)[0])
            raise ValueError(f'LUT zero distance iff solved invariant failed: '
                             f'traj={tr[row]} idx={ix[row]} distance={ds[row]}')
    if not keep.any():
        raise ValueError('No valid labeled states')
    print(json.dumps({'lut': str(path), 'total': len(ds), 'valid': int(keep.sum()),
                      'invalid': int((~keep).sum()), 'max_dist': float(ds[keep].max())}), flush=True)
    return tr[keep], ix[keep], ds[keep].astype(np.float32), int((~keep).sum())



def source_hashes(root, solver_path):
    root, solver_path = Path(root), Path(solver_path)
    sources = {str(f.relative_to(root)): sha256(f)
               for f in sorted((root/'raw').rglob('*')) if f.is_file()}
    sources[str(solver_path.relative_to(root))] = sha256(solver_path)
    return sources


def audit_lut(T, tr, ix, ds, dist_fn=None, seed=20260905, maxnodes=2000000):
    """Independently verify >=8 positive labels on distinct D4 board geometries.

    At most 32 BFS calls, each capped at two million states by default. A cap miss
    is unverified, never accepted as equality; any distance mismatch fails closed.
    """
    if dist_fn is None:
        try:
            from .sokoban_bfs import dist_of as dist_fn
        except ImportError:
            from sokoban_bfs import dist_of as dist_fn
    required, max_attempts = 8, 32
    positive = np.flatnonzero(ds > 0)
    # Oversample candidate rows so duplicate geometries do not consume BFS calls.
    rng = np.random.default_rng(seed)
    rows = rng.choice(positive, min(len(positive), max_attempts * 16), replace=False)
    attempts, seen, samples, cache = 0, set(), [], {}
    for row in rows:
        if attempts >= max_attempts or len(samples) >= required:
            break
        trajectory, index = int(tr[row]), int(ix[row])
        key = board_keys(T[trajectory, 0][None])[0]
        if key in seen:
            continue
        seen.add(key)
        attempts += 1
        cache.clear()  # hold at most one independent oracle board at a time
        print('LABEL_AUDIT_CHECK ' + json.dumps(dict(attempt=attempts,
              verified=len(samples), traj=trajectory, idx=index,
              expected=float(ds[row]), maxnodes=maxnodes)), flush=True)
        recomputed = float(dist_fn(T[trajectory, index], cache, maxnodes))
        if recomputed < 0:
            continue
        if recomputed != float(ds[row]):
            raise ValueError(f'Independent BFS distance mismatch: traj={trajectory} '
                             f'idx={index} label={ds[row]} recomputed={recomputed}')
        samples.append(dict(traj=trajectory, idx=index, board_key=key,
                            distance=float(ds[row]), recomputed=recomputed))
    result = dict(verified=len(samples), attempts=attempts, required=required,
                  max_attempts=max_attempts, maxnodes=maxnodes, seed=seed, samples=samples)
    print('LABEL_AUDIT ' + json.dumps(result), flush=True)
    if len(samples) < required:
        raise ValueError(f'Independent BFS audit verified {len(samples)} labels after '
                         f'{attempts} attempts; require >= {required} distinct boards')
    return result


def write_bank(path, T, tr, ix, ds, selection, cap=0, seed=20260905):
    rows = np.flatnonzero(selection)
    if cap and len(rows) > cap:
        rows = np.sort(np.random.default_rng(seed).choice(rows, cap, replace=False))
    if not len(rows):
        raise ValueError(f'Empty bank: {path}')
    if ds[rows].max() >= 150:
        raise ValueError('BFS distance exceeds supervised 150-bin support; do not silently clip')
    np.savez(path, states=T[tr[rows], ix[rows]], goals=solved_goals(T[tr[rows], 0]),
             dist=ds[rows], traj=tr[rows], idx=ix[rows])
    return {'path': str(path), 'n': len(rows), 'sha256': sha256(path),
            'boards': len(np.unique(tr[rows])), 'distance_mean': float(ds[rows].mean())}



def write_smoke_bank(path, validation_bank):
    """A byte-identical validation copy; never expose official test during smoke."""
    import shutil
    shutil.copyfile(validation_bank['path'], path)
    checksum = sha256(path)
    if checksum != validation_bank['sha256']:
        raise ValueError('Smoke validation-copy checksum mismatch')
    return {**validation_bank, 'path': str(path), 'sha256': checksum,
            'source_bank': 'val', 'purpose': 'smoke_only_validation_copy'}


def write_gap_bank(path, T, L, allowed, n, seed=20260905):
    """Uniform trajectories, i,j in recorded [0,lens-1], including self pairs.

    Reject malformed states at either end. These gap labels are intentionally
    noisy upper bounds, not exact distances; paired retrains share this same bank.
    """
    rng = np.random.default_rng(seed)
    candidates = np.flatnonzero(allowed & (L > 0) & (L <= T.shape[1]))
    if not len(candidates):
        raise ValueError('No eligible gap trajectories')
    states, goals, dist = [], [], []
    total = 0
    for _ in range(10000):
        if total >= n:
            break
        tr = rng.choice(candidates, min(65536, n - total))
        i = (rng.random(len(tr)) * L[tr]).astype(np.int64)
        j = (rng.random(len(tr)) * L[tr]).astype(np.int64)
        lo, hi = np.minimum(i, j), np.maximum(i, j)
        s, g = T[tr, lo], T[tr, hi]
        keep = valid_rows(s, T[tr, 0]) & valid_rows(g, T[tr, 0]) & ((hi-lo) < 150)
        states.append(s[keep]); goals.append(g[keep]); dist.append((hi-lo)[keep])
        total += int(keep.sum())
    if total < n:
        raise ValueError('Gap-bank rejection limit reached')
    np.savez(path, states=np.concatenate(states), goals=np.concatenate(goals),
             dist=np.concatenate(dist).astype(np.float32))
    return {'path': str(path), 'n': total, 'sha256': sha256(path), 'targets': 'trajectory_gap'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--eval-pairs', type=int, default=20000)
    p.add_argument('--gap-pairs', type=int, default=2000000)
    a = p.parse_args()
    root = Path(a.root)
    out = root / 'banks'
    out.mkdir(exist_ok=True)
    import cloudpickle
    train, lens = load_trajectories(root / 'raw/train')
    test, _ = load_trajectories(root / 'raw/test')
    solver_path = root / 'crtr/training_datasets/sokoban_eval_boards/eval_boards.pkl'
    with solver_path.open('rb') as f:
        boards = np.asarray(cloudpickle.load(f))
    if boards.ndim == 4 and boards.shape[-1] == 7:
        boards = boards.argmax(-1)
    testkeys = set(board_keys(test[:, 0]))
    solverkeys = set(board_keys(boards))
    keys = board_keys(train[:, 0])
    tm, vm, excluded = split_boards(keys, testkeys | solverkeys)
    tr, ix, ds, rejected = clean_lut(train, root / 'raw/labels/bfs_lut_all.npz')
    et, ei, ed, rejected_test = clean_lut(test, root/'raw/labels/bfs_lut_test.npz')
    audits = {'train': audit_lut(train, tr, ix, ds),
              'test': audit_lut(test, et, ei, ed)}
    result = {'protocol': 'synthetic_solved_exact_BFS_board_D4_split_v1',
              'train_trajectories': len(train), 'test_trajectories': len(test),
              'train_geometry_count': len(set(keys)), 'test_geometry_count': len(testkeys),
              'solver_geometry_count': len(solverkeys),
              'excluded_train_trajectories': int(excluded.sum()),
              'validation_fraction': .05, 'rejected_train_labels': rejected,
              'sources': source_hashes(root, solver_path),
              'label_audits': audits,
              'banks': {}}
    result['banks']['train'] = write_bank(out/'train.npz', train, tr, ix, ds, tm[tr])
    result['banks']['val'] = write_bank(out/'val.npz', train, tr, ix, ds, vm[tr], a.eval_pairs)
    result['banks']['smoke_test'] = write_smoke_bank(out/'smoke_test.npz', result['banks']['val'])
    result['rejected_test_labels'] = rejected_test
    result['banks']['test'] = write_bank(out/'test.npz', test, et, ei, ed, np.ones(len(ed), bool), a.eval_pairs)
    if a.gap_pairs:
        result['banks']['train_gap'] = write_gap_bank(out/'train_gap.npz', train, lens, tm, a.gap_pairs)
        result['banks']['val_gap'] = write_gap_bank(out/'val_gap.npz', train, lens, vm, a.eval_pairs)
    (out/'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    (out/'READY').write_text('Complete: see manifest.json\n')
    print('RESULT ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

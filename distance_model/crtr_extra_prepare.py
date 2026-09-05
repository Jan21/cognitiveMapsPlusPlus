"""Freeze reconstructed CRTR task banks; exact oracles are label-only.

Lights pairs are random positions along 49-press trajectories starting all-off.
Digit boards follow the sourced puzzlegen IID/reachability-rejection distribution,
NOT the unpublished CRTR path-first sampler. Positions are uniform among cells
that can reach bottom-right; every pair uses that terminal goal. Digit has no
temporal-gap member: supervision is exact directed BFS, including off-path cells.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

if __package__:
    from .crtr_extra_envs import (lights_toggle_matrix, lights_distances,
        generate_digit_board, digit_distances, digit_encode, DIGIT_GENERATOR,
        DIGIT_SOURCE_REVISION)
else:
    from crtr_extra_envs import (lights_toggle_matrix, lights_distances,
        generate_digit_board, digit_distances, digit_encode, DIGIT_GENERATOR,
        DIGIT_SOURCE_REVISION)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def layout_key(board):
    """D4 canonical digest: rotated/reflected layouts cannot cross splits."""
    variants = [np.rot90(view, k).tobytes() for view in (board, board[:, ::-1])
                for k in range(4)]
    return hashlib.sha256(min(variants)).hexdigest()


def pair_keys(bank):
    # Lights dynamics are reversible: swapping endpoints also counts as overlap.
    return {(min(a.tobytes(), b.tobytes()), max(a.tobytes(), b.tobytes()))
            for a, b in zip(bank['states'], bank['goals'])}


def lights_pairs(count, seed, size=7, blocked=None):
    if not 1 <= size <= 7 or count < 1:
        raise ValueError('packed Lights sampler requires size1..7 and positive count')
    rng = np.random.default_rng(seed)
    blocked = set() if blocked is None else blocked
    cells = size * size
    weights = np.left_shift(np.uint64(1), np.arange(cells, dtype=np.uint64))
    masks = (lights_toggle_matrix(size).astype(np.uint64) * weights[:, None]).sum(0)
    output = dict(states=np.empty((count, size, size), np.uint8),
                  goals=np.empty((count, size, size), np.uint8),
                  dist=np.empty(count, np.float32), gap=np.empty(count, np.float32))
    filled = 0
    attempts = 0
    while filled < count:
        batch = min(8192, count - filled)
        actions = rng.integers(0, cells, size=(batch, cells))
        trajectory = np.zeros((batch, cells + 1), dtype=np.uint64)
        trajectory[:, 1:] = np.bitwise_xor.accumulate(masks[actions], axis=1)
        first = rng.integers(0, cells + 1, size=batch)
        last = rng.integers(0, cells + 1, size=batch)
        starts = ((trajectory[np.arange(batch), first, None] & weights) != 0).astype(np.uint8).reshape(-1, size, size)
        goals = ((trajectory[np.arange(batch), last, None] & weights) != 0).astype(np.uint8).reshape(-1, size, size)
        keep = np.ones(batch, dtype=bool)
        if blocked:
            keep = np.asarray([(min(a.tobytes(), b.tobytes()), max(a.tobytes(), b.tobytes())) not in blocked
                               for a, b in zip(starts, goals)], dtype=bool)
        starts, goals = starts[keep], goals[keep]
        end = filled + len(starts)
        output['states'][filled:end], output['goals'][filled:end] = starts, goals
        output['dist'][filled:end] = lights_distances(starts, goals, size=size)
        output['gap'][filled:end] = np.abs(first - last)[keep]
        filled = end
        attempts += batch
        if attempts > max(100000, count * 30):
            raise RuntimeError('heldout rejection exhausted Lights sample budget')
    return output


def digit_pairs(boards, pairs_per_board, seed, size=20, blocked=None):
    if min(boards, pairs_per_board) < 1:
        raise ValueError('positive board and pair counts required')
    rng = np.random.default_rng(seed)
    blocked = set() if blocked is None else set(blocked)
    states = np.empty((boards * pairs_per_board, size, size), np.uint8)
    goals = np.empty_like(states)
    distances = np.empty(boards * pairs_per_board, np.float32)
    case_states = np.empty((boards, size, size), np.uint8)
    case_goals = np.empty_like(case_states)
    case_dist = np.empty(boards, np.float32)
    keys = []
    for index in range(boards):
        for _ in range(10000):
            board, _path = generate_digit_board(rng, size=size)
            key = layout_key(board)
            if key not in blocked:
                break
        else:
            raise RuntimeError('failed to draw a fresh Digit layout')
        blocked.add(key)
        keys.append(key)
        exact = digit_distances(board)
        reachable = np.flatnonzero(exact >= 0)
        positions = rng.choice(reachable, size=pairs_per_board, replace=True)
        goal = digit_encode(board, board.size - 1)
        start = index * pairs_per_board
        for offset, position in enumerate(positions):
            states[start + offset] = digit_encode(board, int(position))
        goals[start:start + pairs_per_board] = goal
        distances[start:start + pairs_per_board] = exact[positions]
        case_states[index], case_goals[index] = digit_encode(board, 0), goal
        case_dist[index] = exact[0]
        if (index + 1) % 1000 == 0:
            print(f'DIGIT boards={index + 1}/{boards}', flush=True)
    return dict(states=states, goals=goals, dist=distances), dict(
        states=case_states, goals=case_goals, dist=case_dist), keys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', choices=('lights', 'digit'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train-pairs', type=int, default=1000000)
    parser.add_argument('--train-boards', type=int, default=25000)
    parser.add_argument('--eval-boards', type=int, default=1000)
    parser.add_argument('--eval-pairs', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20260905)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    metadata = dict(environment=args.env, seed=args.seed, reconstruction=True,
                    exact_labels=True, oracles_used_at_inference=False)
    if args.env == 'lights':
        metadata.update(grid_size=7, grid_size_is_unverified_paper_assumption=True,
                        trajectory_actions=49, trajectory_initial_state='all_off',
                        pair_distribution='two_independent_uniform_trajectory_positions',
                        split_exclusion='identical_unordered_endpoint_pair',
                        search_distribution='uniform_binary_board_to_all_off')
        rng = np.random.default_rng(args.seed + 10)
        states = rng.integers(0, 2, (args.eval_boards, 7, 7), dtype=np.uint8)
        cases = dict(states=states, goals=np.zeros_like(states),
                     dist=lights_distances(states).astype(np.float32))
        test = lights_pairs(args.eval_pairs, args.seed + 2, blocked=pair_keys(cases))
        validation = lights_pairs(args.eval_pairs, args.seed + 1,
                                  blocked=pair_keys(test) | pair_keys(cases))
        training = lights_pairs(args.train_pairs, args.seed,
                                blocked=pair_keys(validation) | pair_keys(test) | pair_keys(cases))
        assert not pair_keys(training) & (pair_keys(validation) | pair_keys(test) | pair_keys(cases))
        assert not pair_keys(validation) & pair_keys(test)
    else:
        if args.train_pairs % args.train_boards or args.eval_pairs % args.eval_boards:
            parser.error('pair counts must divide evenly by board counts')
        metadata.update(grid_size=20, generator=DIGIT_GENERATOR,
                        reference_revision=DIGIT_SOURCE_REVISION,
                        crtr_path_first_distribution_reproduced=False,
                        pair_distribution='uniform_reachable_position_to_bottom_right',
                        split_exclusion='D4_canonical_board_hash',
                        search_distribution='heldout_solvable_iid_board_top_left_to_bottom_right')
        test, cases, test_keys = digit_pairs(args.eval_boards, args.eval_pairs // args.eval_boards,
                                           args.seed + 2)
        validation, _, val_keys = digit_pairs(args.eval_boards, args.eval_pairs // args.eval_boards,
                                             args.seed + 1, blocked=set(test_keys))
        training, _, train_keys = digit_pairs(args.train_boards, args.train_pairs // args.train_boards,
                                             args.seed, blocked=set(test_keys) | set(val_keys))
        assert not set(train_keys) & (set(val_keys) | set(test_keys))
        assert not set(val_keys) & set(test_keys)
        (args.out / 'layout_keys.json').write_text(json.dumps(
            dict(train=train_keys, validation=val_keys, test=test_keys)))
    manifest = dict(metadata=metadata, files={}, elapsed_seconds=None)
    for name, bank in (('train', training), ('validation', validation), ('test', test), ('cases', cases)):
        path = args.out / (name + '.npz')
        np.savez_compressed(path, **bank, metadata_json=json.dumps(metadata, sort_keys=True))
        manifest['files'][name] = dict(path=str(path), sha256=sha256(path), rows=len(bank['dist']),
            distance_min=float(bank['dist'].min()), distance_max=float(bank['dist'].max()),
            distance_mean=float(bank['dist'].mean()))
        print('SAVED ' + json.dumps(manifest['files'][name]), flush=True)
    # GPU smokes must never consume final-test data; separate path required by trainer.
    shutil.copyfile(args.out / 'validation.npz', args.out / 'smoke_test.npz')
    # Solver smokes use heldout validation examples, not the final search cases.
    smoke = {key: value[:2] for key, value in validation.items()}
    np.savez_compressed(args.out / 'smoke_cases.npz', **smoke)
    manifest['elapsed_seconds'] = time.time() - started
    manifest['source_sha256'] = {name: sha256(Path(__file__).with_name(name))
                               for name in ('crtr_extra_prepare.py', 'crtr_extra_envs.py')}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print('COMPLETE ' + json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()

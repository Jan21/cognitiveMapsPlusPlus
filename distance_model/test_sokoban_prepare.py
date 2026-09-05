import numpy as np
import pytest


def board():
    x = np.ones((12, 12), dtype=np.uint8)
    x[[0, -1], :] = 0
    x[:, [0, -1]] = 0
    x[2, 3] = 2
    x[3, 3] = 4
    x[4, 3] = 5
    return x.reshape(144)


def test_goal_reconstruction_and_symmetry_identity():
    from distance_model.sokoban_prepare import solved_goals, board_keys
    x = board()[None]
    g = solved_goals(x)
    assert g[0, 27] == 3
    assert np.count_nonzero(g == 3) == 1
    assert not np.isin(g, [2, 4, 5, 6]).any()
    rot = np.rot90(x.reshape(12, 12)).reshape(1, 144)
    assert board_keys(x) == board_keys(rot)


def test_reject_padding_and_mismatched_layout():
    from distance_model.sokoban_prepare import valid_rows
    x = board()
    bad = x.copy()
    bad[0] = 1
    states = np.stack([x, np.zeros(144, np.uint8), bad])
    goals0 = np.repeat(x[None], 3, 0)
    assert valid_rows(states, goals0).tolist() == [True, False, False]


def test_split_is_board_disjoint_and_excludes_test_geometry():
    from distance_model.sokoban_prepare import board_keys, split_boards
    x = board()
    frames = np.stack([x, np.rot90(x.reshape(12, 12)).reshape(144), x.copy()])
    frames[2, 25] = 0
    keys = board_keys(frames)
    train, val, excluded = split_boards(keys, {keys[2]}, validation_fraction=.5)
    assert excluded.tolist() == [False, False, True]
    assert train[0] == train[1] and val[0] == val[1]
    assert not np.any(train & val)
    assert not train[2] and not val[2]


def test_invalid_lut_indices_fail_closed():
    from distance_model.sokoban_prepare import validate_lut
    with pytest.raises(ValueError, match='indices'):
        validate_lut(np.array([2]), np.array([0]), np.array([1.]), (2, 5, 144))
    with pytest.raises(ValueError, match='distance'):
        validate_lut(np.array([0]), np.array([0]), np.array([-1.]), (2, 5, 144))


@pytest.mark.parametrize('already_solved,label', [(False, 0.), (True, 1.)])
def test_zero_distance_must_match_solved_state(tmp_path, already_solved, label):
    from distance_model.sokoban_prepare import clean_lut, solved_goals
    frame = board()
    state = solved_goals(frame[None])[0] if already_solved else frame
    trajectory = np.stack([frame, state])[None]
    path = tmp_path / 'labels.npz'
    np.savez(path, traj=np.array([0]), idx=np.array([1]), dist=np.array([label]))
    with pytest.raises(ValueError, match='zero.*solved|solved.*zero'):
        clean_lut(trajectory, path)


def audit_examples(n=8):
    frames = []
    for index in range(n):
        x = board().reshape(12, 12).copy()
        for bit in range(6):
            if (index + 1) & (1 << bit):
                x[8, bit + 2] = 0
        frames.append(x.reshape(144))
    return np.stack(frames)[:, None], np.arange(n), np.zeros(n, dtype=np.int64), np.ones(n)


def actual_bfs_oracle():
    import importlib.util
    from pathlib import Path
    source = Path(__file__).resolve().parents[2] / 'delpi-lab/crtr_bench/sokoban_bfs.py'
    if not source.exists():
        source = Path(__file__).with_name('sokoban_bfs.py')
    spec = importlib.util.spec_from_file_location('independent_sokoban_test_oracle', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.dist_of


def test_independent_audit_verifies_eight_distinct_boards_deterministically():
    import distance_model.sokoban_prepare as preparation
    assert hasattr(preparation, 'audit_lut'), 'Independent label audit must exist'
    T, tr, ix, ds = audit_examples()
    first = preparation.audit_lut(T, tr, ix, ds, dist_fn=actual_bfs_oracle())
    second = preparation.audit_lut(T, tr, ix, ds, dist_fn=actual_bfs_oracle())
    assert first == second
    assert first['verified'] == first['attempts'] == 8
    assert len({row['board_key'] for row in first['samples']}) == 8
    assert all(row['distance'] == row['recomputed'] == 1 for row in first['samples'])


def test_independent_audit_rejects_wrong_positive_labels_without_relabeling():
    import distance_model.sokoban_prepare as preparation
    assert hasattr(preparation, 'audit_lut'), 'Independent label audit must exist'
    T, tr, ix, ds = audit_examples()
    ds[:] = 2
    with pytest.raises(ValueError, match='distance mismatch'):
        preparation.audit_lut(T, tr, ix, ds, dist_fn=actual_bfs_oracle())
    assert np.all(ds == 2)


def test_independent_audit_fails_when_cap_prevents_required_verifications():
    import distance_model.sokoban_prepare as preparation
    assert hasattr(preparation, 'audit_lut'), 'Independent label audit must exist'
    T, tr, ix, ds = audit_examples(40)
    with pytest.raises(ValueError, match='verified 0.*32 attempts'):
        preparation.audit_lut(T, tr, ix, ds, dist_fn=actual_bfs_oracle(), maxnodes=1)


def test_sources_include_solver_board_checksum(tmp_path):
    import distance_model.sokoban_prepare as preparation
    assert hasattr(preparation, 'source_hashes'), 'Solver exclusion source must be recorded'
    (tmp_path / 'raw').mkdir()
    (tmp_path / 'raw' / 'labels').write_bytes(b'labels')
    solver = tmp_path / 'crtr' / 'eval_boards.pkl'
    solver.parent.mkdir()
    solver.write_bytes(b'solver boards')
    hashes = preparation.source_hashes(tmp_path, solver)
    assert hashes['raw/labels'] == preparation.sha256(tmp_path / 'raw' / 'labels')
    assert hashes['crtr/eval_boards.pkl'] == preparation.sha256(solver)


def test_smoke_test_bank_is_byte_identical_validation_copy(tmp_path):
    import distance_model.sokoban_prepare as preparation
    assert hasattr(preparation, 'write_smoke_bank'), 'Smoke test must reuse validation data'
    val = tmp_path / 'val.npz'
    np.savez(val, states=board()[None], goals=board()[None], dist=np.array([1.]))
    metadata = {'path': str(val), 'n': 1, 'sha256': preparation.sha256(val)}
    smoke = tmp_path / 'smoke_test.npz'
    result = preparation.write_smoke_bank(smoke, metadata)
    assert smoke.read_bytes() == val.read_bytes()
    assert result['sha256'] == metadata['sha256']
    assert result['path'] == str(smoke)
    assert result['source_bank'] == 'val'
    assert result['purpose'] == 'smoke_only_validation_copy'

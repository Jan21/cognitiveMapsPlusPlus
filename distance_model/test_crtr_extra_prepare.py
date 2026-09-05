import numpy as np


def test_lights_bank_is_deterministic_and_truth_never_exceeds_gap():
    from distance_model.crtr_extra_prepare import lights_pairs
    a = lights_pairs(48, seed=11, size=3)
    b = lights_pairs(48, seed=11, size=3)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert a['states'].shape == (48, 3, 3)
    assert a['dist'].dtype == np.float32
    assert np.all(a['dist'] <= a['gap'])
    assert np.all((a['dist'] == 0) == np.all(a['states'] == a['goals'], axis=(1, 2)))


def test_digit_banks_preserve_layout_and_replay_exact_goal_distance():
    from distance_model.crtr_extra_prepare import digit_pairs
    from distance_model.crtr_extra_envs import digit_distances
    bank, cases, keys = digit_pairs(3, 4, seed=12, size=8)
    assert len(bank['states']) == 12 and len(cases['states']) == 3
    assert len(keys) == 3
    for s, g, d in zip(bank['states'], bank['goals'], bank['dist']):
        board = (s - 1) % 6 + 1
        np.testing.assert_array_equal(board, (g-1) % 6 + 1)
        pos = int(np.flatnonzero(s > 6)[0])
        goal = int(np.flatnonzero(g > 6)[0])
        assert d == digit_distances(board, goal)[pos]


def test_digit_heldout_layouts_are_rejected_during_generation():
    from distance_model.crtr_extra_prepare import digit_pairs
    _, _, keys = digit_pairs(2, 2, seed=13, size=8)
    _, _, fresh_keys = digit_pairs(2, 2, seed=13, size=8, blocked=set(keys))
    assert set(keys).isdisjoint(fresh_keys)


def test_main_manifests_exact_labels_and_validation_only_smoke_data(tmp_path):
    import hashlib
    import json
    from collections import deque
    from distance_model import crtr_extra_prepare as prepare
    from distance_model.crtr_extra_envs import lights_distances

    def read_bank(path):
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def independent_digit_distance(state, goal):
        board = ((state.astype(np.int16) - 1) % 6 + 1)
        np.testing.assert_array_equal(board, (goal.astype(np.int16) - 1) % 6 + 1)
        assert np.count_nonzero(state > 6) == np.count_nonzero(goal > 6) == 1
        start = int(np.flatnonzero(state > 6)[0])
        target = int(np.flatnonzero(goal > 6)[0])
        assert target == board.size - 1
        queue = deque([(start, 0)])
        seen = {start}
        size = board.shape[0]
        while queue:
            position, distance = queue.popleft()
            if position == target:
                return distance
            row, column = divmod(position, size)
            jump = int(board[row, column])
            for rr, cc in ((row - jump, column), (row + jump, column),
                           (row, column - jump), (row, column + jump)):
                if 0 <= rr < size and 0 <= cc < size:
                    successor = rr * size + cc
                    if successor not in seen:
                        seen.add(successor)
                        queue.append((successor, distance + 1))
        raise AssertionError("saved Digit state cannot reach its labelled goal")

    for env in ("lights", "digit"):
        directory = tmp_path / env
        train_rows, evaluation_rows, case_rows = ((64, 16, 4) if env == "lights" else (16, 8, 2))
        args = ["--env", env, "--out", str(directory),
                "--train-pairs", str(train_rows), "--eval-pairs", str(evaluation_rows),
                "--eval-boards", str(case_rows), "--seed", "153"]
        if env == "digit":
            args += ["--train-boards", "4"]
        prepare.main(args)
        manifest = json.loads((directory / "manifest.json").read_text())
        metadata = manifest["metadata"]
        assert metadata["environment"] == env
        assert metadata["reconstruction"] is True
        assert metadata["exact_labels"] is True
        assert metadata["oracles_used_at_inference"] is False
        assert metadata["grid_size"] == (7 if env == "lights" else 20)
        assert manifest["elapsed_seconds"] >= 0
        assert set(manifest["files"]) == {"train", "validation", "test", "cases"}
        for source, digest in manifest["source_sha256"].items():
            assert digest == sha(prepare.Path(prepare.__file__).with_name(source))
        banks = {}
        for name, rows in (("train", train_rows), ("validation", evaluation_rows),
                           ("test", evaluation_rows), ("cases", case_rows)):
            path = directory / (name + ".npz")
            entry = manifest["files"][name]
            assert prepare.Path(entry["path"]).resolve() == path.resolve()
            assert entry["sha256"] == sha(path)
            assert entry["rows"] == rows
            bank = banks[name] = read_bank(path)
            assert bank["states"].dtype == bank["goals"].dtype == np.uint8
            assert bank["dist"].dtype == np.float32
            assert bank["states"].shape == bank["goals"].shape == (
                rows, metadata["grid_size"], metadata["grid_size"])
            assert bank["dist"].shape == (rows,)
            assert np.isfinite(bank["dist"]).all() and np.all(bank["dist"] >= 0)
            assert np.all(bank["dist"] == np.floor(bank["dist"]))
            assert entry["distance_min"] == float(bank["dist"].min())
            assert entry["distance_max"] == float(bank["dist"].max())
            assert entry["distance_mean"] == float(bank["dist"].mean())
            assert json.loads(bank["metadata_json"].item()) == metadata
            if env == "lights":
                np.testing.assert_array_equal(
                    bank["dist"], lights_distances(bank["states"], bank["goals"], size=7))
                if "gap" in bank:
                    assert np.all(bank["dist"] <= bank["gap"])
                    assert np.all((bank["gap"] - bank["dist"]) % 2 == 0)
            else:
                assert "gap" not in bank
                for state, goal, distance in zip(bank["states"], bank["goals"], bank["dist"]):
                    assert distance == independent_digit_distance(state, goal)

        validation_hash = manifest["files"]["validation"]["sha256"]
        final_test_hash = manifest["files"]["test"]["sha256"]
        assert sha(directory / "smoke_test.npz") == validation_hash
        assert validation_hash != final_test_hash
        smoke = read_bank(directory / "smoke_cases.npz")
        assert len(smoke["dist"]) == 2
        for key in ("states", "goals", "dist"):
            np.testing.assert_array_equal(smoke[key], banks["validation"][key][:2])

        if env == "lights":
            keys = {name: prepare.pair_keys(bank) for name, bank in banks.items()}
            assert keys["train"].isdisjoint(keys["validation"] | keys["test"] | keys["cases"])
            assert keys["validation"].isdisjoint(keys["test"] | keys["cases"])
            assert keys["test"].isdisjoint(keys["cases"])
            assert not banks["cases"]["goals"].any()
            assert metadata["grid_size_is_unverified_paper_assumption"] is True
        else:
            keys = json.loads((directory / "layout_keys.json").read_text())
            assert len(keys["train"]) == 4
            assert len(keys["validation"]) == len(keys["test"]) == 2
            assert set(keys["train"]).isdisjoint(keys["validation"] + keys["test"])
            assert set(keys["validation"]).isdisjoint(keys["test"])
            for name in ("train", "validation", "test"):
                decoded = [((state.astype(np.int16) - 1) % 6 + 1).astype(np.uint8)
                           for state in banks[name]["states"]]
                assert set(keys[name]) == {prepare.layout_key(board) for board in decoded}
            for state, goal in zip(banks["cases"]["states"], banks["cases"]["goals"]):
                assert int(np.flatnonzero(state > 6)[0]) == 0
                assert int(np.flatnonzero(goal > 6)[0]) == 399
            case_keys = {prepare.layout_key(((state.astype(np.int16) - 1) % 6 + 1).astype(np.uint8))
                         for state in banks["cases"]["states"]}
            assert case_keys == set(keys["test"])
            assert metadata["crtr_path_first_distribution_reproduced"] is False


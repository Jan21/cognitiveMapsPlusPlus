"""Exact environment tests, independent of neural networks and GPUs."""

from collections import deque
import importlib.util
import unittest

import numpy as np


def lights_bfs(size):
    masks = []
    for row in range(size):
        for col in range(size):
            mask = 0
            for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = row + dr, col + dc
                if 0 <= rr < size and 0 <= cc < size:
                    mask |= 1 << (rr * size + cc)
            masks.append(mask)
    distances, queue = {0: 0}, deque([0])
    while queue:
        state = queue.popleft()
        for mask in masks:
            successor = state ^ mask
            if successor not in distances:
                distances[successor] = distances[state] + 1
                queue.append(successor)
    return distances


def bits_to_boards(values, size):
    return ((np.asarray(values, dtype=np.uint64)[:, None] >> np.arange(size * size, dtype=np.uint64)) & 1).astype(np.uint8).reshape(-1, size, size)


class ExtraEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("distance_model.crtr_extra_envs"))
        from distance_model import crtr_extra_envs as module
        self.module = module

    def test_all_3x3_lights_distances_match_independent_bfs(self):
        oracle = lights_bfs(3)
        states = bits_to_boards(np.arange(512), 3)
        actual = self.module.lights_distances(states, size=3)
        expected = np.array([oracle.get(state, -1) for state in range(512)])
        np.testing.assert_array_equal(actual, expected)
        goals = np.roll(states, 17, axis=0)
        xor_ids = np.arange(512) ^ np.roll(np.arange(512), 17)
        np.testing.assert_array_equal(self.module.lights_distances(states, goals, size=3),
                                      [oracle.get(int(value), -1) for value in xor_ids])

    def test_rank_deficient_lights_minimizes_nullspace_and_certifies_unreachable(self):
        oracle = lights_bfs(4)
        unreachable = next(state for state in range(65536) if state not in oracle)
        values = list(oracle)[::37] + [unreachable]
        actual = self.module.lights_distances(bits_to_boards(values, 4), size=4)
        np.testing.assert_array_equal(actual, [oracle.get(value, -1) for value in values])
        self.assertEqual(actual[-1], -1)

    def test_lights_actions_match_matrix_are_reversible_and_validate_bounds(self):
        states = np.zeros((3, 3, 3), dtype=np.uint8)
        actions = np.array([0, 4, 8])
        updated = self.module.lights_apply(states, actions)
        matrix = self.module.lights_toggle_matrix(3)
        np.testing.assert_array_equal(updated.reshape(3, 9), matrix[:, actions].T)
        np.testing.assert_array_equal(self.module.lights_apply(updated, actions), states)
        self.assertEqual(int(states.sum()), 0)
        with self.assertRaises(ValueError):
            self.module.lights_apply(states, np.array([-1, 4, 8]))

    def test_digit_edges_are_directed_boundary_checked_and_goal_has_zero_distance(self):
        board = np.array([[1, 2, 1], [3, 1, 1], [1, 1, 1]], dtype=np.uint8)
        self.assertIn(1, self.module.digit_successors(board, 0))
        self.assertNotIn(0, self.module.digit_successors(board, 1))
        self.assertEqual(self.module.digit_successors(board, 3), [])
        distances = self.module.digit_distances(board)
        self.assertEqual(distances[8], 0)
        self.assertEqual(distances[3], -1)
        for start in range(9):
            queue, seen, oracle = deque([(start, 0)]), {start}, -1
            while queue:
                position, depth = queue.popleft()
                if position == 8:
                    oracle = depth
                    break
                row, col = divmod(position, 3)
                jump = int(board[row, col])
                for dr, dc in ((-jump, 0), (jump, 0), (0, -jump), (0, jump)):
                    rr, cc = row + dr, col + dc
                    target = rr * 3 + cc
                    if 0 <= rr < 3 and 0 <= cc < 3 and target not in seen:
                        seen.add(target)
                        queue.append((target, depth + 1))
            self.assertEqual(distances[start], oracle)
        self.assertEqual(self.module.digit_distances(board, goal=0)[0], 0)
        with self.assertRaises(ValueError):
            self.module.digit_successors(board, 9)

    def test_digit_generator_replays_and_encoding_preserves_digits(self):
        for seed in range(10):
            board, path = self.module.generate_digit_board(np.random.default_rng(seed), size=20)
            duplicate, same_path = self.module.generate_digit_board(np.random.default_rng(seed), size=20)
            np.testing.assert_array_equal(board, duplicate)
            self.assertEqual(path, same_path)
            self.assertEqual(path[0], 0)
            self.assertEqual(path[-1], 399)
            self.assertEqual(len(path), len(set(path)))
            self.assertTrue(np.all((board >= 1) & (board <= 6)))
            for start, finish in zip(path, path[1:]):
                self.assertIn(finish, self.module.digit_successors(board, start))
            distance = self.module.digit_distances(board)[0]
            self.assertGreaterEqual(distance, 0)
            self.assertLessEqual(distance, len(path) - 1)
            encoded = self.module.digit_encode(board, path[1])
            self.assertEqual(encoded.dtype, np.uint8)
            self.assertEqual(int((encoded > 6).sum()), 1)
            recovered = encoded.copy()
            recovered[recovered > 6] -= 6
            np.testing.assert_array_equal(recovered, board)


if __name__ == "__main__":
    unittest.main()

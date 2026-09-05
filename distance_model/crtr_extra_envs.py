"""Pure NumPy exact dynamics for Lights Out and Digit Jumper reconstructions.

Lights: pressing a cell toggles itself and its four orthogonal neighbors. Moves
commute and cancel in pairs, so shortest distance is the minimum Hamming weight
of a GF(2) solution; inconsistent systems are unreachable.

Digit Jumper: a cell's digit is the required jump length in a cardinal direction.
Out-of-bounds moves are omitted, since their reference self-loops cannot improve
shortest paths. The default goal is bottom-right; geometric successors remain
defined at a goal, while its distance is zero and goal-reaching rollouts stop.

Generator source: martius-lab/puzzlegen, puzzlegen/digit_jump.py,
https://github.com/martius-lab/puzzlegen/blob/main/puzzlegen/digit_jump.py
Original uses iid digit boards, rejection on start-to-goal reachability, and a
BFS solution. This implementation preserves that distribution using NumPy's
Generator, not the original Python Random seed stream. It does not claim the
unpublished CRTR appendix's alternative path-first generator distribution.
"""

from collections import deque
from functools import lru_cache
from numbers import Integral

import numpy as np


DIGIT_GENERATOR = "puzzlegen_iid_rejection_numpy_rng"
DIGIT_SOURCE_REVISION = "1a5ff909b80526fde6fb1045cfab9a42291d3c85"


def _positive_size(size):
    if not isinstance(size, Integral) or isinstance(size, bool) or size < 1:
        raise ValueError("size must be a positive integer")
    return int(size)


def lights_toggle_matrix(size=7):
    """Return uint8 A[cell, action], the square board's GF(2) toggle matrix."""
    size = _positive_size(size)
    matrix = np.zeros((size * size, size * size), dtype=np.uint8)
    for row in range(size):
        for col in range(size):
            action = row * size + col
            for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = row + dr, col + dc
                if 0 <= rr < size and 0 <= cc < size:
                    matrix[rr * size + cc, action] = 1
    return matrix


def _lights_boards(states, size, name="states"):
    states = np.asarray(states)
    if states.ndim != 3 or states.shape[1:] != (size, size):
        raise ValueError(f"{name} must have shape (B,{size},{size})")
    if states.dtype.kind not in "bui" or np.any((states != 0) & (states != 1)):
        raise ValueError(f"{name} must contain binary integer/bool values")
    return states.astype(np.uint8, copy=False)


def lights_apply(states, actions):
    """Apply one flat cell action per B,H,W board, returning a fresh uint8 batch."""
    states = np.asarray(states)
    if states.ndim != 3 or states.shape[1] != states.shape[2]:
        raise ValueError("states must have square shape (B,H,H)")
    size = _positive_size(states.shape[1])
    states = _lights_boards(states, size)
    actions = np.asarray(actions)
    if (actions.shape != (len(states),) or actions.dtype.kind not in "ui"
            or np.any(actions < 0) or np.any(actions >= size * size)):
        raise ValueError("actions must be B valid integer flat cell indices")
    result = states.copy()
    rows, cols = actions // size, actions % size
    indices = np.arange(len(states))
    for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
        # Convert unsigned indices before subtracting a direction at the border.
        rr, cc = rows.astype(np.int64) + dr, cols.astype(np.int64) + dc
        valid = (rr >= 0) & (rr < size) & (cc >= 0) & (cc < size)
        result[indices[valid], rr[valid], cc[valid]] ^= 1
    return result


@lru_cache(maxsize=8)
def _lights_factorization(size):
    reduced = lights_toggle_matrix(size)
    count = size * size
    transform = np.eye(count, dtype=np.uint8)
    pivots = []
    for column in range(count):
        rank = len(pivots)
        candidates = np.flatnonzero(reduced[rank:, column])
        if not len(candidates):
            continue
        pivot = rank + int(candidates[0])
        reduced[[rank, pivot]] = reduced[[pivot, rank]]
        transform[[rank, pivot]] = transform[[pivot, rank]]
        other_rows = np.flatnonzero(reduced[:, column])
        other_rows = other_rows[other_rows != rank]
        reduced[other_rows] ^= reduced[rank]
        transform[other_rows] ^= transform[rank]
        pivots.append(column)
    free = [column for column in range(count) if column not in pivots]
    basis = np.zeros((len(free), count), dtype=np.uint8)
    for index, column in enumerate(free):
        basis[index, column] = 1
        basis[index, pivots] = reduced[:len(pivots), column]
    transform.flags.writeable = False
    basis.flags.writeable = False
    return transform, np.asarray(pivots, dtype=np.int64), basis


def lights_distances(states, goals=None, size=7):
    """Exact B-vector distances, or -1 for unreachable boards.

    Goals may be B,H,W, a shared H,W board, or a broadcastable 1,H,W board.
    Zero is the default goal. Nullspace enumeration is exponential; dimensions
    above 24 raise explicitly rather than return approximate distances.
    """
    size = _positive_size(size)
    states = _lights_boards(states, size)
    if goals is None:
        rhs = states.reshape(len(states), size * size)
    else:
        goals = np.asarray(goals)
        if goals.shape == (size, size):
            goals = goals[None]
        goals = _lights_boards(goals, size, "goals")
        try:
            goals = np.broadcast_to(goals, states.shape)
        except ValueError as exc:
            raise ValueError("goals must match or broadcast to the states batch") from exc
        rhs = (states ^ goals).reshape(len(states), size * size)
    transform, pivots, basis = _lights_factorization(size)
    if len(basis) > 24:
        raise ValueError("nullspace too large for bounded exact enumeration (>24 dimensions)")
    # uint8 overflow is modulo 256 and preserves the final modulo-2 parity.
    transformed = (rhs @ transform.T) & 1
    consistent = ~np.any(transformed[:, len(pivots):], axis=1)
    result = np.full(len(states), -1, dtype=np.int64)
    particular = np.zeros((int(consistent.sum()), size * size), dtype=np.uint8)
    particular[:, pivots] = transformed[consistent, :len(pivots)]
    best = particular.sum(axis=1, dtype=np.int64)
    # Gray-code traversal changes exactly one nullspace basis vector per step.
    for index in range(1, 1 << len(basis)):
        bit = (index & -index).bit_length() - 1
        particular ^= basis[bit]
        best = np.minimum(best, particular.sum(axis=1, dtype=np.int64))
    result[consistent] = best
    return result


def _digit_board(board):
    board = np.asarray(board)
    if (board.ndim != 2 or min(board.shape) < 1 or board.dtype.kind not in "ui"
            or np.any(board < 1) or np.any(board > 6)):
        raise ValueError("board must be a nonempty integer H,W array with digits 1..6")
    return board


def _position(position, count):
    if (not isinstance(position, Integral) or isinstance(position, bool)
            or not 0 <= position < count):
        raise ValueError("position must be a valid flat integer cell index")
    return int(position)


def _digit_successors(board, position):
    height, width = board.shape
    row, col = divmod(position, width)
    jump = int(board[row, col])
    result = []
    # Reference direction order: up, right, left, down (excluding invalid no-ops).
    for rr, cc in ((row - jump, col), (row, col + jump), (row, col - jump), (row + jump, col)):
        if 0 <= rr < height and 0 <= cc < width:
            result.append(rr * width + cc)
    return result


def digit_successors(board, position):
    """Legal directed exact-digit jumps; invalid/self-loop actions are omitted."""
    board = _digit_board(board)
    return _digit_successors(board, _position(position, board.size))


def digit_distances(board, goal=None):
    """Return exact distance TO goal from every flat position using reverse BFS."""
    board = _digit_board(board)
    goal = board.size - 1 if goal is None else _position(goal, board.size)
    predecessors = [[] for _ in range(board.size)]
    for source in range(board.size):
        for destination in _digit_successors(board, source):
            predecessors[destination].append(source)
    distances = np.full(board.size, -1, dtype=np.int64)
    distances[goal] = 0
    queue = deque([goal])
    while queue:
        current = queue.popleft()
        for predecessor in predecessors[current]:
            if distances[predecessor] == -1:
                distances[predecessor] = distances[current] + 1
                queue.append(predecessor)
    return distances


def digit_encode(board, position):
    """Encode digits 1..6 with one agent-marked digit in 7..12, preserving layout."""
    board = _digit_board(board)
    position = _position(position, board.size)
    encoded = board.astype(np.uint8, copy=True)
    encoded.flat[position] += 6
    return encoded


def generate_digit_board(rng, size=20):
    """Reference-distribution iid rejection board and replayable shortest path.

    All nontrivial sizes use digits 1..min(6,size-1), matching puzzlegen. The
    explicit size=1 extension is an already-solved single-cell board. Failed
    rejection after 10,000 attempts raises; no different fallback distribution
    is silently substituted. The returned path contains flat positions.
    """
    size = _positive_size(size)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if size == 1:
        return np.ones((1, 1), dtype=np.uint8), [0]
    for _ in range(10000):
        board = rng.integers(1, min(6, size - 1) + 1, size=(size, size), dtype=np.uint8)
        distances = digit_distances(board)
        if distances[0] < 0:
            continue
        path = [0]
        while path[-1] != board.size - 1:
            current = path[-1]
            successor = next(position for position in _digit_successors(board, current)
                             if distances[position] == distances[current] - 1)
            path.append(successor)
        return board, path
    raise RuntimeError("failed to sample a solvable Digit Jumper board in 10,000 attempts")


__all__ = ["lights_toggle_matrix", "lights_apply", "lights_distances", "digit_successors",
           "digit_distances", "digit_encode", "generate_digit_board"]

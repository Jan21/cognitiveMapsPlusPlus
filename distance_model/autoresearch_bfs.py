"""Optional exact Numba BFS for Switchyard's immutable per-pool map instances.

Returns the same insertion-ordered state/distance dictionary as Yard.bfs, including
its complete-last-expansion node-cap overshoot. No randomness or approximation.
Tables are built from the current Yard each call, so padding/mutations are respected
without adding attributes that would alter the pool-cache fingerprint.
"""

import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def _passable(cell, bits, crate, direction, is_crate, gate_at, chute_at,
              plate, platemask, gatesopen, nbits):
    if cell < 0:
        return False
    gate = gate_at[cell]
    if gate >= 0:
        effective = nbits - 1 if gatesopen else bits
        if not gatesopen and crate == plate:
            effective |= platemask
        if not ((effective >> gate) & 1):
            return False
    chute = chute_at[cell]
    if not is_crate and chute >= 0 and direction != chute:
        return False
    return True


@njit(cache=True)
def _traverse(adjacent, gate_at, chute_at, lever_at, plate, platemask,
              nopush, gatesopen, nbits, worker, crate, bits, maxnodes):
    ncell = len(adjacent)
    total = ncell * ncell * nbits
    distances = np.full(total, -1, dtype=np.int32)
    order = np.empty(min(total, max(1, maxnodes + 5)), dtype=np.int64)
    initial = (worker * ncell + crate) * nbits + bits
    order[0] = initial
    distances[initial] = 0
    head, tail = 0, 1
    while head < tail and tail < maxnodes:
        code = order[head]
        head += 1
        value = distances[code] + 1
        gatebits = code % nbits
        pair = code // nbits
        c, w = pair % ncell, pair // ncell
        # This order is exactly switchyard.DIRS: N, S, W, E, then lever pull.
        for direction in range(4):
            target = adjacent[w, direction]
            next_code = -1
            if target == c:
                if nopush:
                    continue
                new_crate = adjacent[c, direction]
                if (_passable(new_crate, gatebits, c, direction, True, gate_at,
                              chute_at, plate, platemask, gatesopen, nbits)
                        and chute_at[new_crate] < 0 and new_crate != target):
                    # First passability check used OLD crate pressure; the worker
                    # check uses NEW crate pressure, matching Yard.neighbours.
                    if _passable(target, gatebits, new_crate, direction, False,
                                 gate_at, chute_at, plate, platemask, gatesopen, nbits):
                        next_code = (target * ncell + new_crate) * nbits + gatebits
            elif _passable(target, gatebits, c, direction, False, gate_at,
                           chute_at, plate, platemask, gatesopen, nbits):
                next_code = (target * ncell + c) * nbits + gatebits
            if next_code >= 0 and distances[next_code] < 0:
                distances[next_code] = value
                order[tail] = next_code
                tail += 1
        lever = lever_at[w]
        if lever >= 0 and not gatesopen:
            next_code = (w * ncell + c) * nbits + (gatebits ^ lever)
            if distances[next_code] < 0:
                distances[next_code] = value
                order[tail] = next_code
                tail += 1
    visits = order[:tail]
    return visits, distances[visits]


def _tables(yard):
    cells = yard.cells
    ncell = len(cells)
    adjacent = np.full((ncell, 4), -1, dtype=np.int64)
    gates = {cell: index for index, cell in enumerate(yard.gates)}
    gate_at = np.array([gates.get(cell, -1) for cell in cells], dtype=np.int64)
    chute_at = np.array([yard.chutes.get(cell, -1) for cell in cells], dtype=np.int64)
    levers = {cell: int(yard.wiring[index]) for index, cell in enumerate(yard.levers)}
    lever_at = np.array([levers.get(cell, -1) for cell in cells], dtype=np.int64)
    for index, (row, col) in enumerate(cells):
        for direction, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
            nr, nc = row + dr, col + dc
            if 0 <= nr < yard.G and 0 <= nc < yard.G:
                if not yard.wall[nr, nc] or (nr, nc) in gates:
                    adjacent[index, direction] = yard.cid[(nr, nc)]
    return adjacent, gate_at, chute_at, lever_at


def bfs(yard, src, maxnodes=200000):
    """Drop-in Yard.bfs replacement preserving exact dict insertion order."""
    worker, crate, bits = map(int, src)
    ncell, nbits = len(yard.cells), 1 << yard.D
    if not (0 <= worker < ncell and 0 <= crate < ncell and 0 <= bits < nbits):
        raise ValueError("BFS source contains an out-of-range state factor")
    visits, distances = _traverse(*_tables(yard), int(yard.cid[yard.plate]),
        int(yard.platemask), bool(yard.nopush), bool(yard.gatesopen), nbits,
        worker, crate, bits, int(maxnodes))
    # Vectorized decoding + C-level tolist avoids slow per-state NumPy scalar work.
    pairs = visits // nbits
    workers = (pairs // ncell).tolist()
    crates = (pairs % ncell).tolist()
    masks = (visits % nbits).tolist()
    return dict(zip(zip(workers, crates, masks), distances.tolist()))

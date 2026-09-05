"""Ground-truth distance-to-solved for CRTR Sokoban boards (12x12, tile values as in
CRTR SubFields: 0 WALL, 1 FLOOR, 2 BOX_TARGET, 3 BOX_ON_TARGET, 4 BOX, 5 PLAYER,
6 PLAYER_ON_TARGET).

Backward BFS with PULL moves from the solved set (all boxes on targets, agent at any
reachable cell), capped at --maxnodes states per board. Emits, for every (trajectory,
state index) in the given shards, the exact distance-to-solved where reached within the
cap (npz: traj, idx, dist). States beyond the cap are omitted (the trainer skips them).

Usage (per shard file range, parallelizable across CPU jobs):
    python sokoban_bfs.py --data <shard dir> --out lut_000.npz --shard_lo 0 --shard_hi 4
"""
import argparse, glob, os, collections
import numpy as np

G = 12


def parse(flat):
    b = np.asarray(flat, dtype=np.int64).reshape(G, G)
    walls = (b == 0)
    targets = np.isin(b, (2, 3, 6))
    boxes = np.isin(b, (3, 4))
    apos = np.argwhere(np.isin(b, (5, 6)))
    agent = tuple(apos[0]) if len(apos) else None       # goal frames carry no agent
    return walls, targets, frozenset(map(tuple, np.argwhere(boxes))), agent


def norm_agent(walls, boxes, agent):
    """canonical agent cell = min reachable cell (agent region), so states differing only
    by agent position within one region collapse; returns (region frozenset, canon)."""
    seen = {agent}; dq = collections.deque([agent])
    while dq:
        r, c = dq.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (r + dr, c + dc)
            if 0 <= n[0] < G and 0 <= n[1] < G and not walls[n] and n not in boxes and n not in seen:
                seen.add(n); dq.append(n)
    return seen, min(seen)


def backward_bfs(walls, targets, maxnodes):
    """Unit-cost reverse BFS over exact states (boxes frozenset, agent cell). Seeds: boxes
    on targets, agent at ANY free cell, dist 0 (Sokoban's solved test ignores the agent).
    Reverse moves, each costing 1 (matching forward moves): (a) plain agent step; (b)
    reverse of a push = agent steps from the box's old cell away from the box while the
    box returns with it (a 'pull'). Returns dict[(boxes, agent)] = exact distance."""
    tset = frozenset(map(tuple, np.argwhere(targets)))

    def okcell(p, boxes):
        return (0 <= p[0] < G and 0 <= p[1] < G and not walls[p] and p not in boxes)

    dist = {}
    dq = collections.deque()
    for r in range(G):
        for c in range(G):
            if okcell((r, c), tset):
                dist[(tset, (r, c))] = 0
                dq.append((tset, (r, c)))
    while dq and len(dist) < maxnodes:
        boxes, agent = dq.popleft()
        d = dist[(boxes, agent)]
        ar, ac = agent
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            # (a) reverse plain move: agent came from (ar-dr, ac-dc) -> any adjacent free
            prev = (ar + dr, ac + dc)
            if okcell(prev, boxes) and (boxes, prev) not in dist:
                dist[(boxes, prev)] = d + 1
                dq.append((boxes, prev))
            # (b) reverse push (pull): forward push was agent at prev2 -> agent(=box old
            # cell) with box moving from agent-cell to (ar-dr, ac-dc)... formulated
            # directly: if a box sits at (ar-dr, ac-dc) ... clearer construction below.
        # reverse push: in the forward move, agent stood at A, box at B=A+u, box went to
        # B+u and agent to B. So the CURRENT state has agent at B, box at B+u. Reverse:
        # box back to B, agent back to A=B-u.
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            bcur = (ar + dr, ac + dc)                  # box now, in push direction u=(dr,dc)
            aprev = (ar - dr, ac - dc)                 # agent before the push
            if bcur in boxes and okcell(aprev, boxes):
                nb = frozenset((boxes - {bcur}) | {agent})
                if okcell(aprev, nb) and (nb, aprev) not in dist:
                    dist[(nb, aprev)] = d + 1
                    dq.append((nb, aprev))
    return dist


def dist_of(flat, cache, maxnodes):
    walls, targets, boxes, agent = parse(flat)
    if agent is None:                                   # agent-less goal frame
        tset = frozenset(map(tuple, np.argwhere(targets)))
        return 0 if boxes == tset else -1
    key = (walls.tobytes(), targets.tobytes())
    if key not in cache:
        cache.clear()                                   # one board per trajectory: keep ONE
        cache[key] = backward_bfs(walls, targets, maxnodes)
    table = cache[key]
    return table.get((boxes, agent), -1)


def main():
    import cloudpickle
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard_lo", type=int, default=0)
    ap.add_argument("--shard_hi", type=int, default=999)
    ap.add_argument("--traj_lo", type=int, default=0, help="global trajectory range lo")
    ap.add_argument("--traj_hi", type=int, default=10**9)
    ap.add_argument("--maxnodes", type=int, default=2000000)
    ap.add_argument("--stride", type=int, default=1, help="label every k-th state per traj")
    a = ap.parse_args()
    files = sorted(f for f in glob.glob(os.path.join(a.data, "**", "*.pkl"), recursive=True)
                   if "lens" not in f and ".cache" not in f)
    files = files[a.shard_lo:a.shard_hi]
    cache = {}
    TR, IX, DS = [], [], []
    base = 0
    for f in files:
        with open(f, "rb") as fh:
            t = cloudpickle.load(fh).numpy()
        if t.ndim == 4:
            t = t.reshape(t.shape[0], t.shape[1], -1)
        lf = next((c for c in (f.replace(".pkl", "_lens.pkl"), f.replace("trajectories", "lens"))
                   if os.path.exists(c)), None)
        if lf:
            with open(lf, "rb") as fh:
                L = cloudpickle.load(fh).numpy()
        else:
            L = np.full(len(t), t.shape[1])
        for ti in range(len(t)):
            gt = base + ti
            if gt < a.traj_lo or gt >= a.traj_hi:
                continue
            for si in range(0, int(L[ti]), a.stride):
                d = dist_of(t[ti, si], cache, a.maxnodes)
                if d >= 0:
                    TR.append(gt); IX.append(si); DS.append(d)
            if (gt - a.traj_lo) % 200 == 0:
                print(f"traj {gt}: labeled {len(DS)}", flush=True)
        base += len(t)
    np.savez(a.out, traj=np.array(TR), idx=np.array(IX), dist=np.array(DS, np.float32))
    print(f"wrote {a.out}: {len(DS)} labels, {len(cache)} boards", flush=True)


if __name__ == "__main__":
    main()

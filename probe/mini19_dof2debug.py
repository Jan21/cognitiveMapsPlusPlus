"""Why does DOF-2 (key=0, mover-only free) nan? Load a marker model and dissect the measurement at
key=0 probes: free-detection, per-agent 1-step distances, the jitter-pool distance distribution
(range/points), and the VGT verdict."""
import argparse, numpy as np, torch
import mini17_image as im

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--G", type=int, default=12); ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--encoder", default="marker")
    ap.add_argument("--keygate", action="store_true")
    ap.add_argument("--ckpt", default="/home/jan/projects/CIIRC/colabs/Alma/cognitiveMapsPlusPlus/factored_vis/mini17_image.pt")
    args = ap.parse_args()
    im.set_G(args.G); N, NP = im.N, im.NP; dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = im.ImageDist(d=args.d, heads=args.heads, encoder=args.encoder, keygate=args.keygate); head.load_state_dict(torch.load(args.ckpt, map_location="cpu")); head.to(dev).eval()
    rng = np.random.default_rng(3)
    print(f"DOF-2 debug: {args.encoder} G={args.G}, key=0 (only mover free)\n")
    with torch.no_grad():
        for t in range(6):
            probe = np.concatenate([rng.integers(1, NP, N), [0]]); pt = torch.as_tensor(probe[None], device=dev)
            steps = []
            for j in range(N):
                q = probe.copy(); q[j] = im.step2d(q[j], 1, 0); steps.append(round(float(head(pt, torch.as_tensor(q[None], device=dev))[0]), 2))
            free = im.free_agents(head, probe, dev)
            # jitter the detected-free agents, look at the distance distribution
            W, M = min(16, args.G // 3), 80000; pool = np.tile(probe, (M, 1))
            for j in free:
                r = (pool[:, j] // args.G + rng.integers(-W, W + 1, M)) % args.G; c = (pool[:, j] % args.G + rng.integers(-W, W + 1, M)) % args.G
                pool[:, j] = r * args.G + c
            d = im.dist_all(head, probe, pool, dev); dd = np.sort(d[d > 1e-9])
            near = np.sort(d[np.argsort(d)[1:1201]]); nn = near[near > 1e-9]
            lr = (np.log(nn[-1]) - np.log(nn[0])) if nn.size and nn[0] > 0 else 0
            vg, _ = im.measure(head, probe, rng, dev)
            print(f"probe{t}: 1step={steps} free={free} | nearest1200: n={nn.size} min={nn[0]:.3f} med={nn[nn.size//2]:.3f} max={nn[-1]:.3f} logrange={lr:.2f} | vgt={vg}")

if __name__ == "__main__":
    main()

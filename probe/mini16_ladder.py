import numpy as np, torch
import mini11_estimators as m11
m11.N, m11.G, m11.Kk = 5, 48, 5
from mini12_gate_guided import measure
from sweep_worker import AttnDist1D_iso
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
head = AttnDist1D_iso(d=48); head.load_state_dict(torch.load("/home/jan/projects/CIIRC/colabs/Alma/cognitiveMapsPlusPlus/factored_vis/mini15_model.pt", map_location="cpu")); head.to(dev).eval()
rj = np.random.default_rng(7)
print("48k best-config model | estimated DOF vs true DOF (gate-guided VGT, 15 probes each)\n")
print(f"{'true DOF':>9} | {'estimated (mean)':>16} | {'std':>6} | {'per-probe range':>18}")
for m in range(5):
    probes = [np.concatenate([rj.integers(1,48,5),[m]]) for _ in range(15)]
    vg = [measure(head,p,rj,dev,W=16,M=100000,L=1500)[0] for p in probes]
    vg = np.array([x for x in vg if np.isfinite(x) and 0<x<40])
    print(f"{m+1:>9} | {vg.mean():>16.2f} | {vg.std():>6.2f} | {vg.min():.2f} - {vg.max():.2f}")

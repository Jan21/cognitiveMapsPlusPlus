#!/usr/bin/env python3
"""Phase B' (NO fgpix anywhere): wait for the slot objectness-hint runs, pick the best integ slot variant, then submit the final
L0-L6 x map/wire x 3-seed ladders on Leonardo: integ (slots16/d256/L3 plain + best hint variant) vs each baseline at its
best non-fgpix encoder (unconstrained) AND its param-matched (slots, mean-pool) config."""
import subprocess, json, time, re
from collections import defaultdict
def ssh(host, cmd, t=60):
    try: return subprocess.run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=20",host,cmd],capture_output=True,text=True,timeout=t).stdout
    except Exception: return ""
# ---- wait for objhint (30 runs on CIIRC)
for it in range(40):
    out=ssh("ciirc-old-cluster",'cd ~/cognitiveMapsPlusPlus/distance_model; grep -h "^RESULT" objhint_*.out 2>/dev/null; echo "===Q $(squeue -h -u hulajan1 -n objhint | wc -l)"')
    n=out.count("RESULT"); q=re.search(r"===Q (\d+)",out)
    print(f"[{time.strftime('%H:%M')}] objhint {n}/30 (q {q.group(1) if q else '?'})", flush=True)
    if n>=28 or (q and q.group(1)=="0" and it>1): break
    time.sleep(600)
tab=defaultdict(lambda: defaultdict(list))
for l in out.split("\n"):
    if not l.startswith("RESULT"): continue
    d=json.loads(l[7:]); t=d["tag"]; parts=t.split("_"); sp=parts[1][2:]; var="_".join(parts[2:-1])
    c=d["integ"]["test_corr"]
    if c is not None: tab[var][sp].append(c)
mean=lambda v: sum(v)/len(v) if v else -1
score={v:(mean(tab[v]["map"])+mean(tab[v]["wire"]))/2 for v in tab}
print("hint variants (mean of map+wire):", {k:round(v,3) for k,v in score.items()})
plain=(0.843+0.852)/2
bestv=max(score,key=score.get) if score else None
use_hint = bestv if (bestv and score[bestv]>plain+0.01) else None
print("plain slots16/d256/L3 ref:",round(plain,3),"-> hint variant used:",use_hint)
HINT={"objch":"--objch 1","fgm2":"--fgmask 2","objch_fgm2":"--objch 1 --fgmask 2","objch_cc":"--objch 1 --coordconv 1","objch_cc_fgm2":"--objch 1 --coordconv 1 --fgmask 2"}
# ---- phase B'
base="--enc pureimage --heads 4 --nmaps 200 --poolq 2000 --steps 80000 --gradclip 1.0 --warmup 2000"
R=[("L0","--gatesopen --nopush"),("L1","--gatesopen"),("L2","--wire1 --noplate"),("L3","--noplate"),("L4","--nchute 0"),("L5",""),("L6","--ngate 4 --nlever 3")]
INTEG="--cnnk 1 --readout xattn --slots 16 --d 256 --layers 3 --T 4 --lr 1e-3 --nobaseline"
BASE={ # head: (unconstrained, matched)  -- best NON-fgpix configs from L5 tuning
 "iqe":   ("--readout pixels --cnnk 3 --coordconv 1 --cnndepth 3 --cnnw 128 --basepool flat --baselayers 0 --d 128 --lr 1e-3", "--readout xattn --slots 12 --cnnk 1 --basepool mean --baselayers 4 --d 128 --lr 1e-3"),
 "mrn":   ("--readout xattn --slots 12 --cnnk 1 --basepool mean --baselayers 4 --d 128 --lr 2e-3", None),
 "sym":   ("--readout pixels --cnnk 1 --coordconv 1 --basepool flat --baselayers 0 --d 256 --lr 1e-3", "--readout xattn --slots 12 --cnnk 1 --basepool mean --baselayers 4 --d 128 --lr 1e-3"),
 "scalar":("--readout pixels --cnnk 3 --coordconv 1 --cnndepth 4 --cnnw 128 --basepool flat --baselayers 4 --d 128 --lr 1e-3", "--readout xattn --slots 12 --cnnk 1 --basepool mean --baselayers 4 --d 128 --lr 1e-3"),
}
A=[]
for rung,fl in R:
    for sp in ("map","wire"):
        for sd in (0,1,2):
            A.append((f"Bp_{rung}{sp}_integ_s{sd}", f"{base} {fl} --split {sp} --seed {sd} {INTEG}"))
            if use_hint: A.append((f"Bp_{rung}{sp}_integH_s{sd}", f"{base} {fl} --split {sp} --seed {sd} {INTEG} {HINT[use_hint]}"))
            for hh,(unc,mat) in BASE.items():
                A.append((f"Bp_{rung}{sp}_{hh}_s{sd}", f"{base} {fl} --split {sp} --seed {sd} {unc} --{hh}only"))
                A.append((f"Bp_{rung}{sp}_{hh}O_s{sd}", f"{base} {fl} --split {sp} --seed {sd} {unc} --objch 1 --{hh}only"))     # + objectness channel (fair)
                if mat:
                    A.append((f"Bp_{rung}{sp}_{hh}M_s{sd}", f"{base} {fl} --split {sp} --seed {sd} {mat} --{hh}only"))
                    A.append((f"Bp_{rung}{sp}_{hh}MO_s{sd}", f"{base} {fl} --split {sp} --seed {sd} {mat} --objch 1 --{hh}only"))
lines=[f' "python3 switchyard.py --train {cmd} --tag {tag}"' for tag,cmd in A]
sb=f'''#!/bin/bash
#SBATCH -A EUHPC_B38_121
#SBATCH -p boost_usr_prod
#SBATCH --job-name=phaseBp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --array=0-{len(A)-1}%{len(A)}
#SBATCH --output=/leonardo_scratch/large/userexternal/jhula000/cmpp_out/phaseBp_%a.out
# PHASE B' (no fgpix): final ladders L0-L6 x map/wire x 3 seeds. integ = 16 slots / d256 / 3 layers (+ best objectness-hint variant);
# baselines at best NON-fgpix encoder (unconstrained) and at param-matched slots/mean-pool config. One setting per model at all rungs.
module purge; module load profile/deeplrn cineca-ai/4.3.0 >/dev/null 2>&1
export PYTHONPATH=$HOME/cmpp/torch-quasimetric:$PYTHONPATH
cd $HOME/cmpp/distance_model
A=(
{chr(10).join(lines)}
)
echo "TASK $SLURM_ARRAY_TASK_ID / ${{#A[@]}} : ${{A[$SLURM_ARRAY_TASK_ID]}}"
srun ${{A[$SLURM_ARRAY_TASK_ID]}}
'''
open("/home/jan/projects/CIIRC/colabs/Alma/cognitiveMapsPlusPlus/distance_model/leo_phaseBp.sbatch","w").write(sb)
json.dump({"integ":INTEG,"hint":use_hint,"hint_scores":score,"baselines":BASE},open("/home/jan/projects/CIIRC/colabs/Alma/cognitiveMapsPlusPlus/distance_model/phaseBp_config.json","w"),indent=1)
print("phase B' runs:",len(A))
subprocess.run(["scp","-q","/home/jan/projects/CIIRC/colabs/Alma/cognitiveMapsPlusPlus/distance_model/switchyard.py","/home/jan/projects/CIIRC/colabs/Alma/cognitiveMapsPlusPlus/distance_model/leo_phaseBp.sbatch","leonardo:~/cmpp/distance_model/"])
print(ssh("leonardo","cd ~/cmpp/distance_model && sbatch leo_phaseBp.sbatch"))

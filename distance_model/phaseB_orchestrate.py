#!/usr/bin/env python3
"""Wait for phase A (baseline tuning on fgpix+coordconv tokens), choose each baseline's best (readout, depth, lr)
per rung group, generate + submit phase B (final pure-image ladders) on Leonardo."""
import subprocess, json, time, re, sys
from collections import defaultdict
S="/tmp/claude-1000/-home-jan-projects-CIIRC-colabs-Alma-cognitiveMapsPlusPlus/"
def ssh(host, cmd, t=60):
    try: return subprocess.run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=20",host,cmd],capture_output=True,text=True,timeout=t).stdout
    except Exception as e: return ""
def head(d):
    for h in ("integ","iqe","mrn","sym","scalar"):
        if h in d: return h,d[h]
# ---- wait for phase A
for it in range(60):
    leo=ssh("leonardo",'grep -h "^RESULT" $CINECA_SCRATCH/cmpp_out/tuneA_*.out 2>/dev/null; echo "===Q $(squeue --me -h | wc -l)"')
    ci=ssh("ciirc-old-cluster",'cd ~/cognitiveMapsPlusPlus/distance_model; grep -h "^RESULT" tuneA5_*.out 2>/dev/null; echo "===Q $(squeue -h -u hulajan1 -n tuneA5 | wc -l)"')
    nl=leo.count("RESULT"); nc=ci.count("RESULT")
    ql=re.search(r"===Q (\d+)",leo); qc=re.search(r"===Q (\d+)",ci)
    print(f"[{time.strftime('%H:%M')}] A: leo {nl}/99 (q {ql.group(1) if ql else '?'}), ciirc {nc}/36 (q {qc.group(1) if qc else '?'})", flush=True)
    if nl>=95 and nc>=34: break
    if ql and qc and ql.group(1)=="0" and qc.group(1)=="0" and it>2: break
    time.sleep(600)
open(S+"tuneA.txt","w").write(leo+ci)
# ---- parse A + earlier slot tuning (ft_) from leo.txt
rows=[]
for fn in ("tuneA.txt","leo.txt"):
    for l in open(S+fn):
        if not l.startswith("RESULT"): continue
        d=json.loads(l[7:]); t=d.get("tag","")
        if t.startswith("tA_") or t.startswith("ft_"): rows.append(d)
best=defaultdict(dict)   # best[(group, head)] = (score, readout, L, lr)
for d in rows:
    t=d["tag"]; h,r=head(d); c=r["test_corr"]
    if c is None: continue
    if t.startswith("tA_"):
        _,rung,ro,hh,Ls,lrs=t.split("_"); ro="fgcc" if ro=="fgcc" else "slots"
    else:
        _,rung,hh,Ls,lrs=t.split("_"); ro="slots"
    L=int(Ls[1:]); lr=lrs[2:]
    key=(rung,hh)
    if key not in best or c>best[key][0]: best[key]=(c,ro,L,lr)
print("best per (rung, head):"); 
for k in sorted(best): print("  ",k,best[k])
def cfg(rung,hh):
    grp = "L2" if rung in ("L0","L1","L2") else ("L3" if rung=="L3" else ("L5" if ("L5",hh) in best else "L3"))
    return best.get((grp,hh)) or best.get(("L3",hh)) or best.get(("L2",hh))
# ---- generate phase B
base="--enc pureimage --cnnk 1 --d 128 --heads 4 --cnnw 64 --cnndepth 2 --nmaps 200 --poolq 2000 --steps 80000 --gradclip 1.0 --warmup 2000"
R=[("L0","--gatesopen --nopush"),("L1","--gatesopen"),("L2","--wire1 --noplate"),("L3","--noplate"),("L4","--nchute 0"),("L5",""),("L6","--ngate 4 --nlever 3")]
A=[]
for rung,fl in R:
    for sp in ("map","wire"):
        for sd in (0,1,2):
            A.append((f"B_{rung}{sp}_integ_s{sd}", f"{base} {fl} --split {sp} --seed {sd} --readout fgpix --coordconv 1 --T 4 --layers 4 --lr 1e-3 --nobaseline"))
            A.append((f"B_{rung}{sp}_integL3_s{sd}", f"{base} {fl} --split {sp} --seed {sd} --readout fgpix --coordconv 1 --T 4 --layers 3 --lr 1e-3 --nobaseline"))
            for hh in ("iqe","mrn","sym","scalar"):
                c=cfg(rung,hh)
                if not c: continue
                _,ro,L,lr=c
                rd = "--readout fgpix --coordconv 1" if ro=="fgcc" else "--readout xattn --slots 12"
                A.append((f"B_{rung}{sp}_{hh}_s{sd}", f"{base} {fl} --split {sp} --seed {sd} {rd} --{hh}only --baselayers {L} --lr {lr}"))
lines=[f' "python3 switchyard.py --train {cmd} --tag {tag}"' for tag,cmd in A]
sb=f'''#!/bin/bash
#SBATCH -A EUHPC_B38_121
#SBATCH -p boost_usr_prod
#SBATCH --job-name=phaseB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --array=0-{len(A)-1}%{len(A)}
#SBATCH --output=/leonardo_scratch/large/userexternal/jhula000/cmpp_out/phaseB_%a.out
# PHASE B: final pure-image ladders L0-L6 x map/wire x 3 seeds. integ = fgpix+coordconv (4 layers; 3-layer variant), baselines at their
# best (representation, depth, lr) from phase A / on-bed tuning. Same encoder, data, loss, steps, warm-up, clipping.
module purge; module load profile/deeplrn cineca-ai/4.3.0 >/dev/null 2>&1
export PYTHONPATH=$HOME/cmpp/torch-quasimetric:$PYTHONPATH
cd $HOME/cmpp/distance_model
A=(
{chr(10).join(lines)}
)
echo "TASK $SLURM_ARRAY_TASK_ID / ${{#A[@]}} : ${{A[$SLURM_ARRAY_TASK_ID]}}"
srun ${{A[$SLURM_ARRAY_TASK_ID]}}
'''
open("/home/jan/projects/CIIRC/colabs/Alma/cognitiveMapsPlusPlus/distance_model/leo_phaseB.sbatch","w").write(sb)
json.dump({f"{k[0]}|{k[1]}":v for k,v in best.items()}, open("/home/jan/projects/CIIRC/colabs/Alma/cognitiveMapsPlusPlus/distance_model/phaseA_best.json","w"), indent=1)
print("phase B runs:",len(A))
subprocess.run(["scp","-q","/home/jan/projects/CIIRC/colabs/Alma/cognitiveMapsPlusPlus/distance_model/leo_phaseB.sbatch","leonardo:~/cmpp/distance_model/"])
print(ssh("leonardo","cd ~/cmpp/distance_model && sbatch leo_phaseB.sbatch"))

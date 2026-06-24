#!/usr/bin/env python3
"""
Train ONE RT-DETR fold in its OWN process. This is the fix for the
"run stops after fold 0" bug: Ultralytics DDP cannot be re-initialised a 2nd
time inside the same Python process, so looping folds in one notebook process
crashes fold 1+. Launching each fold via subprocess gives every fold a fresh
interpreter + fresh torch.distributed group -> the crash is structurally
impossible to repeat.

The SMOKE cell and the HEAVY cell call this SAME script (only epochs differ),
so a green smoke run is a real guarantee for the heavy run.

argv: <fold_k> <ds_dir> <runs_dir> <mode: smoke|full>
"""
import os, sys
import rtdetr_lib as L

k    = int(sys.argv[1])
ds   = sys.argv[2]
runs = sys.argv[3]
mode = sys.argv[4] if len(sys.argv) > 4 else "full"
smoke = (mode == "smoke")

name = f"smoke_f{k}" if smoke else f"rtdetr_l_f{k}"
L.train_one(
    f"{ds}/data_fold{k}.yaml", runs, name,
    model="rtdetr-l.pt", imgsz=1280, batch=8,
    epochs=(2 if smoke else 100), patience=20,
    device="0,1", smoke=smoke,
)

best = f"{runs}/{name}/weights/best.pt"
print(f"[_train_fold] fold {k} ({mode}) done -> best.pt exists: {os.path.exists(best)}")
sys.exit(0 if os.path.exists(best) else 1)

#!/usr/bin/env python3
"""
Train ONE RT-DETR fold in its OWN process. Fixes TWO bugs that killed multi-fold runs:

1) DDP re-init: Ultralytics can't re-initialise DDP twice in one Python process, so
   looping folds in a single notebook process crashes fold 1+. A fresh subprocess per
   fold gives each a clean interpreter + torch.distributed group.

2) GPU clamp leak: Ultralytics predict/val with device='0' sets CUDA_VISIBLE_DEVICES='0'
   in the parent kernel; the training subprocess INHERITS it and then sees only 1 GPU
   -> 'device=1, num_gpus=1' crash on device='0,1'. We undo that clamp here (must happen
   BEFORE importing torch) and auto-pick device from the GPUs actually present.

The SMOKE cell and the HEAVY cell call this SAME script (only epochs differ),
so a green smoke run is a real guarantee for the heavy run.

argv: <fold_k> <ds_dir> <runs_dir> <mode: smoke|full>
"""
import os, sys
os.environ.pop("CUDA_VISIBLE_DEVICES", None)   # undo any clamp inherited from the notebook

import rtdetr_lib as L
import torch

k    = int(sys.argv[1])
ds   = sys.argv[2]
runs = sys.argv[3]
mode = sys.argv[4] if len(sys.argv) > 4 else "full"
smoke = (mode == "smoke")

ngpu = torch.cuda.device_count()
device = "0,1" if ngpu >= 2 else ("0" if ngpu == 1 else "cpu")
print(f"[_train_fold] fold {k} ({mode}): visible GPUs={ngpu} -> device={device}")

name = f"smoke_f{k}" if smoke else f"rtdetr_l_f{k}"
L.train_one(
    f"{ds}/data_fold{k}.yaml", runs, name,
    model="rtdetr-l.pt", imgsz=1280, batch=8,
    epochs=(2 if smoke else 100), patience=20,
    device=device, smoke=smoke,
)

best = f"{runs}/{name}/weights/best.pt"
print(f"[_train_fold] fold {k} ({mode}) done -> best.pt exists: {os.path.exists(best)}")
sys.exit(0 if os.path.exists(best) else 1)

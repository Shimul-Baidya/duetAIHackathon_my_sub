"""
Track #2 RT-DETR-l — single source of truth (used by the Kaggle notebook + CLIs).
Pure functions, no module-level side effects. GPU steps need ultralytics + a GPU.
"""
import os, json, time, glob, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import submission_utils as S

NAMES = ["Rickshaw","Motorcycle","Tempu","Sedan Car","Pickup","Microbus","Mini Bus",
         "Mini Truck","Agro Use","Medium Truck","Large Bus","Heavy Truck","Trailer"]

# ----------------------------------------------------------------- data
def find_competition_src(default="data"):
    """Auto-locate the dir that CONTAINS train/train.csv (+ test/images), at any nesting
    depth under /kaggle/input. Falls back to any train.csv if the test dir name differs."""
    if os.path.exists(f"{default}/train/train.csv"):
        return os.path.abspath(default)
    cands = sorted(glob.glob("/kaggle/input/**/train/train.csv", recursive=True))
    for csv in cands:                                  # .../<SRC>/train/train.csv
        src = os.path.dirname(os.path.dirname(csv))
        if os.path.isdir(f"{src}/test/images"):
            print(f"[data] auto-detected SRC = {src}")
            return src
    if cands:                                          # train.csv found but no test/images sibling
        src = os.path.dirname(os.path.dirname(cands[0]))
        print(f"[data] WARNING: using SRC = {src} but no {src}/test/images — check test path")
        return src
    found = glob.glob("/kaggle/input/**/*.csv", recursive=True)[:8]
    raise FileNotFoundError(
        "Could not find train/train.csv under /kaggle/input. Add the competition as a Kaggle "
        f"input, or set SRC manually. CSVs seen: {found or 'none'}")

def build_dataset(src, ds_out, k=3):
    """CSV -> YOLO labels + camera-stratified frame-phase K-fold. Idempotent. CPU only."""
    import pandas as pd
    src_img = os.path.abspath(f"{src}/train/images")
    out = os.path.abspath(ds_out)
    img_out, lab_out, fold_out = f"{out}/images", f"{out}/labels", f"{out}/folds"
    for d in (img_out, lab_out, fold_out): os.makedirs(d, exist_ok=True)
    df = pd.read_csv(f"{src}/train/train.csv")
    df["stem"] = df.image_id.str.replace(".txt","",regex=False)
    df["cam"]  = df.stem.str.split("^").str[0]
    df["frame"]= df.stem.str.extract(r"_(\d+)$").astype(int)
    for stem,g in df.groupby("stem"):
        open(f"{lab_out}/{stem}.txt","w").write(
            "\n".join(f"{int(r.class_id)} {r.x_center:.6f} {r.y_center:.6f} "
                      f"{r.width:.6f} {r.height:.6f}" for r in g.itertuples())+"\n")
    for stem in df.stem.unique():
        d=f"{img_out}/{stem}.jpg"
        if not (os.path.exists(d) or os.path.islink(d)):
            try: os.symlink(f"{src_img}/{stem}.jpg", d)
            except FileExistsError: pass
    meta=df.groupby("stem").agg(cam=("cam","first"),frame=("frame","first")).reset_index()
    sf={}
    for cam,g in meta.groupby("cam"):
        for rank,row in g.sort_values("frame").reset_index(drop=True).iterrows():
            sf[row.stem]=rank % k
    meta["fold"]=meta.stem.map(sf); meta.to_csv(f"{out}/folds.csv",index=False)
    for ki in range(k):
        val=meta[meta.fold==ki].stem; trn=meta[meta.fold!=ki].stem
        open(f"{fold_out}/fold{ki}_val.txt","w").write("\n".join(f"{img_out}/{s}.jpg" for s in val)+"\n")
        open(f"{fold_out}/fold{ki}_train.txt","w").write("\n".join(f"{img_out}/{s}.jpg" for s in trn)+"\n")
        _write_yaml(f"{out}/data_fold{ki}.yaml", out, f"folds/fold{ki}_train.txt", f"folds/fold{ki}_val.txt")
    print(f"[data] {out} | {len(meta)} imgs | K={k}")
    return out

def _write_yaml(path, root, train_rel, val_rel):
    open(path,"w").write(f"path: {root}\ntrain: {train_rel}\nval: {val_rel}\n"
                         f"nc: 13\nnames: {NAMES}\n")

def make_smoke_subset(ds_out, n_train=64, n_val=24):
    """Tiny data yaml carved from fold0 so the smoke test finishes in ~2 min."""
    out=os.path.abspath(ds_out)
    tr=open(f"{out}/folds/fold0_train.txt").read().split()[:n_train]
    va=open(f"{out}/folds/fold0_val.txt").read().split()[:n_val]
    open(f"{out}/folds/smoke_train.txt","w").write("\n".join(tr)+"\n")
    open(f"{out}/folds/smoke_val.txt","w").write("\n".join(va)+"\n")
    _write_yaml(f"{out}/data_smoke.yaml", out, "folds/smoke_train.txt", "folds/smoke_val.txt")
    return f"{out}/data_smoke.yaml"

# ----------------------------------------------------------------- train
def train_one(data_yaml, runs, name, model="rtdetr-l.pt", imgsz=1024, batch=8,
              epochs=100, patience=20, device="0,1", workers=4, seed=42, smoke=False):
    """Train one RT-DETR run. smoke=True => 2 epochs, no plots. Returns summary dict."""
    import torch
    from ultralytics import RTDETR
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0=time.time()
    m=RTDETR(model)
    m.train(
        data=data_yaml, epochs=(2 if smoke else epochs), patience=patience,
        imgsz=imgsz, batch=batch, device=device, workers=workers, seed=seed,
        project=runs, name=name, exist_ok=True, amp=True, cache=False,
        plots=(not smoke), val=True,
        optimizer="AdamW", lr0=1e-4, lrf=0.1, cos_lr=True, warmup_epochs=(0 if smoke else 5),
        weight_decay=1e-4,
        # RT-DETR is unstable with mosaic in ultralytics: mosaic emits 2x-size (2560) images
        # that aren't cropped back to imgsz in the RT-DETR pipeline -> collate stack crash.
        # RT-DETR was designed WITHOUT mosaic; use single-image augs (still differs from
        # Track#1's mosaic-heavy YOLO -> ensemble diversity preserved).
        mosaic=0.0, mixup=0.0, copy_paste=0.0, close_mosaic=0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, scale=0.5, translate=0.1, fliplr=0.5,
        flipud=0.0, degrees=0.0, perspective=0.0,
    )
    best=f"{runs}/{name}/weights/best.pt"
    peak=(torch.cuda.max_memory_allocated()/1e9) if torch.cuda.is_available() else 0
    summ={"name":name,"best":best,"minutes":round((time.time()-t0)/60,1),
          "imgsz":imgsz,"batch":batch,"peak_gpu_gb":round(peak,1)}
    if not smoke:
        res=RTDETR(best).val(data=data_yaml, imgsz=imgsz, batch=batch, device=str(device).split(",")[0],
                             split="val", plots=False, verbose=False)
        summ["map50"]=float(res.box.map50); summ["map5095"]=float(res.box.map)
        summ["per_class_ap50"]={res.names[c]:round(float(res.box.ap50[i]),4)
                                for i,c in enumerate(res.box.ap_class_index)}
        os.makedirs(f"{runs}/_summary",exist_ok=True)
        json.dump(summ,open(f"{runs}/_summary/{name}.json","w"),indent=2)
        print(f"[{name}] mAP50={summ['map50']:.4f} mAP50-95={summ['map5095']:.4f} "
              f"peak={summ['peak_gpu_gb']}GB ({summ['minutes']}min)")
        print("  per-class AP50: "+" ".join(f"{n}:{v:.2f}" for n,v in summ["per_class_ap50"].items()))
    else:
        print(f"[smoke] OK best.pt={os.path.exists(best)} peak={summ['peak_gpu_gb']}GB "
              f"({summ['minutes']}min) -> imgsz={imgsz} batch={batch} fit on this GPU set.")
    return summ

# ----------------------------------------------------------------- infer
def predict_one(weights, test_dir, imgsz=1024, conf=0.001, iou=0.7, max_det=300,
                device="0", tta=False):
    """Predict one model on test set. Returns cache {stem.txt:[[cls,conf,cx,cy,w,h]]}."""
    from ultralytics import RTDETR
    m=RTDETR(weights)
    imgs=sorted(glob.glob(f"{test_dir}/*.jpg"))
    cache={}
    for r in m.predict(imgs, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                       augment=tta, device=device, verbose=False, stream=True):
        stem=os.path.splitext(os.path.basename(r.path))[0]; b=r.boxes; dets=[]
        if b is not None and len(b):
            xywhn=b.xywhn.cpu().numpy(); cf=b.conf.cpu().numpy(); cl=b.cls.cpu().numpy()
            for (cx,cy,w,h),c,k in zip(xywhn,cf,cl):
                dets.append([int(k),float(c),float(cx),float(cy),float(w),float(h)])
        cache[f"{stem}.txt"]=dets
    return cache

def infer_all_folds(runs, test_dir, out_dir, imgsz=1024, device="0", tta=False,
                    wbf_iou=0.55, skip_thr=0.001, conf_type="avg", reuse_cache=True):
    """Predict every fold that has a best.pt, cache each, WBF-fuse, write submission.
    Works with ANY number of completed folds (even just fold0). reuse_cache=True skips
    folds already cached -> cheap to re-run after each new fold finishes."""
    os.makedirs(f"{out_dir}/caches", exist_ok=True)
    weights=sorted(glob.glob(f"{runs}/rtdetr_l_f*/weights/best.pt"))
    assert weights, f"no trained folds under {runs}/rtdetr_l_f*/weights/best.pt"
    caches=[]
    for w in weights:
        tag=w.split("/")[-3]                       # rtdetr_l_fK
        cpath=f"{out_dir}/caches/{tag}.json"
        if reuse_cache and os.path.exists(cpath):
            c=S.load_cache(cpath); print(f"[infer] {tag}: reused cache ({len(c)} imgs)")
        else:
            c=predict_one(w, test_dir, imgsz=imgsz, device=device, tta=tta)
            S.save_cache(c, cpath); print(f"[infer] {tag}: {len(c)} imgs -> caches/{tag}.json")
        caches.append(c)
    fused = caches[0] if len(caches)==1 else S.fuse_caches(
        caches, weights=[1.0]*len(caches), iou_thr=wbf_iou, skip_box_thr=skip_thr, conf_type=conf_type)
    S.save_cache(fused, f"{out_dir}/caches/rtdetr_l_WBF.json")
    n=S.write_submission(fused, test_dir, f"{out_dir}/submission_rtdetr.csv",
                         conf_scale=100.0, conf_thr=0.001, max_det=300)
    print(f"[submission] {out_dir}/submission_rtdetr.csv  ({n} test imgs, {len(caches)} fold(s))")
    return f"{out_dir}/submission_rtdetr.csv"

def validate_submission(csv_path, test_dir):
    ids=set(S.list_test_ids(test_dir))
    lines=open(csv_path).read().strip().split("\n")
    out=[l.split(",",1)[0] for l in lines[1:]]
    ok = lines[0]=="image_id,PredictionString" and len(out)==len(ids)==len(set(out)) and set(out)==ids
    print(f"[validate] {'OK' if ok else 'FAIL'} | rows={len(out)} expected={len(ids)} | header_ok={lines[0]=='image_id,PredictionString'}")
    return ok

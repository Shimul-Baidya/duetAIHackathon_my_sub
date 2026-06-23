"""Convert train.csv -> YOLO txt labels + camera-stratified frame-phase K-fold.
Runs locally (CPU) or on Kaggle. Idempotent. No GPU needed.
Usage: python build_dataset.py --src data --out track2/yolo_ds --k 3
"""
import argparse, os, json, pandas as pd, numpy as np
from collections import defaultdict

NAMES=["Rickshaw","Motorcycle","Tempu","Sedan Car","Pickup","Microbus","Mini Bus",
       "Mini Truck","Agro Use","Medium Truck","Large Bus","Heavy Truck","Trailer"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src",default="data")
    ap.add_argument("--out",default="track2/yolo_ds")
    ap.add_argument("--k",type=int,default=3)
    ap.add_argument("--link",default="symlink",choices=["symlink","copy","none"])
    a=ap.parse_args()
    src_img=os.path.abspath(f"{a.src}/train/images")
    out=os.path.abspath(a.out)
    img_out=f"{out}/images"; lab_out=f"{out}/labels"; fold_out=f"{out}/folds"
    for d in (img_out,lab_out,fold_out): os.makedirs(d,exist_ok=True)

    df=pd.read_csv(f"{a.src}/train/train.csv")
    df["stem"]=df.image_id.str.replace(".txt","",regex=False)
    df["cam"]=df.stem.str.split("^").str[0]
    df["frame"]=df.stem.str.extract(r"_(\d+)$").astype(int)

    # ---- write YOLO label files (one per image) ----
    for stem,g in df.groupby("stem"):
        lines=[f"{int(r.class_id)} {r.x_center:.6f} {r.y_center:.6f} {r.width:.6f} {r.height:.6f}"
               for r in g.itertuples()]
        open(f"{lab_out}/{stem}.txt","w").write("\n".join(lines)+"\n")
    # ---- link/copy images ----
    import shutil
    for stem in df.stem.unique():
        s=f"{src_img}/{stem}.jpg"; d=f"{img_out}/{stem}.jpg"
        if a.link=="none": continue
        if os.path.exists(d) or os.path.islink(d): continue
        if a.link=="symlink": os.symlink(s,d)
        else: shutil.copy(s,d)

    # ---- camera-stratified frame-phase K-fold ----
    # within each camera: sort by frame, fold = rank % k  (mirrors interleaved test frames)
    stem_fold={}
    meta=df.groupby("stem").agg(cam=("cam","first"),frame=("frame","first")).reset_index()
    for cam,g in meta.groupby("cam"):
        g=g.sort_values("frame").reset_index(drop=True)
        for rank,row in g.iterrows():
            stem_fold[row.stem]=rank % a.k
    meta["fold"]=meta.stem.map(stem_fold)
    meta.to_csv(f"{out}/folds.csv",index=False)

    # ---- per-fold train/val image lists (absolute paths) ----
    for k in range(a.k):
        val=meta[meta.fold==k].stem.tolist()
        trn=meta[meta.fold!=k].stem.tolist()
        open(f"{fold_out}/fold{k}_val.txt","w").write("\n".join(f"{img_out}/{s}.jpg" for s in val)+"\n")
        open(f"{fold_out}/fold{k}_train.txt","w").write("\n".join(f"{img_out}/{s}.jpg" for s in trn)+"\n")
        # data yaml
        yml=f"""# RT-DETR fold {k}
path: {out}
train: folds/fold{k}_train.txt
val: folds/fold{k}_val.txt
nc: 13
names: {NAMES}
"""
        open(f"{out}/data_fold{k}.yaml","w").write(yml)

    # ---- report per-fold class balance (esp rare classes) ----
    df["fold"]=df.stem.map(stem_fold)
    print(f"K={a.k} folds. images/fold:")
    print(meta.groupby(["fold","cam"]).size().unstack(fill_value=0))
    print("\nVAL instances per class per fold:")
    tab=df.groupby(["fold","class_id"]).size().unstack(fill_value=0)
    tab.columns=[f"{c}:{NAMES[c]}" for c in tab.columns]
    print(tab.T)
    # flag rare-class folds with zero val instances
    warn=[]
    for k in range(a.k):
        for c in range(13):
            n=((df.fold==k)&(df.class_id==c)).sum()
            if n==0: warn.append((k,c,NAMES[c]))
    print("\n[WARN] (fold,class) with ZERO val instances:",warn if warn else "none")
    json.dump({"k":a.k,"n_images":int(meta.shape[0]),"out":out,
               "zero_val_classes":[[int(k),int(c),n] for k,c,n in warn]},
              open(f"{out}/build_summary.json","w"),indent=2)
    print("\nDataset built ->",out)

if __name__=="__main__": main()

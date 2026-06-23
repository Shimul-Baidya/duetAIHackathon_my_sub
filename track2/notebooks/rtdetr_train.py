"""CLI wrapper around rtdetr_lib (the notebook kaggle_rtdetr_track2.ipynb is the main path).
  python rtdetr_train.py --src /kaggle/input/<comp> --folds 0 1 2 --imgsz 1024 --batch 8 --device 0,1
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import rtdetr_lib as L

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--src", default=None, help="competition dir (auto-detected if omitted)")
    p.add_argument("--ds_out", default="/kaggle/working/yolo_ds")
    p.add_argument("--runs", default="/kaggle/working/runs")
    p.add_argument("--folds", type=int, nargs="+", default=[0,1,2])
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--device", default="0,1")
    p.add_argument("--smoke", action="store_true", help="2-epoch sanity run on 64 imgs")
    a=p.parse_args()
    src=a.src or L.find_competition_src()
    ds=L.build_dataset(src, a.ds_out, k=a.k)
    if a.smoke:
        y=L.make_smoke_subset(ds); L.train_one(y, a.runs, "smoke", imgsz=a.imgsz,
            batch=a.batch, device=a.device, smoke=True); return
    res=[L.train_one(f"{ds}/data_fold{k}.yaml", a.runs, f"rtdetr_l_f{k}", imgsz=a.imgsz,
                     batch=a.batch, epochs=a.epochs, patience=a.patience, device=a.device)
         for k in a.folds]
    good=[s for s in res if "map50" in s]
    if good: print(f"\nCV mAP50 = {sum(s['map50'] for s in good)/len(good):.4f}")

if __name__=="__main__": main()

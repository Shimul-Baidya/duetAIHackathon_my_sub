"""CLI wrapper around rtdetr_lib (the notebook kaggle_rtdetr_track2.ipynb is the main path).
  python rtdetr_infer.py --src /kaggle/input/<comp> --runs /kaggle/working/runs --out /kaggle/working
Works with ANY number of completed folds (even just fold0) -> submission_rtdetr.csv.
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import rtdetr_lib as L

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--src", default=None)
    p.add_argument("--runs", default="/kaggle/working/runs")
    p.add_argument("--out", default="/kaggle/working")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--device", default="0")
    p.add_argument("--tta", action="store_true")
    a=p.parse_args()
    src=a.src or L.find_competition_src()
    sub=L.infer_all_folds(a.runs, f"{src}/test/images", a.out, imgsz=a.imgsz,
                          device=a.device, tta=a.tta)
    L.validate_submission(sub, f"{src}/test/images")

if __name__=="__main__": main()

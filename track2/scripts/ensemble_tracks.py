"""
GRAND ENSEMBLE — WBF-fuse Track#1 (YOLO11-m) + Track#2 (RT-DETR-l) -> final submission.
This is the submission that should top the leaderboard: two decorrelated architectures.

Inputs may be either prediction-cache JSONs (schema in submission_utils) or raw
submission.csv files (auto-converted). Give Track#1 a slightly higher weight if its
solo CV mAP is higher.

Example:
  python ensemble_tracks.py \
    --src data \
    --inputs track1_wbf.csv:1.2  track2/out/caches/rtdetr_l_WBF.json:1.0 \
    --out final_submission.csv
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import submission_utils as S

def load_any(spec):
    """'path[:weight]' -> (cache, weight). .csv auto-converted to cache."""
    path, _, w = spec.partition(":")
    weight = float(w) if w else 1.0
    cache = S.submission_to_cache(path) if path.lower().endswith(".csv") else S.load_cache(path)
    return cache, weight

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src", default="data")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="list of 'cache.json[:w]' or 'submission.csv[:w]'")
    ap.add_argument("--out", default="final_submission.csv")
    ap.add_argument("--iou", type=float, default=0.55)
    ap.add_argument("--skip", type=float, default=0.001)
    ap.add_argument("--conf_type", default="avg")
    ap.add_argument("--conf_scale", type=float, default=100.0)
    a=ap.parse_args()
    caches, weights = [], []
    for spec in a.inputs:
        c,w=load_any(spec); caches.append(c); weights.append(w)
        print(f"  loaded {spec.split(':')[0]}  w={w}  ({len(c)} imgs)")
    fused=S.fuse_caches(caches, weights=weights, iou_thr=a.iou,
                        skip_box_thr=a.skip, conf_type=a.conf_type)
    n=S.write_submission(fused, f"{a.src}/test/images", a.out,
                         conf_scale=a.conf_scale, conf_thr=0.001)
    print(f"[grand ensemble] {a.out}  ({n} test images, {len(caches)} models)")

if __name__=="__main__": main()

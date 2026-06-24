#!/usr/bin/env python3
"""
Local mAP@0.5 scorer — NO GPU and NO images needed (boxes only).

Works for ANY candidate as long as you have its predictions on *labeled* images
(i.e. train images, since the test labels are hidden). Use it to:
  * reproduce / sanity-check a model's holdout mAP,
  * compare RT-DETR vs YOLO vs an ensemble on a common holdout,
  * tune WBF weights -- all WITHOUT spending a Kaggle submission.

Ground truth  : a dir of YOLO label .txt  ("cls cx cy w h", normalized).
Predictions   : a cache JSON {"<stem>.txt": [[cls,conf,cx,cy,w,h], ...]}  OR a
                submission.csv (image_id, PredictionString). conf scale is
                auto-detected (0-1 vs 0-100).
IoU is computed in PIXEL space (default 1280x720) -- normalized IoU is WRONG here
because x and y are scaled by different image dims.

Examples
--------
# fold-0 holdout score of a predictions cache:
python map_eval.py --gt yolo_ds/labels --folds yolo_ds/folds.csv --fold 0 \
    --pred caches/rtdetr_l_f0_VAL.json

# score a submission.csv (only meaningful on labeled/train images):
python map_eval.py --gt yolo_ds/labels --pred some_val_submission.csv
"""
import argparse, csv, glob, json, os
import numpy as np

CLASSES = ["Rickshaw","Motorcycle","Tempu","Sedan Car","Pickup","Microbus","Mini Bus",
           "Mini Truck","Agro Use","Medium Truck","Large Bus","Heavy Truck","Trailer"]


def _xywhn_to_xyxy(b, W, H):
    cx, cy, w, h = b
    return [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]


def load_gt(labels_dir, stems=None, W=1280, H=720):
    gt = {}
    files = (glob.glob(f"{labels_dir}/*.txt") if stems is None
             else [f"{labels_dir}/{s}.txt" for s in stems])
    for fp in files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        boxes = []
        if os.path.exists(fp):
            for ln in open(fp):
                p = ln.split()
                if len(p) < 5:
                    continue
                cls = int(float(p[0]))
                boxes.append([cls] + _xywhn_to_xyxy(list(map(float, p[1:5])), W, H))
        gt[stem] = boxes  # empty list = image with no objects (still scored)
    return gt


def load_pred(path, W=1280, H=720):
    """Returns {stem: [[cls,conf,x1,y1,x2,y2], ...]}."""
    pred = {}
    if path.endswith(".json"):
        raw = json.load(open(path))
        rows = [(k, v) for k, v in raw.items()]
    else:  # submission.csv
        rows = []
        for r in csv.DictReader(open(path)):
            k = r["image_id"]; s = (r.get("PredictionString") or "").split()
            dets = [[float(s[i]), float(s[i+1])] +
                    [float(x) for x in s[i+2:i+6]] for i in range(0, len(s) - 5, 6)]
            rows.append((k, dets))
    # auto-detect conf scale
    mx = max((d[1] for _, ds in rows for d in ds), default=1.0)
    scale = 100.0 if mx > 1.5 else 1.0
    for k, ds in rows:
        stem = k[:-4] if k.endswith(".txt") else k
        out = []
        for d in ds:
            cls = int(d[0]); conf = d[1] / scale
            out.append([cls, conf] + _xywhn_to_xyxy(d[2:6], W, H))
        pred[stem] = out
    return pred


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def ap_voc(rec, prec):
    """All-points (VOC2010+/area) interpolation."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1]))


def evaluate(gt, pred, iou_thr=0.5, classes=range(13)):
    aps = {}
    for c in classes:
        npos = sum(1 for st in gt for b in gt[st] if b[0] == c)
        dets = []  # (conf, stem, box)
        for st in gt:                          # only score images we have GT for
            for d in pred.get(st, []):
                if d[0] == c:
                    dets.append((d[1], st, d[2:6]))
        if npos == 0:
            continue                            # class absent in this holdout -> skip
        dets.sort(key=lambda x: -x[0])
        matched = {st: np.zeros(sum(1 for b in gt[st] if b[0] == c), bool) for st in gt}
        tp = np.zeros(len(dets)); fp = np.zeros(len(dets))
        for i, (cf, st, box) in enumerate(dets):
            gtb = [b[1:] for b in gt[st] if b[0] == c]
            best, bj = iou_thr, -1
            for j, g in enumerate(gtb):
                v = iou(box, g)
                if v >= best:
                    best, bj = v, j
            if bj >= 0 and not matched[st][bj]:
                tp[i] = 1; matched[st][bj] = True
            else:
                fp[i] = 1
        ctp, cfp = np.cumsum(tp), np.cumsum(fp)
        rec = ctp / npos
        prec = ctp / np.maximum(ctp + cfp, 1e-9)
        aps[c] = ap_voc(rec, prec)
    return aps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="dir of YOLO label .txt")
    ap.add_argument("--pred", required=True, help="cache .json or submission .csv")
    ap.add_argument("--folds", help="folds.csv (stem,...,fold) to restrict GT")
    ap.add_argument("--fold", type=int, help="evaluate only this fold's val stems")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="drop preds with conf (0-100 scale) below this. PROVE that "
                         "raising it LOWERS mAP: try 0 vs 25 vs 40.")
    ap.add_argument("--imgw", type=int, default=1280)
    ap.add_argument("--imgh", type=int, default=720)
    a = ap.parse_args()

    stems = None
    if a.folds is not None and a.fold is not None:
        stems = [r["stem"] for r in csv.DictReader(open(a.folds))
                 if int(r["fold"]) == a.fold]
    gt = load_gt(a.gt, stems, a.imgw, a.imgh)
    pred = load_pred(a.pred, a.imgw, a.imgh)
    if a.min_conf > 0:                       # conf is stored 0-1 internally
        thr = a.min_conf / 100.0
        pred = {s: [d for d in ds if d[1] >= thr] for s, ds in pred.items()}
        kept = sum(len(ds) for ds in pred.values())
        print(f"[min-conf {a.min_conf}] kept {kept} preds "
              f"(~{kept/max(len(pred),1):.0f}/img)")
    aps = evaluate(gt, pred)

    print(f"images scored : {len(gt)}   (predictions provided for "
          f"{sum(1 for s in gt if pred.get(s))})")
    print(f"{'class':<14} AP@0.5")
    for c in sorted(aps):
        print(f"{CLASSES[c]:<14} {aps[c]:.4f}")
    if aps:
        print("-" * 24)
        print(f"{'mAP@0.5':<14} {np.mean(list(aps.values())):.4f}  "
              f"(over {len(aps)} classes present)")


if __name__ == "__main__":
    main()

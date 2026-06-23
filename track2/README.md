# Track #2 — RT-DETR-l (diversity track)

Complementary to **Track #1 (YOLO11-m)**. Goal: a *decorrelated* model whose errors
differ from YOLO so the WBF ensemble gains. RT-DETR = transformer, set-based loss,
NMS-free → genuinely different failure modes than an anchor/NMS YOLO.

Metric: **mAP@0.5**. Deadline-aware: get a working submission first, improve if time remains.

---

## 0. TL;DR — run on Kaggle (same workflow as Track #1)
Open **`track2/notebooks/kaggle_rtdetr_track2.ipynb`** on Kaggle and Run All. It clones this
repo, installs deps, auto-finds the competition data, runs a 2-min **smoke test**, trains on
**both T4 GPUs**, and writes `submission_rtdetr.csv`. Self-sufficient — one fold is enough.
Exact click-by-click steps are in section **7**.

CLI equivalent (if you prefer a terminal):
```bash
python track2/notebooks/rtdetr_train.py --device 0,1 --batch 8 --imgsz 1024 --folds 0 1 2
python track2/notebooks/rtdetr_infer.py            # -> /kaggle/working/submission_rtdetr.csv
# optional grand ensemble with Track#1 (its submission.csv works directly):
python track2/scripts/ensemble_tracks.py --src $SRC \
  --inputs track1_submission.csv:1.2 /kaggle/working/caches/rtdetr_l_WBF.json:1.0 \
  --out final_submission.csv
```

---

## 1. Final architecture choice
**RT-DETR-l**, COCO-pretrained (`rtdetr-l.pt`), Ultralytics API.
- Chosen over RT-DETR-x: x is ~2× heavier, OOM/time risk on a 16 GB Kaggle GPU isn't
  justified for a diversity member. **l** is the score/time sweet spot.
- Not another YOLO variant (would be too correlated with Track #1 → weak ensemble gain).
- **Fallback if RT-DETR is unstable/slow:** YOLOv9-e (different backbone/PGI, trains easier).
  Swap `RTDETR("rtdetr-l.pt")`→`YOLO("yolov9e.pt")`; everything else (data, folds, infer,
  WBF) is unchanged.

## 2. Training configuration
| | |
|---|---|
| imgsz | **1280** default = native res, best for the small/foggy CCTV objects (T4×2 fits batch 4; ~1.5–2 h/fold → ~5 h for 3 folds, well under the 12 h limit). Fallback **1024** (batch 8) if OOM or you want it faster. |
| folds | **3**, camera-stratified **frame-phase** split (same methodology as Track#1) |
| epochs / patience | 100 / 20 (early-stops; `close_mosaic` last 10) |
| optimizer | AdamW, lr0 1e-4, lrf 0.1, **cosine**, warmup 5, wd 1e-4 |
| batch | 4 (T4×2 @1280, 2/GPU) · 8 (@1024). OOM in smoke test → drop imgsz to 1024 or batch to 2. |
| amp | on |
| **augmentation (different on purpose)** | mosaic 0.5, mixup 0.15, copy_paste 0.3, hsv (.015/.5/.4), scale 0.5, translate 0.1, fliplr 0.5. **No** flipud/rotate/perspective — CCTV geometry is fixed. |
| weather/noise | `pip install albumentations` → Ultralytics auto-adds Blur/MedianBlur/CLAHE/ToGray at low p (CCTV07 is foggy — this helps cheaply). |

**Why 3 folds, fold-0 first:** train fold 0 alone → make a 1-fold insurance submission early,
then add folds 1–2 and re-ensemble. Never leaves you with zero submission near the deadline.

## 3. Validation (camera-stratified frame-phase, k=3 — mirrors interleaved test frames)
Reported automatically per fold to `runs/_summary/foldK.json`: overall mAP50, mAP50-95, and
**per-class AP50**. Watch the rare classes (the score risk):
**4 Pickup, 7 Mini Truck, 8 Agro Use, 11 Heavy Truck, 12 Trailer.**
Fold balance is exact (137/46/31/56 imgs per camera per fold; every class present in every
val fold, Agro Use 27–29/fold) — see `reports/`. *(Numbers fill in after the Kaggle GPU run.)*

## 4. Inference procedure
`rtdetr_infer.py`: predict each fold's `best.pt` on the 327 test images at **conf 0.001**
(keep recall high — mAP integrates the whole PR curve; do **not** threshold at 0.5 like the
sample), max_det 300 → save per-fold caches → **WBF** (iou 0.55, skip 0.001, conf_type avg)
→ `submission_rtdetr.csv`. Caches also feed the cross-track grand ensemble.

## 5. Submission file (format is LOCKED to the real `sample_submission.csv`)
- header `image_id,PredictionString`; **exactly one row per test image (327)**.
- ⚠️ **`image_id` keeps the `.txt` extension** (e.g. `...000061.txt`), NOT a bare stem.
  The prompt said "bare stem" — that is wrong vs the real sample; matched the sample.
- `PredictionString` = space-joined `class conf cx cy w h`, normalized coords,
  **conf on a 0–100 scale** (matches sample; mAP is rank-based so scale is harmless).
- Images with no detections → empty PredictionString (still one row).
- Validated locally against `sample_submission.csv` (all sample ids covered, 327 unique rows).

## 6. Reproducibility
1. **Local prep (CPU, done):** `python track2/scripts/build_dataset.py --src data --out track2/yolo_ds --k 3`
   → YOLO labels, `folds.csv`, `data_fold{0,1,2}.yaml`. Deterministic (frame-rank % k).
2. **Kaggle train:** set `SRC`, run `rtdetr_train.py` (seed 42).
3. **Kaggle infer:** run `rtdetr_infer.py` → `submission_rtdetr.csv` + `caches/*.json`.
4. **Grand ensemble:** `ensemble_tracks.py` fuses Track#1 ⊕ Track#2 (accepts each track's
   `.csv` or `.json`; weight the higher-CV track ~1.2). This is the leaderboard submission.

---

## Data audit summary (see `reports/audit_summary.json`, `reports/annotation_check.png`)
**Labels are pristine → no cleaning performed** (would only waste time):
0 invalid / out-of-bounds / zero-area / duplicate / near-duplicate / extreme-AR boxes;
0 missing or background images. All 810 imgs are 1280×720.
- Small objects matter (7.1% of boxes < 0.001 area ≈ 30 px) → keep resolution high.
- **Class imbalance is the real risk:** Sedan 2480, Tempu 2032 … vs Agro Use **83**,
  Mini Truck 127, Mini Bus 164. Rare-class AP50 is where the ensemble must hold up.
- Cameras are **CCTV01/02/07/10** (not "01–04" as the brief stated); test draws from the
  same 4 cameras with interleaved frames → the frame-phase split is the correct validation.

## Files
```
track2/
  README.md                      this report
  scripts/
    audit.py                     data-quality audit            -> reports/audit_summary.json
    build_dataset.py             CSV->YOLO + K-fold split      -> yolo_ds/
    visualize.py                 annotation sanity images      -> reports/annotation_check.png
    submission_utils.py          cache + self-contained WBF + submission writer (no deps)
    rtdetr_lib.py                single source of truth: build/train/infer/submit functions
    ensemble_tracks.py           cross-track grand WBF ensemble
  notebooks/
    kaggle_rtdetr_track2.ipynb   ** MAIN ** Kaggle notebook (clone->smoke test->train->submit)
    rtdetr_train.py              CLI wrapper (terminal alternative to the notebook)
    rtdetr_infer.py              CLI wrapper (terminal alternative to the notebook)
  yolo_ds/                       built dataset (labels, folds, yamls)
  reports/                       audit json + annotation_check.png
```

"""Self-contained prediction-cache + WBF + submission writer.
No external deps (works offline on Kaggle). All boxes normalized YOLO (cx,cy,w,h).
Cache schema (JSON): { "<stem>.txt": [[cls,conf,cx,cy,w,h], ...], ... }
conf stored in 0-1.  Submission writes conf*conf_scale (default 100, matching sample).
"""
import json, os, glob, numpy as np

# ---------- IO ----------
def list_test_ids(test_img_dir):
    """Return sorted list of submission image_ids (stem + '.txt')."""
    stems=[os.path.splitext(os.path.basename(p))[0]
           for p in glob.glob(os.path.join(test_img_dir,"*.jpg"))]
    return sorted(f"{s}.txt" for s in stems)

def save_cache(cache,path):
    os.makedirs(os.path.dirname(os.path.abspath(path)),exist_ok=True)
    json.dump({k:[[round(float(x),6) for x in b] for b in v] for k,v in cache.items()},
              open(path,"w"))

def load_cache(path): return json.load(open(path))

def submission_to_cache(csv_path, conf_scale=100.0):
    """Ingest any track's submission.csv -> cache (conf back to 0-1).
    Lets us WBF-fuse Track#1 even if it only emits a CSV."""
    import csv
    cache={}
    for row in csv.reader(open(csv_path)):
        if not row or row[0]=="image_id": continue
        iid=row[0]; ps=(row[1] if len(row)>1 else "").split()
        dets=[]
        for i in range(0,len(ps)-5,6):
            cls=int(float(ps[i])); conf=float(ps[i+1])/conf_scale
            cx,cy,w,h=map(float,ps[i+2:i+6])
            dets.append([cls,conf,cx,cy,w,h])
        cache[iid]=dets
    return cache

# ---------- geometry ----------
def yolo_to_xyxy(b):  # cx,cy,w,h -> x1,y1,x2,y2
    cx,cy,w,h=b; return [cx-w/2,cy-h/2,cx+w/2,cy+h/2]
def xyxy_to_yolo(b):
    x1,y1,x2,y2=b; return [(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1]
def iou_xyxy(a,b):
    xa=max(a[0],b[0]);ya=max(a[1],b[1]);xb=min(a[2],b[2]);yb=min(a[3],b[3])
    iw=max(0.,xb-xa);ih=max(0.,yb-ya);inter=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.

# ---------- WBF (self-contained, per class) ----------
def wbf_image(model_dets, weights, iou_thr=0.55, skip_box_thr=0.001, conf_type="avg"):
    """model_dets: list over models; each is list of [cls,conf,cx,cy,w,h] (conf 0-1).
    Returns fused list of [cls,conf,cx,cy,w,h]."""
    M=len(model_dets); w=weights if weights is not None else [1.0]*M
    # gather, filter, group by class
    by_cls={}
    for mi,dets in enumerate(model_dets):
        for d in dets:
            cls=int(d[0]); conf=float(d[1])
            if conf<skip_box_thr: continue
            by_cls.setdefault(cls,[]).append((iou_to:=yolo_to_xyxy(d[2:6]),conf,w[mi]))
    out=[]
    for cls,items in by_cls.items():
        # sort by conf desc
        items=sorted(items,key=lambda t:-t[1])
        clusters=[]  # each: dict(box,confs,weights,boxes_weighted_sum)
        for box,conf,wt in items:
            best=-1;best_iou=iou_thr
            for ci,cl in enumerate(clusters):
                i=iou_xyxy(box,cl["fbox"])
                if i>best_iou: best_iou=i;best=ci
            if best<0:
                clusters.append({"boxes":[box],"confs":[conf],"ws":[wt],
                                 "wsum":[c*conf for c in box] if False else None})
                cl=clusters[-1]; cl["num_sum"]=[wt*conf*c for c in box]; cl["den"]=wt*conf
                cl["fbox"]=box[:]
            else:
                cl=clusters[best]
                cl["boxes"].append(box);cl["confs"].append(conf);cl["ws"].append(wt)
                cl["num_sum"]=[ns+wt*conf*c for ns,c in zip(cl["num_sum"],box)]
                cl["den"]+=wt*conf
                cl["fbox"]=[ns/cl["den"] for ns in cl["num_sum"]]
        for cl in clusters:
            sw=sum(cl["ws"])
            if conf_type=="avg":
                fconf=sum(c*wt for c,wt in zip(cl["confs"],cl["ws"]))/sum(cl["ws"])
                fconf=fconf*min(M,len(cl["confs"]))/M      # reward model agreement
            elif conf_type=="max":
                fconf=max(cl["confs"])
            else:  # box_and_model_avg-ish
                fconf=sum(cl["confs"])/M
            out.append([cls,float(fconf)]+xyxy_to_yolo(cl["fbox"]))
    out.sort(key=lambda d:-d[1])
    return out

def fuse_caches(caches, weights=None, iou_thr=0.55, skip_box_thr=0.001, conf_type="avg"):
    """caches: list of cache dicts. Returns fused cache over union of image_ids."""
    ids=set()
    for c in caches: ids|=set(c.keys())
    fused={}
    for iid in ids:
        md=[c.get(iid,[]) for c in caches]
        fused[iid]=wbf_image(md,weights,iou_thr,skip_box_thr,conf_type)
    return fused

# ---------- submission ----------
def write_submission(cache, test_img_dir, out_csv, conf_scale=100.0,
                     conf_thr=0.001, max_det=300):
    ids=list_test_ids(test_img_dir)               # ensures one row per test image
    rows=["image_id,PredictionString"]
    for iid in ids:
        dets=sorted(cache.get(iid,[]),key=lambda d:-d[1])[:max_det]
        toks=[]
        for cls,conf,cx,cy,w,h in dets:
            if conf<conf_thr: continue
            cx=min(max(cx,0),1);cy=min(max(cy,0),1)
            w=min(max(w,1e-6),1);h=min(max(h,1e-6),1)
            toks.append(f"{int(cls)} {conf*conf_scale:.2f} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        rows.append(f"{iid},{' '.join(toks)}")
    d=os.path.dirname(os.path.abspath(out_csv)); os.makedirs(d,exist_ok=True)
    open(out_csv,"w").write("\n".join(rows)+"\n")
    return len(ids)

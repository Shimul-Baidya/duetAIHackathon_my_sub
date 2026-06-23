import pandas as pd, numpy as np, os, json, collections
from PIL import Image
ROOT="data"; TRAIN_IMG=f"{ROOT}/train/images"
df=pd.read_csv(f"{ROOT}/train/train.csv")
NAMES=["Rickshaw","Motorcycle","Tempu","Sedan Car","Pickup","Microbus","Mini Bus","Mini Truck","Agro Use","Medium Truck","Large Bus","Heavy Truck","Trailer"]
print("rows",len(df),"unique image_ids",df.image_id.nunique())
df["stem"]=df.image_id.str.replace(".txt","",regex=False)
df["jpg"]=df.stem+".jpg"
df["cam"]=df.stem.str.split("^").str[0]
df["frame"]=df.stem.str.extract(r"_(\d+)$").astype(int)

# image existence + sizes
imgs=sorted(os.listdir(TRAIN_IMG))
img_set=set(imgs)
missing=sorted(set(df.jpg)-img_set)
no_label=sorted(img_set-set(df.jpg))
print("images on disk",len(imgs),"| labeled images",df.jpg.nunique(),
      "| label refs missing img",len(missing),"| imgs w/o any label(background)",len(no_label))
# sample sizes per camera
sizes={}
for cam,g in df.groupby("cam"):
    f=g.jpg.iloc[0]
    sizes[cam]=Image.open(f"{TRAIN_IMG}/{f}").size
print("img size by cam (WxH):",sizes)

# ---- box validity ----
x,y,w,h=df.x_center,df.y_center,df.width,df.height
df["x1"]=x-w/2; df["y1"]=y-h/2; df["x2"]=x+w/2; df["y2"]=y+h/2
df["area"]=w*h
df["ar"]=w/h
oob_center=((x<0)|(x>1)|(y<0)|(y>1)).sum()
nonpos=((w<=0)|(h<=0)).sum()
out_left=(df.x1<-1e-6).sum(); out_top=(df.y1<-1e-6).sum()
out_right=(df.x2>1+1e-6).sum(); out_bot=(df.y2>1+1e-6).sum()
zero_area=(df.area<=0).sum()
print(f"\n[VALIDITY] center-OOB={oob_center} nonpos_wh={nonpos} zero_area={zero_area}")
print(f"  box exceeds border: left={out_left} top={out_top} right={out_right} bottom={out_bot} (total rows w/ any overflow={((df.x1<-1e-6)|(df.y1<-1e-6)|(df.x2>1+1e-6)|(df.y2>1+1e-6)).sum()})")
overflow=df[(df.x1<-1e-6)|(df.y1<-1e-6)|(df.x2>1+1e-6)|(df.y2>1+1e-6)]
# how severe is overflow?
ov_amt=pd.concat([-df.x1,-df.y1,df.x2-1,df.y2-1],axis=1).max(axis=1)
print("  overflow magnitude quantiles (frac of img):",
      {q:round(float(ov_amt[ov_amt>1e-6].quantile(q)),4) for q in [.5,.9,.99,1.0]} if (ov_amt>1e-6).any() else "none")

# ---- duplicates ----
exact=df.duplicated(subset=["image_id","class_id","x_center","y_center","width","height"]).sum()
print(f"\n[DUPLICATES] exact-identical rows={exact}")
# near-duplicate: same image+class, IoU>0.9
def iou(a,b):
    xa=max(a[0],b[0]);ya=max(a[1],b[1]);xb=min(a[2],b[2]);yb=min(a[3],b[3])
    iw=max(0,xb-xa);ih=max(0,yb-ya);inter=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0
near=0
for (img,cls),g in df.groupby(["image_id","class_id"]):
    if len(g)<2: continue
    b=g[["x1","y1","x2","y2"]].values
    for i in range(len(b)):
        for j in range(i+1,len(b)):
            if iou(b[i],b[j])>0.9: near+=1
print(f"[DUPLICATES] near-dup pairs (same img+class, IoU>0.9)={near}")

# ---- outliers ----
print(f"\n[SIZE] area quantiles:",{q:round(float(df.area.quantile(q)),5) for q in [0,.01,.05,.5,.95,.99,1.0]})
print(f"[SIZE] tiny boxes area<0.0005 (<~28px@1280): {(df.area<0.0005).sum()} ({100*(df.area<0.0005).mean():.1f}%)")
print(f"[SIZE] tiny boxes area<0.001: {(df.area<0.001).sum()} ({100*(df.area<0.001).mean():.1f}%)")
print(f"[SIZE] huge boxes area>0.25: {(df.area>0.25).sum()}; area>0.5: {(df.area>0.5).sum()}")
print(f"[SIZE] aspect ratio quantiles:",{q:round(float(df.ar.quantile(q)),3) for q in [0,.01,.5,.99,1.0]})
print(f"[SIZE] extreme AR (>6 or <1/6): {((df.ar>6)|(df.ar<1/6)).sum()}")

# ---- class distribution ----
print("\n[CLASS DISTRIBUTION] (instances | images containing)")
ic=df.class_id.value_counts().sort_index()
imgc=df.groupby("class_id").image_id.nunique()
for c in range(13):
    n=int(ic.get(c,0)); ni=int(imgc.get(c,0))
    print(f"  {c:2d} {NAMES[c]:13s} inst={n:5d} ({100*n/len(df):4.1f}%)  imgs={ni:4d}")

# ---- boxes per image ----
bpi=df.groupby("image_id").size()
print(f"\n[DENSITY] boxes/image: mean={bpi.mean():.1f} median={bpi.median():.0f} max={bpi.max()} ; imgs>30 boxes={ (bpi>30).sum() }")
print("  top crowded:",list(bpi.sort_values(ascending=False).head(5).items()))

# ---- frames per camera (for split) ----
print("\n[FRAMES per camera]")
for cam,g in df.groupby("cam"):
    fr=sorted(g.frame.unique())
    print(f"  {cam}: {g.jpg.nunique()} imgs, frame range {fr[0]}..{fr[-1]}")

# save cleaned-candidate report
os.makedirs("track2/reports",exist_ok=True)
rep={"rows":int(len(df)),"images":int(df.jpg.nunique()),
     "center_oob":int(oob_center),"nonpos_wh":int(nonpos),"zero_area":int(zero_area),
     "border_overflow_rows":int(len(overflow)),"exact_dup":int(exact),"near_dup_pairs":int(near),
     "tiny_lt0.0005":int((df.area<0.0005).sum()),"huge_gt0.25":int((df.area>0.25).sum()),
     "img_size_by_cam":{k:list(v) for k,v in sizes.items()},
     "class_instances":{NAMES[c]:int(ic.get(c,0)) for c in range(13)},
     "background_images":int(len(no_label))}
json.dump(rep,open("track2/reports/audit_summary.json","w"),indent=2)
df.to_parquet("track2/reports/train_parsed.parquet") if False else None
print("\nsaved -> track2/reports/audit_summary.json")

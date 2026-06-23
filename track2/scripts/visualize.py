import pandas as pd, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.patches as P
from PIL import Image
NAMES=["Rickshaw","Motorcycle","Tempu","Sedan Car","Pickup","Microbus","Mini Bus",
       "Mini Truck","Agro Use","Medium Truck","Large Bus","Heavy Truck","Trailer"]
df=pd.read_csv("data/train/train.csv"); df["stem"]=df.image_id.str.replace(".txt","",regex=False)
df["area"]=df.width*df.height
IMG="data/train/images"
def draw(ax,stem,title):
    im=Image.open(f"{IMG}/{stem}.jpg"); W,H=im.size; ax.imshow(im)
    for r in df[df.stem==stem].itertuples():
        x=(r.x_center-r.width/2)*W; y=(r.y_center-r.height/2)*H
        ax.add_patch(P.Rectangle((x,y),r.width*W,r.height*H,fill=False,ec="lime",lw=1))
        ax.text(x,y-2,NAMES[int(r.class_id)],color="yellow",fontsize=5)
    ax.set_title(title,fontsize=8); ax.axis("off")
bpi=df.groupby("stem").size()
crowded=bpi.idxmax()
tiny=df.loc[df.area.idxmin(),"stem"]
agro=df[df.class_id==8].stem.iloc[0]
htruck=df[df.class_id==11].stem.iloc[0]
fig,ax=plt.subplots(2,2,figsize=(14,8))
draw(ax[0,0],crowded,f"Most crowded: {bpi.max()} boxes")
draw(ax[0,1],tiny,f"Smallest box (area={df.area.min():.5f})")
draw(ax[1,0],agro,"Contains rare class 'Agro Use'(8)")
draw(ax[1,1],htruck,"Contains rare class 'Heavy Truck'(11)")
plt.tight_layout(); plt.savefig("track2/reports/annotation_check.png",dpi=130)
print("saved track2/reports/annotation_check.png | crowded=",crowded)

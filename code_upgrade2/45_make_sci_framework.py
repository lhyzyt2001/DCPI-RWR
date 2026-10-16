from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

NAVY="#2F5D7C"; BLUE="#3B82B4"; TEAL="#2A9D8F"; AMBER="#E9A03B"; ORANGE="#D66A2C"; VIOLET="#7E6DB0"
INK="#1E2A33"; MID="#66727E"; LINE="#AAB4BE"; BG="#F5F7F9"; WHITE="#FFFFFF"

def card(ax,x,y,w,h,fc=WHITE,ec="#D7DEE5",lw=.8,r=.016,z=1):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0.006,rounding_size={r}",facecolor=fc,edgecolor=ec,linewidth=lw,zorder=z)
    ax.add_patch(p); return p

def label(ax,x,y,s,size=7,weight="normal",color=INK,ha="center",va="center",z=6):
    return ax.text(x,y,s,fontsize=size,fontweight=weight,color=color,ha=ha,va=va,zorder=z)

def arrow(ax,x1,y1,x2,y2,color=AMBER,lw=1.25,style="-|>",ms=9,z=5):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=ms,color=color,linewidth=lw,zorder=z))

def stage(ax,n,x,title,color):
    ax.add_patch(Circle((x,.922),.017,facecolor=color,edgecolor="none",zorder=5))
    label(ax,x,.922,str(n),size=6.6,weight="bold",color="white")
    label(ax,x,.870,title,size=7.7,weight="bold",color=INK)

def network(ax,cx,cy,s,color):
    pts=[(cx-.33*s,cy),(cx-.12*s,cy+.26*s),(cx+.17*s,cy+.18*s),(cx+.33*s,cy-.10*s),(cx+.02*s,cy-.27*s),(cx-.25*s,cy-.19*s)]
    for a,b in [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(1,4),(2,4)]:
        ax.plot([pts[a][0],pts[b][0]],[pts[a][1],pts[b][1]],color="#95A0AA",lw=.55,zorder=2)
    for i,(px,py) in enumerate(pts):
        ax.add_patch(Circle((px,py),.058*s,facecolor=color if i in (0,2) else WHITE,edgecolor=INK,linewidth=.45,zorder=3))

def inputs(ax):
    x,y,w,h=.030,.285,.185,.525; card(ax,x,y,w,h)
    # Release snapshots
    sx,sy=x+.039,y+.380
    for i,col in enumerate(["#B9DDF0",BLUE,NAVY]):
        ax.add_patch(Rectangle((sx+i*.009,sy+i*.009),.060,.052,facecolor=col,edgecolor=WHITE,linewidth=.4,zorder=2+i))
    label(ax,x+w/2,y+.348,"Open Targets",size=7.1,weight="bold")
    card(ax,x+.025,y+.286,.061,.050,fc="#EDF5FA",ec="#C8DDEA",lw=.55,r=.008)
    label(ax,x+.0555,y+.318,"26.03",size=5.7,weight="bold",color=NAVY)
    label(ax,x+.0555,y+.298,"historical",size=4.7,color=MID)
    arrow(ax,x+.091,y+.311,x+.108,y+.311,color=LINE,lw=.75,ms=6)
    card(ax,x+.113,y+.286,.061,.050,fc="#FCF1EA",ec="#E9C8B5",lw=.55,r=.008)
    label(ax,x+.1435,y+.318,"26.06",size=5.7,weight="bold",color=ORANGE)
    label(ax,x+.1435,y+.298,"incident",size=4.7,color=MID)
    # Network views
    xs=[x+.044,x+.093,x+.142]
    for xx,col,name in zip(xs,[BLUE,TEAL,AMBER],["STRING","TISCH","KEGG"]):
        network(ax,xx,y+.235,.052,col); label(ax,xx,y+.190,name,size=5.6)
    label(ax,x+w/2,y+.134,"Network views",size=6.5,weight="bold",color=MID)
    label(ax,x+w/2,y+.088,"Six cancer-specific",size=6.1,color=INK)
    label(ax,x+w/2,y+.061,"historical seed sets",size=6.1,color=INK)

def diffusion(ax):
    x,y,w,h=.255,.285,.200,.525; card(ax,x,y,w,h)
    lanes=[("STRING",BLUE,"P_STRING",y+.380),("Cell-type",TEAL,"P_cell",y+.255),("Pathway",AMBER,"P_path",y+.130)]
    for name,col,pmat,yy in lanes:
        network(ax,x+.035,yy,.040,col)
        label(ax,x+.063,yy,name,size=5.4,weight="bold",color=col,ha="left")
        arrow(ax,x+.121,yy,x+.134,yy,color=col,lw=.85,ms=7)
        # Transition-matrix glyph rather than a fitted model block.
        gx=x+.140; gy=yy-.024; cell=.012
        for rr in range(3):
            for cc in range(3):
                fill=col if (rr+cc)%3==0 else "#F1F4F6"
                ax.add_patch(Rectangle((gx+cc*cell,gy+rr*cell),cell-.001,cell-.001,facecolor=fill,edgecolor="white",linewidth=.25,zorder=3))
        label(ax,x+.158,yy-.043,pmat,size=4.5,color=MID)

def integration(ax):
    x,y,w,h=.495,.285,.235,.525; card(ax,x,y,w,h)
    # Reference diffusion used in every comparison.
    card(ax,x+.018,y+.420,.199,.052,fc="#EDF5FA",ec=BLUE,lw=.65,r=.010)
    label(ax,x+.117,y+.446,"Reference: STRING-RWR",size=6.1,weight="bold",color=BLUE)
    # DCPI-RWR: separate layer diffusion followed by degree conditioning and rank fusion.
    label(ax,x+.022,y+.382,"DCPI-RWR",size=7.0,weight="bold",color=TEAL,ha="left")
    card(ax,x+.018,y+.292,.120,.068,fc="#EEF8F5",ec=TEAL,lw=.72,r=.010)
    label(ax,x+.078,y+.331,"Layer-wise RWR +",size=5.2,weight="bold",color=TEAL)
    label(ax,x+.078,y+.309,"degree-bin percentile",size=5.2,color=INK)
    arrow(ax,x+.142,y+.326,x+.157,y+.326,color=TEAL,lw=.8,ms=7)
    card(ax,x+.160,y+.292,.057,.068,fc="#EEF8F5",ec=TEAL,lw=.72,r=.010)
    label(ax,x+.188,y+.331,"Weighted",size=5.1,weight="bold",color=TEAL)
    label(ax,x+.188,y+.309,"rank fusion",size=5.1,color=INK)
    ax.plot([x+.018,x+w-.018],[y+.260,y+.260],color="#E2E7EC",lw=.7,zorder=2)
    # Parallel joint multiplex strategy.
    label(ax,x+.022,y+.226,"Cancer-adaptive supra-RWR",size=6.2,weight="bold",color=ORANGE,ha="left")
    for j,col in enumerate([BLUE,TEAL,AMBER]):
        yy=y+.169-j*.027
        ax.add_patch(Rectangle((x+.030,yy),.062,.013,facecolor=col,edgecolor="none",alpha=.9,zorder=3))
    arrow(ax,x+.111,y+.106,x+.111,y+.190,color=VIOLET,lw=.85,style="<->",ms=8)
    card(ax,x+.137,y+.117,.080,.061,fc="#FBF7EE",ec=ORANGE,lw=.68,r=.009)
    label(ax,x+.177,y+.151,"Joint supra-RWR",size=5.1,weight="bold",color=ORANGE)
    label(ax,x+.177,y+.130,"layer switching",size=4.9,color=INK)
    card(ax,x+.112,y+.061,.105,.045,fc="#FFF9EE",ec="#E9D8AF",lw=.55,r=.008)
    label(ax,x+.1645,y+.0835,"Evidence-weighted\nrestart (optional)",size=4.45,weight="bold",color="#7A5510")

def evaluation(ax):
    x,y,w,h=.770,.285,.200,.525; card(ax,x,y,w,h)
    # Locked release sequence
    label(ax,x+w/2,y+.442,"Locked cross-release evaluation",size=6.1,weight="bold",color=MID)
    nodes=[(x+.040,BLUE,"26.03","Tune"),(x+.100,AMBER,"LOCK","Freeze"),(x+.160,ORANGE,"26.06","Test")]
    for cx,col,big,small in nodes:
        ax.add_patch(Circle((cx,y+.373),.020,facecolor=col,edgecolor="white",linewidth=.7,zorder=4))
        label(ax,cx,y+.373,big,size=4.7,weight="bold",color="white")
        label(ax,cx,y+.335,small,size=5.3,color=MID)
    arrow(ax,x+.062,y+.373,x+.077,y+.373,color=LINE,lw=.8,ms=6)
    arrow(ax,x+.122,y+.373,x+.137,y+.373,color=LINE,lw=.8,ms=6)
    # Ranking glyph
    card(ax,x+.025,y+.132,.150,.145,fc="#F7F4FC",ec="#D9D1EB",lw=.7,r=.012)
    for i,ww in enumerate([.102,.083,.066,.050]):
        yy=y+.240-i*.024
        ax.add_patch(Rectangle((x+.049,yy),ww,.011,facecolor=VIOLET,edgecolor="none",alpha=.92-i*.12,zorder=3))
    label(ax,x+.100,y+.153,"Cancer-specific rankings",size=5.9,weight="bold",color=VIOLET)
    metrics=[(x+.021,.052,"AUPRC",BLUE),(x+.075,.052,"Top-100",TEAL),(x+.129,.050,"Top-100\nhub fraction",ORANGE)]
    for xx,ww,lab,col in metrics:
        card(ax,xx,y+.045,ww,.050,fc="#FFFFFF",ec=col,lw=.62,r=.008)
        label(ax,xx+ww/2,y+.070,lab,size=4.8,weight="bold",color=col)

def footer(ax):
    x,y,w,h=.030,.070,.940,.135; card(ax,x,y,w,h,fc="#F7F9FB",ec="#E0E6EB",lw=.7,r=.016)
    label(ax,x+.022,y+.090,"Robustness &\ninterpretation",size=6.5,weight="bold",color=NAVY,ha="left")
    items=[("Degree\nmatching",BLUE),("Degree\nstrata",TEAL),("Hub removal /\npermutation",ORANGE),("Gene-cluster\nbootstrap",VIOLET),("Context layer +\nTCGA expression",AMBER)]
    start=x+.203; bw=.135; gap=.010
    for i,(name,col) in enumerate(items):
        xx=start+i*(bw+gap)
        card(ax,xx,y+.031,bw,.074,fc=WHITE,ec=col,lw=.62,r=.010)
        ax.add_patch(Rectangle((xx+.010,y+.094),bw-.020,.004,facecolor=col,edgecolor="none",zorder=4))
        label(ax,xx+bw/2,y+.066,name,size=5.1,weight="bold",color=INK)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"font.size":7,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})
    fig,ax=plt.subplots(figsize=(7.2,3.55)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    stage(ax,1,.123,"Evidence & networks",NAVY); stage(ax,2,.355,"Network layers",BLUE); stage(ax,3,.612,"Diffusion strategies",TEAL); stage(ax,4,.870,"Locked evaluation",ORANGE)
    inputs(ax); diffusion(ax); integration(ax); evaluation(ax); footer(ax)
    arrow(ax,.220,.548,.250,.548,color=AMBER,lw=1.2); arrow(ax,.460,.548,.490,.548,color=AMBER,lw=1.2); arrow(ax,.735,.548,.765,.548,color=AMBER,lw=1.2)
    fig.subplots_adjust(left=.008,right=.992,bottom=.014,top=.992)
    base=a.output/"Figure_SCI_DCPI_RWR_framework"
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    overflow=[]
    for t in ax.texts:
        b=t.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        if b.x0 < 0 or b.y0 < 0 or b.x1 > 1 or b.y1 > 1:
            overflow.append({"text":t.get_text(),"bbox":[round(b.x0,4),round(b.y0,4),round(b.x1,4),round(b.y1,4)]})
    (a.output/"Figure_SCI_DCPI_RWR_framework_layout_QA.json").write_text(
        json.dumps({"canvas_inches":[7.2,3.55],"global_text_overflow_count":len(overflow),"global_text_overflow":overflow},indent=2),
        encoding="utf-8")
    fig.savefig(base.with_suffix(".pdf"),bbox_inches="tight"); fig.savefig(base.with_suffix(".svg"),bbox_inches="tight"); fig.savefig(base.with_suffix(".png"),dpi=600,bbox_inches="tight"); plt.close(fig)

if __name__=="__main__": main()

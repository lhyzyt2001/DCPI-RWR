from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


BLUE = "#2C7FB8"
TEAL = "#1B9E77"
ORANGE = "#D95F02"
PURPLE = "#756BB1"
GOLD = "#E6A700"
DARK = "#222222"
MID = "#5B6573"
LIGHT = "#F7F8FA"
LINE = "#8A94A3"


def rounded(ax, x, y, w, h, *, fc="white", ec=LINE, lw=1.0, ls="-", radius=0.015, z=1):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=z,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, *, size=7.2, weight="normal", color=DARK, ha="center", va="center", z=5):
    ax.text(x, y, text, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=z)


def arrow(ax, x1, y1, x2, y2, *, color=GOLD, lw=1.4, style="-|>", connection="arc3"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=9, color=color, linewidth=lw,
                                 connectionstyle=connection, zorder=4))


def panel(ax, x, y, w, h, letter, title, *, dashed=False):
    rounded(ax, x, y, w, h, fc="white", ec="#52606D", lw=0.9, ls="--" if dashed else "-", radius=0.012)
    label(ax, x + 0.014, y + h - 0.035, letter, size=9.0, weight="bold", ha="left")
    label(ax, x + 0.040, y + h - 0.035, title, size=8.0, weight="bold", ha="left")


def network_icon(ax, cx, cy, scale, color):
    pts = [(cx - .34*scale, cy), (cx - .10*scale, cy + .28*scale),
           (cx + .20*scale, cy + .15*scale), (cx + .35*scale, cy - .18*scale),
           (cx - .05*scale, cy - .28*scale)]
    edges = [(0,1),(1,2),(2,3),(3,4),(4,0),(1,4),(2,4)]
    for a,b in edges:
        ax.plot([pts[a][0],pts[b][0]],[pts[a][1],pts[b][1]],color="#7D8792",lw=.7,zorder=3)
    for i,(px,py) in enumerate(pts):
        ax.add_patch(Circle((px,py), .07*scale, facecolor=color if i in {0,2} else "white",
                            edgecolor=DARK, linewidth=.6,zorder=4))


def draw_inputs(ax):
    x,y,w,h = .025,.51,.205,.445
    panel(ax,x,y,w,h,"a","Data and network views")
    # Open Targets snapshot cards
    rounded(ax,x+.018,y+.285,w-.036,.095,fc="#EAF2F8",ec=BLUE,lw=.9)
    label(ax,x+.055,y+.335,"26.03",size=7.3,weight="bold",color=BLUE)
    label(ax,x+.122,y+.335,"Historical seeds",size=7.0,ha="left")
    label(ax,x+.055,y+.305,"26.06",size=7.3,weight="bold",color=ORANGE)
    label(ax,x+.122,y+.305,"Incident evidence",size=7.0,ha="left")
    # network icons
    network_icon(ax,x+.052,y+.205,.075,BLUE)
    network_icon(ax,x+.108,y+.205,.075,TEAL)
    network_icon(ax,x+.164,y+.205,.075,ORANGE)
    label(ax,x+.052,y+.145,"STRING",size=6.5)
    label(ax,x+.108,y+.145,"TISCH",size=6.5)
    label(ax,x+.164,y+.145,"KEGG",size=6.5)
    # cancer chips
    label(ax,x+.020,y+.105,"Cancer cohorts",size=6.6,weight="bold",ha="left",color=MID)
    chips=["LUAD","BRCA","CRC","PRAD","MEL","OV"]
    for i,name in enumerate(chips):
        cx=x+.037+(i%3)*.061; cy=y+.066-(i//3)*.035
        rounded(ax,cx-.025,cy-.012,.050,.024,fc="#F3F5F7",ec="#A5ADB7",lw=.55,radius=.006)
        label(ax,cx,cy,name,size=5.8)


def draw_models(ax):
    x,y,w,h=.275,.51,.255,.445
    panel(ax,x,y,w,h,"b","Network-diffusion models")
    # baseline
    rounded(ax,x+.020,y+.290,w-.040,.075,fc="#EAF2F8",ec=BLUE,lw=1.0)
    network_icon(ax,x+.055,y+.327,.045,BLUE)
    label(ax,x+.090,y+.337,"STRING-RWR",size=7.4,weight="bold",ha="left",color=BLUE)
    label(ax,x+.090,y+.313,"Performance baseline",size=6.2,ha="left",color=MID)
    # core
    rounded(ax,x+.020,y+.165,w-.040,.100,fc="#EAF7F2",ec=TEAL,lw=1.0)
    label(ax,x+.035,y+.238,"DCPI-RWR",size=7.4,weight="bold",ha="left",color=TEAL)
    labels=["RWR","Degree-bin\npercentile","Rank\nfusion"]
    xs=[x+.058,x+.132,x+.205]
    for cx,txt in zip(xs,labels):
        rounded(ax,cx-.029,y+.185,.058,.037,fc="white",ec=TEAL,lw=.65,radius=.006)
        label(ax,cx,y+.203,txt,size=5.6)
    arrow(ax,x+.088,y+.203,x+.101,y+.203,color=TEAL,lw=.9)
    arrow(ax,x+.161,y+.203,x+.175,y+.203,color=TEAL,lw=.9)
    # supra extension
    rounded(ax,x+.020,y+.040,w-.040,.100,fc="#FFF4E8",ec=ORANGE,lw=1.0)
    label(ax,x+.035,y+.113,"Adaptive supra-RWR",size=7.2,weight="bold",ha="left",color=ORANGE)
    for j,(txt,col) in enumerate([("STRING",BLUE),("Cell",TEAL),("Pathway",ORANGE)]):
        yy=y+.059+j*.017
        ax.add_patch(Rectangle((x+.040,yy),.060,.011,facecolor=col,edgecolor="none",alpha=.82,zorder=3))
        label(ax,x+.107,yy+.006,txt,size=5.4,ha="left",color=MID)
    arrow(ax,x+.146,y+.068,x+.146,y+.101,color=PURPLE,lw=.9,style="<->")
    label(ax,x+.160,y+.086,"Layer switching",size=5.8,ha="left",color=PURPLE)
    rounded(ax,x+.150,y+.047,.075,.021,fc="white",ec=ORANGE,lw=.55,radius=.004)
    label(ax,x+.188,y+.058,"Evidence weights",size=5.3)


def draw_evaluation(ax):
    x,y,w,h=.575,.51,.185,.445
    panel(ax,x,y,w,h,"c","Locked evaluation")
    # timeline
    ax.plot([x+.046,x+.046],[y+.105,y+.330],color="#9AA3AD",lw=1.3,zorder=2)
    for cy,col in [(y+.315,BLUE),(y+.215,GOLD),(y+.115,ORANGE)]:
        ax.add_patch(Circle((x+.046,cy),.010,facecolor=col,edgecolor="white",linewidth=.6,zorder=4))
    label(ax,x+.070,y+.318,"Tune on 26.03",size=7.0,weight="bold",ha="left",color=BLUE)
    label(ax,x+.070,y+.290,"Historical labels only",size=6.0,ha="left",color=MID)
    label(ax,x+.070,y+.218,"Lock model",size=7.0,weight="bold",ha="left",color=DARK)
    label(ax,x+.070,y+.190,"No future-label tuning",size=6.0,ha="left",color=MID)
    label(ax,x+.070,y+.118,"Test on 26.06",size=7.0,weight="bold",ha="left",color=ORANGE)
    label(ax,x+.070,y+.090,"Incident genes",size=6.0,ha="left",color=MID)
    # metrics
    for i,(txt,col) in enumerate([("AUPRC",BLUE),("Top-100 hits",TEAL),("Hub fraction",ORANGE)]):
        cx=x+.040+i*.052
        rounded(ax,cx-.023,y+.035,.046,.027,fc="#F7F8FA",ec=col,lw=.7,radius=.005)
        label(ax,cx,y+.049,txt,size=5.3)


def draw_outputs(ax):
    x,y,w,h=.805,.51,.170,.445
    panel(ax,x,y,w,h,"d","Study outputs")
    # Pareto mini-plot
    rounded(ax,x+.018,y+.225,w-.036,.135,fc="#FAFAFB",ec="#AAB2BC",lw=.7)
    ax.plot([x+.043,x+.043,x+.135],[y+.250,y+.335,y+.335],color="#7E8791",lw=.7,zorder=3)
    pts=[(x+.060,y+.315,BLUE),(x+.098,y+.288,ORANGE),(x+.078,y+.270,TEAL)]
    for px,py,col in pts:
        ax.add_patch(Circle((px,py),.008,facecolor=col,edgecolor="white",linewidth=.4,zorder=4))
    label(ax,x+.089,y+.240,"Performance–bias trade-off",size=6.1,weight="bold")
    # ranking
    rounded(ax,x+.018,y+.075,w-.036,.120,fc="#F3F7FB",ec=PURPLE,lw=.8)
    for i,width in enumerate([.095,.078,.062,.050]):
        yy=y+.165-i*.020
        ax.add_patch(Rectangle((x+.040,yy),width,.010,facecolor=PURPLE,alpha=.80-.12*i,edgecolor="none",zorder=3))
    label(ax,x+.087,y+.094,"Ranked candidate genes",size=6.4,weight="bold",color=PURPLE)
    rounded(ax,x+.037,y+.033,.097,.024,fc="#FFF7E6",ec=GOLD,lw=.65,radius=.005)
    label(ax,x+.086,y+.045,"No universal winner",size=5.8,weight="bold",color="#8A5A00")


def draw_secondary(ax):
    x,y,w,h=.275,.075,.700,.355
    panel(ax,x,y,w,h,"e","Robustness and biological characterization",dashed=True)
    # three subgroups
    groups=[
        (x+.020,BLUE,"Bias audit",["Degree matching","Centrality stratification","Hub sensitivity"]),
        (x+.250,TEAL,"Uncertainty",["Edge perturbation","Gene-cluster bootstrap","Cancer heterogeneity"]),
        (x+.480,ORANGE,"Secondary analyses",["PU sensitivity","Cell-context ablation","TCGA expression anchor"]),
    ]
    for gx,gw,title,items in groups:
        rounded(ax,gx,y+.060,.200,.205,fc="#FAFBFC",ec="#AAB2BC",lw=.65,ls="--")
        label(ax,gx+.015,y+.235,title,size=6.9,weight="bold",ha="left",color=gw)
        for j,item in enumerate(items):
            cy=y+.190-j*.048
            ax.add_patch(Circle((gx+.022,cy),.0055,facecolor=gw,edgecolor="none",zorder=4))
            label(ax,gx+.036,cy,item,size=6.1,ha="left")
    label(ax,x+.020,y+.026,"Dashed panel: sensitivity or exploratory analyses; not used to tune the locked 26.06 evaluation.",
          size=5.7,ha="left",color=MID)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    mpl.rcParams.update({
        "font.family":"sans-serif",
        "font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":7,
        "pdf.fonttype":42,
        "ps.fonttype":42,
        "svg.fonttype":"none",
    })
    fig,ax=plt.subplots(figsize=(7.2,4.5))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    draw_inputs(ax); draw_models(ax); draw_evaluation(ax); draw_outputs(ax); draw_secondary(ax)
    arrow(ax,.232,.735,.272,.735)
    arrow(ax,.532,.735,.572,.735)
    arrow(ax,.762,.735,.802,.735)
    arrow(ax,.667,.505,.667,.435,color="#7F8C8D",lw=1.0)
    fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.99)
    base=args.output/"Figure_final_DCPI_RWR_workflow"
    fig.savefig(base.with_suffix(".pdf"),bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"),bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"),dpi=600,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    main()

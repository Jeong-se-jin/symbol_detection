#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AP1000 v45 — Precision Pipe-Attached Component Detection
========================================================
Precision-first, scale-normalized rules:
1) H/V extraction as v44.
2) A carrier boundary must itself be a pipe-like straight witness.  The minimum
   witness length is normalized by measured raster stroke width, not image pixels.
3) For each carrier, pair with the NEAREST following pipe-like carrier in the
   same tolerant lane that has off-axis component evidence at both endpoints.
   Short H/V strokes inside the symbol are ignored because they are not pipe-like
   witnesses.
4) Inside the exact displacement slab, require ONE local foreground component to
   touch BOTH outer carrier endpoints.  This proves the bbox belongs to an inline
   object attached to the pipe rather than an unrelated text/frame gap.
5) Final bbox is the extent of ONLY that bridge component inside the slab.
No global flood/path traversal, no bbox aspect-ratio classifier, no OCR/tag rules.
"""
from __future__ import annotations
import argparse, csv, json, math, sys, time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
import AP1000_v44_LATENT_CARRIER_DILATION_BBOX as v44
import AP1000_v34_AXIS_RELATIVE_FSM_RUN_GRAPH as v34
import AP1000_v37_TOLERANT_LONGEST_TRACK_HIERARCHY as v37

BBox=Tuple[int,int,int,int]

@dataclass
class Proposal:
    o:str; before:int; after:int; source:str
    bridge_bbox:Optional[BBox]=None
    bridge_area:int=0
    bridge_offaxis_extent:int=0

@dataclass
class Candidate:
    id:int; o:str; before_run:int; after_run:int
    before_axis:int; after_axis:int
    displacement_start:int; displacement_end:int
    before_length:int; after_length:int
    pipe_witness_min:int
    bridge_area:int; bridge_offaxis_extent:int
    bbox:BBox; proposal_source:str
    status:str='PENDING'; owner:Optional[int]=None; reason:str=''
    @property
    def displacement_span(self): return self.displacement_end-self.displacement_start+1
    @property
    def bbox_area(self):
        x1,y1,x2,y2=self.bbox; return (x2-x1+1)*(y2-y1+1)


def _stroke_width(median_radius:float)->float:
    # distance-transform radius is approximately half the raster stroke width.
    return max(1.0,2.0*float(median_radius)+1.0)


def pipe_witness_min_px(median_radius:float, factor:float=8.0)->int:
    return max(8,int(math.ceil(factor*_stroke_width(median_radius))))


def _baseline_normal(o,a,b,s):
    s0=float(a.end); s1=float(b.start)
    if s1<=s0: return int(round((a.axis+b.axis)/2))
    t=(float(s)-s0)/(s1-s0)
    return int(round((1.0-t)*a.axis+t*b.axis))


def _endpoint_labels(labels,o,a,b,x0,y0,margin,side):
    """Labels intersecting actual off-axis endpoint seeds (+ tiny raster neighborhood)."""
    run=a if side==0 else b
    run_side=1 if side==0 else 0
    seeds=v34.offaxis_endpoint_seeds(run,run_side,_endpoint_labels.ink)
    labs=set()
    H,W=labels.shape
    for x,y in seeds:
        lx=x-x0; ly=y-y0
        for dy in range(-margin,margin+1):
            for dx in range(-margin,margin+1):
                xx=lx+dx; yy=ly+dy
                if 0<=xx<W and 0<=yy<H:
                    z=int(labels[yy,xx])
                    if z>0: labs.add(z)
    return labs


def bridge_component(o,a,b,ink,margin,stroke_width):
    """Return bbox only if one local foreground object bridges both carrier endpoints.

    Connectivity is computed ONLY inside the exact longitudinal displacement slab.
    Perpendicular window is limited to 2*span.  A one-stroke-width repair dilation
    is used only for membership, while bbox coordinates come from original ink.
    """
    H,W=ink.shape
    lo,hi=int(a.end),int(b.start)
    if hi<=lo: return None
    span=hi-lo+1
    lim=max(int(math.ceil(2.0*span)),int(math.ceil(6.0*stroke_width)))
    center=int(round((a.axis+b.axis)/2))
    if o=='H':
        x0=max(0,lo); x1=min(W-1,hi); y0=max(0,center-lim); y1=min(H-1,center+lim)
    else:
        y0=max(0,lo); y1=min(H-1,hi); x0=max(0,center-lim); x1=min(W-1,center+lim)
    crop=ink[y0:y1+1,x0:x1+1].astype(np.uint8)
    if crop.size==0 or not np.any(crop): return None

    # Membership repair only. Kernel derives from measured recognition tolerance.
    k=max(1,2*int(margin)+1)
    lab_img=cv2.dilate(crop,np.ones((k,k),np.uint8),iterations=1) if k>1 else crop
    _,labels,_,_=cv2.connectedComponentsWithStats(lab_img,8)
    _endpoint_labels.ink=ink
    L=_endpoint_labels(labels,o,a,b,x0,y0,margin,0)
    R=_endpoint_labels(labels,o,a,b,x0,y0,margin,1)
    bridge=L & R
    if not bridge: return None

    # Keep only original ink whose repaired-membership label bridges both endpoints.
    mask=np.isin(labels,list(bridge)) & (crop>0)
    pts=np.argwhere(mask)
    if len(pts)==0: return None
    yy0,xx0=pts.min(axis=0); yy1,xx1=pts.max(axis=0)
    bx1=x0+int(xx0); by1=y0+int(yy0); bx2=x0+int(xx1); by2=y0+int(yy1)

    # The geometry must actually leave the carrier corridor; otherwise this is
    # just a tiny line repair / junction artifact, not a component body.
    if o=='H':
        off=max(abs(by1-int(a.axis)),abs(by2-int(a.axis)),abs(by1-int(b.axis)),abs(by2-int(b.axis)))
    else:
        off=max(abs(bx1-int(a.axis)),abs(bx2-int(a.axis)),abs(bx1-int(b.axis)),abs(bx2-int(b.axis)))
    min_off=max(int(margin)+1,int(math.ceil(0.75*stroke_width)))
    if off<min_off: return None
    return (bx1,by1,bx2,by2),int(len(pts)),int(off)


def precision_proposals(rs,active,ink,margin,median_radius,witness_factor=8.0,max_displacement_strokes=30.0,max_gap_fraction=0.05):
    import bisect
    witness=pipe_witness_min_px(median_radius,witness_factor)
    stroke=_stroke_width(median_radius)
    # Only individual straight runs with enough scale-normalized support can be
    # outer pipe boundaries. Internal symbol strokes/text strokes do not qualify.
    pipe_ids=[rid for rid in active if rs[rid].length>=witness]
    bins={'H':defaultdict(list),'V':defaultdict(list)}
    starts={'H':{},'V':{}}
    for rid in pipe_ids:
        r=rs[rid]; bins[r.o][int(r.axis)].append(rid)
    for o in ('H','V'):
        for ax,ids in bins[o].items():
            ids.sort(key=lambda i:rs[i].start); starts[o][ax]=[rs[i].start for i in ids]

    horizon=max(witness*3,int(round(min(ink.shape)*max_gap_fraction)))
    max_local_gap=max(witness,int(math.ceil(max_displacement_strokes*stroke)))
    out=[]; rejected_no_bridge=0; rejected_no_endpoint=0; rejected_nonlocal_gap=0
    for aid in pipe_ids:
        a=rs[aid]
        # nearest eligible successors first; do not deliberately widen gaps.
        succ=[]
        for ax2 in range(int(a.axis)-margin,int(a.axis)+margin+1):
            ids=bins[a.o].get(ax2)
            if not ids: continue
            k=bisect.bisect_right(starts[a.o][ax2],int(a.end)+1)
            succ.extend(ids[k:k+12])
        succ=sorted(set(succ),key=lambda i:(rs[i].start,abs(int(rs[i].axis)-int(a.axis))))
        chosen=None
        for bid in succ:
            b=rs[bid]
            if b.start<=a.end or abs(int(a.axis)-int(b.axis))>margin: continue
            gap=int(b.start-a.end-1)
            if gap<2: continue
            if gap>horizon: break
            # Precision-first locality: an inline P&ID symbol is a local interruption
            # of its carrier. Normalize by measured line weight, never by image DPI.
            # This constrains longitudinal displacement only; actuator/protrusion
            # distance perpendicular to the pipe remains unrestricted by this rule.
            if gap+1>max_local_gap:
                rejected_nonlocal_gap+=1; continue
            # Both endpoints must physically turn into non-axis ink.
            if not v34.offaxis_endpoint_seeds(a,1,ink) or not v34.offaxis_endpoint_seeds(b,0,ink):
                rejected_no_endpoint+=1; continue
            bridged=bridge_component(a.o,a,b,ink,margin,stroke)
            if bridged is None:
                rejected_no_bridge+=1; continue
            bb,area,off=bridged
            chosen=Proposal(a.o,aid,bid,'PIPE_WITNESS_NEAREST_BRIDGE',bb,area,off)
            break
        if chosen is not None: out.append(chosen)

    # Pair symmetry: if A->B is chosen, B should be the nearest valid pipe boundary
    # from the opposite direction as well. Build reverse-nearest map from proposals.
    by_after=defaultdict(list)
    for p in out: by_after[p.after].append(p)
    final=[]
    for p in out:
        # Among proposals ending at B, the one with largest A.end is reverse-nearest.
        group=by_after[p.after]
        reverse=max(group,key=lambda q:rs[q.before].end)
        if reverse.before==p.before: final.append(p)
    final.sort(key=lambda p:(p.o,rs[p.before].axis,rs[p.before].start,rs[p.after].start))
    return final,{
        'measured_stroke_width_px':round(stroke,3),
        'pipe_witness_factor_x_stroke_width':float(witness_factor),
        'max_displacement_strokes':float(max_displacement_strokes),
        'max_displacement_px':int(max_local_gap),
        'pipe_witness_min_px':int(witness),
        'pipe_eligible_runs':len(pipe_ids),
        'nearest_bridge_proposals':len(final),
        'proposal_search_horizon_px':int(horizon),
        'rejected_nonlocal_displacement':int(rejected_nonlocal_gap),
        'rejected_no_endpoint_attachment':int(rejected_no_endpoint),
        'rejected_no_single_bridge_component':int(rejected_no_bridge),
    }


def build_candidates(ps,rs,witness):
    out=[]
    for p in ps:
        a,b=rs[p.before],rs[p.after]
        if p.bridge_bbox is None: continue
        c=Candidate(len(out),p.o,p.before,p.after,int(a.axis),int(b.axis),int(a.end),int(b.start),
                    int(a.length),int(b.length),int(witness),int(p.bridge_area),int(p.bridge_offaxis_extent),
                    p.bridge_bbox,p.source)
        c.reason='BOTH_BOUNDARIES_PIPE_LIKE; ONE_LOCAL_COMPONENT_BRIDGES_BOTH_ENDPOINTS'
        out.append(c)
    return out


def bbox_intersects(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    return not(ax2<bx1 or bx2<ax1 or ay2<by1 or by2<ay1)

def center(c):
    return ((c.bbox[0]+c.bbox[2])//2,(c.bbox[1]+c.bbox[3])//2)

def duplicate(a,b,margin):
    if a.o==b.o and max(abs(a.before_axis-b.before_axis),abs(a.after_axis-b.after_axis))<=margin:
        alo,ahi=sorted((a.displacement_start,a.displacement_end)); blo,bhi=sorted((b.displacement_start,b.displacement_end))
        if max(alo,blo)<=min(ahi,bhi): return True
    if not bbox_intersects(a.bbox,b.bbox): return False
    ax,ay=center(a); bx,by=center(b)
    A=a.bbox; B=b.bbox
    return B[0]<=ax<=B[2] and B[1]<=ay<=B[3] and A[0]<=bx<=A[2] and A[1]<=by<=A[3]

def resolve(cands,margin):
    order=sorted(range(len(cands)),key=lambda i:(min(cands[i].before_length,cands[i].after_length),cands[i].bridge_area,-cands[i].bbox_area),reverse=True)
    accepted=[]
    for i in order:
        c=cands[i]; owner=None
        for j in accepted:
            if duplicate(c,cands[j],margin): owner=j; break
        if owner is None:
            c.status='ACCEPTED'; c.reason='PIPE_ATTACHED_BRIDGE_ACCEPTED'; accepted.append(i)
        else:
            c.status='DUPLICATE'; c.owner=cands[owner].id; c.reason='SAME_LOCAL_INLINE_COMPONENT'
    return accepted


def draw_raw(g,rs,active): return v44.draw_raw_hv(g,rs,active)

def draw_pipe_witness(g,rs,pipe_ids):
    im=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)
    for rid in pipe_ids:
        r=rs[rid]
        if r.o=='H': cv2.line(im,(r.start,r.axis),(r.end,r.axis),(0,160,0),1)
        else: cv2.line(im,(r.axis,r.start),(r.axis,r.end),(0,160,0),1)
    return im

def draw_proposals(g,rs,ps):
    im=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)
    for p in ps:
        a,b=rs[p.before],rs[p.after]
        if p.o=='H': cv2.line(im,(a.end,a.axis),(b.start,b.axis),(0,0,255),1)
        else: cv2.line(im,(a.axis,a.end),(b.axis,b.start),(0,0,255),1)
    return im

def draw_final(g,rs,cands,accepted):
    im=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)
    for i in accepted:
        c=cands[i]; a,b=rs[c.before_run],rs[c.after_run]
        if c.o=='H':
            cv2.line(im,(a.start,a.axis),(a.end,a.axis),(0,180,0),1); cv2.line(im,(b.start,b.axis),(b.end,b.axis),(0,180,0),1)
        else:
            cv2.line(im,(a.axis,a.start),(a.axis,a.end),(0,180,0),1); cv2.line(im,(b.axis,b.start),(b.axis,b.end),(0,180,0),1)
        x1,y1,x2,y2=c.bbox; cv2.rectangle(im,(x1,y1),(x2,y2),(255,0,255),1)
    return im


def save_csv(out,cands):
    with open(out/'precision_components.csv','w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['candidate_id','status','owner','reason','proposal_source','orientation','before_run','after_run','before_axis','after_axis','axis_shift_px','before_length','after_length','pipe_witness_min','displacement_start','displacement_end','displacement_span','bridge_area','bridge_offaxis_extent','x1','y1','x2','y2','width','height','area'])
        for c in cands:
            x1,y1,x2,y2=c.bbox
            w.writerow([c.id,c.status,c.owner,c.reason,c.proposal_source,c.o,c.before_run,c.after_run,c.before_axis,c.after_axis,abs(c.after_axis-c.before_axis),c.before_length,c.after_length,c.pipe_witness_min,c.displacement_start,c.displacement_end,c.displacement_span,c.bridge_area,c.bridge_offaxis_extent,x1,y1,x2,y2,x2-x1+1,y2-y1+1,c.bbox_area])


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('input'); ap.add_argument('-o','--output',default='AP1000_v45_PRECISION_PIPE_ATTACHED_RESULT')
    ap.add_argument('--min-line-length',type=int,default=3); ap.add_argument('--pipe-witness-factor',type=float,default=8.0); ap.add_argument('--max-displacement-strokes',type=float,default=30.0)
    args=ap.parse_args(); out=Path(args.output); out.mkdir(parents=True,exist_ok=True); t0=time.time()
    g=v34.load(Path(args.input)); ink=v34.binarize(g); sk=v34.skeletonize(ink)
    margin,med=v37.estimate_gap_margin(ink,sk); rs,hb0,vb0,hid0,vid0=v34.extract(sk)
    hb,vb,hid,vid,active,lstats=v44.filter_line_entities(rs,hb0,vb0,hid0,vid0,args.min_line_length); v34.annotate(rs,hid,vid)
    ps,pstats=precision_proposals(rs,active,ink,margin,med,args.pipe_witness_factor,args.max_displacement_strokes)
    witness=pstats['pipe_witness_min_px']; cands=build_candidates(ps,rs,witness); accepted=resolve(cands,margin)
    pipe_ids=[rid for rid in active if rs[rid].length>=witness]
    cv2.imwrite(str(out/'01_raw_HV_runs.png'),draw_raw(g,rs,active),[cv2.IMWRITE_PNG_COMPRESSION,1])
    cv2.imwrite(str(out/'02_pipe_like_HV_witnesses.png'),draw_pipe_witness(g,rs,pipe_ids),[cv2.IMWRITE_PNG_COMPRESSION,1])
    cv2.imwrite(str(out/'03_verified_displacements.png'),draw_proposals(g,rs,ps),[cv2.IMWRITE_PNG_COMPRESSION,1])
    cv2.imwrite(str(out/'04_v45_precision_bbox.png'),draw_final(g,rs,cands,accepted),[cv2.IMWRITE_PNG_COMPRESSION,1])
    save_csv(out,cands)
    areas=[cands[i].bbox_area for i in accepted]
    summary={'version':'v45','mode':'PRECISION_FIRST_PIPE_ATTACHED_ONLY','image_size':[g.shape[1],g.shape[0]],'line_recognition_margin_px':margin,'median_skeleton_stroke_radius_px':round(float(med),3),**lstats,**pstats,'candidates_built':len(cands),'accepted_components':len(accepted),'duplicates':sum(c.status=='DUPLICATE' for c in cands),'accepted_bbox_area_median':float(np.median(areas)) if areas else 0.0,'accepted_bbox_area_max':int(max(areas)) if areas else 0,'global_connectivity_used':False,'local_connectivity_scope':'EXACT_DISPLACEMENT_SLAB_ONLY','rule':'nearest scale-normalized pipe-like H/V boundaries + single local foreground component must bridge both pipe endpoints','runtime_sec':round(time.time()-t0,3)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AP1000 v46 — Precision Multipart Pipe-Attached Component Detection
=================================================================
Keeps v45's precision-first connected-symbol detector intact, then adds a
geometry-only multipart fallback for intentionally disconnected inline symbols
such as orifices / split plates / broken-raster symbols.

Important:
- No OCR/tag/class exception for Orifice.
- No global flood/path traversal.
- Existing connected components retain v45 behavior, including tall actuators.
- Multipart fallback is confined to the exact displacement slab and must be
  locally balanced around the carrier so aligned text/instrument-box edges do
  not become components.
"""
from __future__ import annotations
import argparse, csv, json, math, sys, time
from collections import defaultdict, deque
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
import AP1000_v45_PRECISION_PIPE_ATTACHED_BBOX as v45

BBox=Tuple[int,int,int,int]

@dataclass
class Proposal:
    o:str; before:int; after:int; source:str
    symbol_bbox:Optional[BBox]=None
    symbol_area:int=0
    symbol_offaxis_extent:int=0
    fragment_count:int=0
    graph_hops:int=0
    side_balance:float=1.0
    geometry_mode:str='CONNECTED'

@dataclass
class Candidate:
    id:int; o:str; before_run:int; after_run:int
    before_axis:int; after_axis:int
    displacement_start:int; displacement_end:int
    before_length:int; after_length:int
    pipe_witness_min:int
    symbol_area:int; symbol_offaxis_extent:int
    fragment_count:int; graph_hops:int; side_balance:float
    geometry_mode:str; bbox:BBox; proposal_source:str
    status:str='PENDING'; owner:Optional[int]=None; reason:str=''
    @property
    def displacement_span(self): return self.displacement_end-self.displacement_start+1
    @property
    def bbox_area(self):
        x1,y1,x2,y2=self.bbox; return (x2-x1+1)*(y2-y1+1)


def _stroke_width(median_radius:float)->float:
    return max(1.0,2.0*float(median_radius)+1.0)

def pipe_witness_min_px(median_radius:float,factor:float=8.0)->int:
    return max(8,int(math.ceil(factor*_stroke_width(median_radius))))

def _local_axis(o,a,b,s):
    s0=float(a.end); s1=float(b.start)
    if s1<=s0: return int(round((a.axis+b.axis)/2))
    t=(float(s)-s0)/(s1-s0)
    return int(round((1.0-t)*a.axis+t*b.axis))


def _label_geometry(labels,crop,o,a,b,x0,y0):
    ys,xs=np.nonzero(crop); by=defaultdict(list)
    for yy,xx in zip(ys.tolist(),xs.tolist()):
        z=int(labels[yy,xx])
        if z>0: by[z].append((xx,yy))
    info={}
    for z,pts in by.items():
        arr=np.asarray(pts,np.int32); xx0,yy0=arr.min(axis=0); xx1,yy1=arr.max(axis=0)
        neg=pos=0
        if o=='H':
            long0,long1=x0+int(xx0),x0+int(xx1)
            for xx,yy in pts:
                gx=x0+xx; gy=y0+yy; d=gy-_local_axis(o,a,b,gx)
                if d<0: neg=max(neg,-d)
                else: pos=max(pos,d)
        else:
            long0,long1=y0+int(yy0),y0+int(yy1)
            for xx,yy in pts:
                gx=x0+xx; gy=y0+yy; d=gx-_local_axis(o,a,b,gy)
                if d<0: neg=max(neg,-d)
                else: pos=max(pos,d)
        info[z]={'bbox':(x0+int(xx0),y0+int(yy0),x0+int(xx1),y0+int(yy1)),
                 'long':(int(long0),int(long1)),'neg':int(neg),'pos':int(pos),
                 'off':int(max(neg,pos)),'area':len(pts)}
    return info


def _labels_in_endpoint_window(labels,crop,o,a,b,x0,y0,side,attach_long,attach_normal):
    H,W=crop.shape
    if o=='H':
        gx=int(a.end if side==0 else b.start); gy=int(a.axis if side==0 else b.axis)
        if side==0: xa=max(0,gx-x0); xb=min(W-1,gx-x0+attach_long)
        else: xa=max(0,gx-x0-attach_long); xb=min(W-1,gx-x0)
        ya=max(0,gy-y0-attach_normal); yb=min(H-1,gy-y0+attach_normal)
    else:
        gx=int(a.axis if side==0 else b.axis); gy=int(a.end if side==0 else b.start)
        xa=max(0,gx-x0-attach_normal); xb=min(W-1,gx-x0+attach_normal)
        if side==0: ya=max(0,gy-y0); yb=min(H-1,gy-y0+attach_long)
        else: ya=max(0,gy-y0-attach_long); yb=min(H-1,gy-y0)
    if xa>xb or ya>yb: return set()
    m=crop[ya:yb+1,xa:xb+1]>0
    z=labels[ya:yb+1,xa:xb+1][m]
    return set(int(q) for q in np.unique(z) if int(q)>0)


def _baseline_labels(labels,crop,o,a,b,x0,y0,corridor):
    keep=set(); ys,xs=np.nonzero(crop)
    for yy,xx in zip(ys.tolist(),xs.tolist()):
        gx=x0+xx; gy=y0+yy; s=gx if o=='H' else gy; normal=gy if o=='H' else gx
        if abs(normal-_local_axis(o,a,b,s))<=corridor:
            z=int(labels[yy,xx])
            if z>0: keep.add(z)
    return keep


def _interval_gap(A,B):
    a0,a1=A; b0,b1=B
    if a1<b0: return b0-a1-1
    if b1<a0: return a0-b1-1
    return 0


def multipart_component(o,a,b,ink,margin,stroke_width,fragment_gap_strokes=6.0,
                        min_side_balance=0.30,two_fragment_max_normal_ratio=1.50):
    """Geometry-only fallback for disconnected inline symbols.

    It deliberately does NOT require one raw connected foreground component.
    Instead, baseline-local fragments form a proximity graph. To prevent aligned
    text/instrument boxes from being promoted, the resulting multipart family must
    have meaningful ink on BOTH sides of the carrier, and a two-fragment family
    cannot extend excessively far normal to the pipe compared with its displacement.
    """
    H,W=ink.shape; lo,hi=int(a.end),int(b.start)
    if hi<=lo: return None
    span=hi-lo+1; center=int(round((a.axis+b.axis)/2))
    lim=max(int(math.ceil(2.0*span)),int(math.ceil(6.0*stroke_width)))
    if o=='H': x0=max(0,lo); x1=min(W-1,hi); y0=max(0,center-lim); y1=min(H-1,center+lim)
    else: y0=max(0,lo); y1=min(H-1,hi); x0=max(0,center-lim); x1=min(W-1,center+lim)
    crop=ink[y0:y1+1,x0:x1+1].astype(np.uint8)
    if crop.size==0 or not np.any(crop): return None

    # Raster repair only; deliberate whitespace remains as separate labels.
    k=max(1,2*int(margin)+1)
    repair=cv2.dilate(crop,np.ones((k,k),np.uint8),iterations=1) if k>1 else crop
    _,labels,_,_=cv2.connectedComponentsWithStats(repair,8)
    info=_label_geometry(labels,crop,o,a,b,x0,y0)
    if not info: return None

    corridor=max(int(margin)+1,int(math.ceil(2.0*stroke_width)))
    attach_long=max(int(margin)+2,int(math.ceil(4.0*stroke_width)))
    attach_normal=max(corridor,int(math.ceil(3.0*stroke_width)))
    frag_gap=max(int(margin)+1,int(math.ceil(fragment_gap_strokes*stroke_width)))
    min_off=max(int(margin)+1,int(math.ceil(0.75*stroke_width)))

    baseline=_baseline_labels(labels,crop,o,a,b,x0,y0,corridor)
    useful={z for z in baseline if z in info and info[z]['off']>=min_off}
    if not useful: return None
    L=_labels_in_endpoint_window(labels,crop,o,a,b,x0,y0,0,attach_long,attach_normal) & useful
    R=_labels_in_endpoint_window(labels,crop,o,a,b,x0,y0,1,attach_long,attach_normal) & useful
    if not L or not R: return None

    nodes=sorted(useful); adj={z:set() for z in nodes}
    for i,z in enumerate(nodes):
        for q in nodes[i+1:]:
            if _interval_gap(info[z]['long'],info[q]['long'])<=frag_gap:
                adj[z].add(q); adj[q].add(z)

    dq=deque((z,0) for z in L); seen=set(L); target=None; hops=0
    while dq:
        z,d=dq.popleft()
        if z in R: target=z; hops=d; break
        for q in adj.get(z,()):
            if q not in seen: seen.add(q); dq.append((q,d+1))
    if target is None: return None

    family=set(L); dq=deque(L)
    while dq:
        z=dq.popleft()
        for q in adj.get(z,()):
            if q not in family: family.add(q); dq.append(q)
    if not (family & R): return None

    # Multipart-only precision constraints. Connected v45 detections bypass these.
    neg=max(info[z]['neg'] for z in family); pos=max(info[z]['pos'] for z in family)
    if neg<min_off or pos<min_off: return None
    balance=float(min(neg,pos))/float(max(neg,pos)) if max(neg,pos)>0 else 0.0
    if balance<float(min_side_balance): return None
    if len(family)<=2 and max(neg,pos)>float(two_fragment_max_normal_ratio)*float(span): return None

    mask=np.isin(labels,list(family)) & (crop>0); pts=np.argwhere(mask)
    if len(pts)==0: return None
    yy0,xx0=pts.min(axis=0); yy1,xx1=pts.max(axis=0)
    bb=(x0+int(xx0),y0+int(yy0),x0+int(xx1),y0+int(yy1))
    return bb,int(len(pts)),int(max(neg,pos)),int(len(family)),int(hops),float(balance)


def precision_proposals(rs,active,ink,margin,median_radius,witness_factor=8.0,
                        max_displacement_strokes=30.0,max_gap_fraction=0.05,
                        fragment_gap_strokes=6.0,min_side_balance=0.30,
                        two_fragment_max_normal_ratio=1.50):
    import bisect
    witness=pipe_witness_min_px(median_radius,witness_factor); stroke=_stroke_width(median_radius)
    pipe_ids=[rid for rid in active if rs[rid].length>=witness]
    bins={'H':defaultdict(list),'V':defaultdict(list)}; starts={'H':{},'V':{}}
    for rid in pipe_ids:
        r=rs[rid]; bins[r.o][int(r.axis)].append(rid)
    for o in ('H','V'):
        for ax,ids in bins[o].items():
            ids.sort(key=lambda i:rs[i].start); starts[o][ax]=[rs[i].start for i in ids]

    horizon=max(witness*3,int(round(min(ink.shape)*max_gap_fraction)))
    max_local_gap=max(witness,int(math.ceil(max_displacement_strokes*stroke)))
    out=[]; reject_nonlocal=reject_geometry=0; connected_n=multipart_n=0
    for aid in pipe_ids:
        a=rs[aid]; succ=[]
        for ax2 in range(int(a.axis)-margin,int(a.axis)+margin+1):
            ids=bins[a.o].get(ax2)
            if not ids: continue
            k=bisect.bisect_right(starts[a.o][ax2],int(a.end)+1); succ.extend(ids[k:k+12])
        succ=sorted(set(succ),key=lambda i:(rs[i].start,abs(int(rs[i].axis)-int(a.axis))))
        chosen=None
        for bid in succ:
            b=rs[bid]
            if b.start<=a.end or abs(int(a.axis)-int(b.axis))>margin: continue
            gap=int(b.start-a.end-1)
            if gap<2: continue
            if gap>horizon: break
            if gap+1>max_local_gap: reject_nonlocal+=1; continue

            # Preserve v45 exactly whenever one local foreground component bridges.
            connected=v45.bridge_component(a.o,a,b,ink,margin,stroke)
            if connected is not None:
                bb,area,off=connected
                chosen=Proposal(a.o,aid,bid,'PIPE_WITNESS_NEAREST_CONNECTED',bb,area,off,1,0,1.0,'CONNECTED')
                connected_n+=1; break

            # Only v45 misses reach this disconnected/multipart fallback.
            multi=multipart_component(a.o,a,b,ink,margin,stroke,fragment_gap_strokes,
                                      min_side_balance,two_fragment_max_normal_ratio)
            if multi is None: reject_geometry+=1; continue
            bb,area,off,nfrag,hops,balance=multi
            chosen=Proposal(a.o,aid,bid,'PIPE_WITNESS_NEAREST_MULTIPART',bb,area,off,nfrag,hops,balance,'MULTIPART')
            multipart_n+=1; break
        if chosen is not None: out.append(chosen)

    by_after=defaultdict(list)
    for p in out: by_after[p.after].append(p)
    final=[]
    for p in out:
        reverse=max(by_after[p.after],key=lambda q:rs[q.before].end)
        if reverse.before==p.before: final.append(p)
    final.sort(key=lambda p:(p.o,rs[p.before].axis,rs[p.before].start,rs[p.after].start))
    return final,{
        'measured_stroke_width_px':round(stroke,3),'pipe_witness_factor_x_stroke_width':float(witness_factor),
        'max_displacement_strokes':float(max_displacement_strokes),'max_displacement_px':int(max_local_gap),
        'multipart_fragment_gap_strokes':float(fragment_gap_strokes),'multipart_fragment_gap_px':int(math.ceil(fragment_gap_strokes*stroke)),
        'multipart_min_side_balance':float(min_side_balance),'two_fragment_max_normal_to_displacement_ratio':float(two_fragment_max_normal_ratio),
        'pipe_witness_min_px':int(witness),'pipe_eligible_runs':len(pipe_ids),'nearest_verified_proposals':len(final),
        'connected_proposals_before_symmetry':int(connected_n),'multipart_proposals_before_symmetry':int(multipart_n),
        'proposal_search_horizon_px':int(horizon),'rejected_nonlocal_displacement':int(reject_nonlocal),
        'rejected_no_connected_or_multipart_geometry':int(reject_geometry)}


def build_candidates(ps,rs,witness):
    out=[]
    for p in ps:
        if p.symbol_bbox is None: continue
        a,b=rs[p.before],rs[p.after]
        c=Candidate(len(out),p.o,p.before,p.after,int(a.axis),int(b.axis),int(a.end),int(b.start),int(a.length),int(b.length),
                    int(witness),int(p.symbol_area),int(p.symbol_offaxis_extent),int(p.fragment_count),int(p.graph_hops),
                    float(p.side_balance),p.geometry_mode,p.symbol_bbox,p.source)
        c.reason='V45_CONNECTED' if p.geometry_mode=='CONNECTED' else 'LOCAL_MULTIPART_FRAGMENT_CHAIN'
        out.append(c)
    return out


def bbox_intersects(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    return not(ax2<bx1 or bx2<ax1 or ay2<by1 or by2<ay1)
def center(c): return ((c.bbox[0]+c.bbox[2])//2,(c.bbox[1]+c.bbox[3])//2)
def duplicate(a,b,margin):
    if a.o==b.o and max(abs(a.before_axis-b.before_axis),abs(a.after_axis-b.after_axis))<=margin:
        alo,ahi=sorted((a.displacement_start,a.displacement_end)); blo,bhi=sorted((b.displacement_start,b.displacement_end))
        if max(alo,blo)<=min(ahi,bhi): return True
    if not bbox_intersects(a.bbox,b.bbox): return False
    ax,ay=center(a); bx,by=center(b); A=a.bbox; B=b.bbox
    return B[0]<=ax<=B[2] and B[1]<=ay<=B[3] and A[0]<=bx<=A[2] and A[1]<=by<=A[3]
def resolve(cands,margin):
    order=sorted(range(len(cands)),key=lambda i:(min(cands[i].before_length,cands[i].after_length),cands[i].symbol_area,-cands[i].bbox_area),reverse=True)
    accepted=[]
    for i in order:
        c=cands[i]; owner=None
        for j in accepted:
            if duplicate(c,cands[j],margin): owner=j; break
        if owner is None:
            c.status='ACCEPTED'; c.reason=('PIPE_ATTACHED_CONNECTED_ACCEPTED' if c.geometry_mode=='CONNECTED' else 'PIPE_ATTACHED_MULTIPART_ACCEPTED'); accepted.append(i)
        else:
            c.status='DUPLICATE'; c.owner=cands[owner].id; c.reason='SAME_LOCAL_INLINE_COMPONENT'
    return accepted


def draw_raw(g,rs,active): return v44.draw_raw_hv(g,rs,active)
def draw_pipe_witness(g,rs,pipe_ids): return v45.draw_pipe_witness(g,rs,pipe_ids)
def draw_proposals(g,rs,ps):
    im=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)
    for p in ps:
        a,b=rs[p.before],rs[p.after]; col=(0,0,255) if p.geometry_mode=='MULTIPART' else (0,160,255)
        if p.o=='H': cv2.line(im,(a.end,a.axis),(b.start,b.axis),col,1)
        else: cv2.line(im,(a.axis,a.end),(b.axis,b.start),col,1)
    return im

def draw_final(g,rs,cands,accepted):
    im=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)
    for i in accepted:
        c=cands[i]; a,b=rs[c.before_run],rs[c.after_run]
        if c.o=='H':
            cv2.line(im,(a.start,a.axis),(a.end,a.axis),(0,180,0),1); cv2.line(im,(b.start,b.axis),(b.end,b.axis),(0,180,0),1)
        else:
            cv2.line(im,(a.axis,a.start),(a.axis,a.end),(0,180,0),1); cv2.line(im,(b.axis,b.start),(b.axis,b.end),(0,180,0),1)
        x1,y1,x2,y2=c.bbox; col=(255,0,255) if c.geometry_mode=='CONNECTED' else (0,0,255)
        cv2.rectangle(im,(x1,y1),(x2,y2),col,1)
    return im


def save_csv(out,cands):
    with open(out/'multipart_components.csv','w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['candidate_id','status','owner','reason','proposal_source','geometry_mode','orientation','before_run','after_run','before_axis','after_axis','axis_shift_px','before_length','after_length','pipe_witness_min','displacement_start','displacement_end','displacement_span','symbol_area','symbol_offaxis_extent','fragment_count','fragment_graph_hops','side_balance','x1','y1','x2','y2','width','height','area'])
        for c in cands:
            x1,y1,x2,y2=c.bbox
            w.writerow([c.id,c.status,c.owner,c.reason,c.proposal_source,c.geometry_mode,c.o,c.before_run,c.after_run,c.before_axis,c.after_axis,abs(c.after_axis-c.before_axis),c.before_length,c.after_length,c.pipe_witness_min,c.displacement_start,c.displacement_end,c.displacement_span,c.symbol_area,c.symbol_offaxis_extent,c.fragment_count,c.graph_hops,round(c.side_balance,4),x1,y1,x2,y2,x2-x1+1,y2-y1+1,c.bbox_area])


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('input'); ap.add_argument('-o','--output',default='AP1000_v46_MULTIPART_PIPE_ATTACHED_RESULT')
    ap.add_argument('--min-line-length',type=int,default=3); ap.add_argument('--pipe-witness-factor',type=float,default=8.0)
    ap.add_argument('--max-displacement-strokes',type=float,default=30.0); ap.add_argument('--fragment-gap-strokes',type=float,default=6.0)
    ap.add_argument('--multipart-min-side-balance',type=float,default=0.30); ap.add_argument('--two-fragment-max-normal-ratio',type=float,default=1.50)
    args=ap.parse_args(); out=Path(args.output); out.mkdir(parents=True,exist_ok=True); t0=time.time()
    g=v34.load(Path(args.input)); ink=v34.binarize(g); sk=v34.skeletonize(ink)
    margin,med=v37.estimate_gap_margin(ink,sk); rs,hb0,vb0,hid0,vid0=v34.extract(sk)
    hb,vb,hid,vid,active,lstats=v44.filter_line_entities(rs,hb0,vb0,hid0,vid0,args.min_line_length); v34.annotate(rs,hid,vid)
    ps,pstats=precision_proposals(rs,active,ink,margin,med,args.pipe_witness_factor,args.max_displacement_strokes,
                                  fragment_gap_strokes=args.fragment_gap_strokes,min_side_balance=args.multipart_min_side_balance,
                                  two_fragment_max_normal_ratio=args.two_fragment_max_normal_ratio)
    witness=pstats['pipe_witness_min_px']; cands=build_candidates(ps,rs,witness); accepted=resolve(cands,margin)
    pipe_ids=[rid for rid in active if rs[rid].length>=witness]
    cv2.imwrite(str(out/'01_raw_HV_runs.png'),draw_raw(g,rs,active),[cv2.IMWRITE_PNG_COMPRESSION,1])
    cv2.imwrite(str(out/'02_pipe_like_HV_witnesses.png'),draw_pipe_witness(g,rs,pipe_ids),[cv2.IMWRITE_PNG_COMPRESSION,1])
    cv2.imwrite(str(out/'03_verified_displacements.png'),draw_proposals(g,rs,ps),[cv2.IMWRITE_PNG_COMPRESSION,1])
    cv2.imwrite(str(out/'04_v46_precision_multipart_bbox.png'),draw_final(g,rs,cands,accepted),[cv2.IMWRITE_PNG_COMPRESSION,1])
    save_csv(out,cands)
    areas=[cands[i].bbox_area for i in accepted]; con=sum(cands[i].geometry_mode=='CONNECTED' for i in accepted); mul=sum(cands[i].geometry_mode=='MULTIPART' for i in accepted)
    summary={'version':'v46','mode':'PRECISION_PIPE_ATTACHED_CONNECTED_PLUS_MULTIPART_FALLBACK','image_size':[g.shape[1],g.shape[0]],'line_recognition_margin_px':margin,'median_skeleton_stroke_radius_px':round(float(med),3),**lstats,**pstats,'candidates_built':len(cands),'accepted_components':len(accepted),'accepted_connected':int(con),'accepted_multipart_fallback':int(mul),'duplicates':sum(c.status=='DUPLICATE' for c in cands),'accepted_bbox_area_median':float(np.median(areas)) if areas else 0.0,'accepted_bbox_area_max':int(max(areas)) if areas else 0,'global_connectivity_used':False,'v45_connected_geometry_preserved':True,'multipart_scope':'EXACT_DISPLACEMENT_SLAB_ONLY','runtime_sec':round(time.time()-t0,3)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()

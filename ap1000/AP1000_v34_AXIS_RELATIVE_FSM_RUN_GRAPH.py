#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AP1000 v34 — Axis-Relative FSM + Normal-Run Graph
==================================================

Fixes the two structural problems in v33:
1) Same-axis raster fragments inside a valve no longer close the FSM component.
2) H/V runs are NOT globally removed from displacement.  Displacement is
   evaluated relative to one parent axis: only that parent axis itself is
   forbidden while testing/recovering the component.

Event detection
---------------
For each pair of adjacent same-axis H/V runs, test whether their endpoints are
connected by foreground when the parent axis row/column is forbidden, with
motion constrained to the longitudinal slab between the endpoints.  This keeps
all diagonal, vertical, horizontal, and raster stair-step pixels of a symbol.

Adjacent component gaps separated only by internal same-axis fragments are
merged if the OUTER endpoints are still connected under the same axis-forbidden
connectivity.  Hence:
  pipe -- valve-internal-H -- pipe  => one component
while two independent components separated by a real carrier run do not merge.

Disconnected two-boundary structures (e.g. orifice plates) remain supported as
a local scale-free special case.

Shape recovery
--------------
After the FSM interval is fixed, bbox recovery uses a normal-direction RLE graph
on the RAW skeleton.  For the selected parent axis only, normal runs crossing
that axis are split at the axis.  Thus the parent carrier itself cannot connect
tracks, but every other H/V/diagonal fragment remains available.  All
axis-adjacent tracks inside one FSM interval are unioned for the final envelope.

No text pre-filter, Hough/LSD, length clustering, or bbox margin.
"""
from __future__ import annotations
import argparse, csv, json, time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

import cv2
import numpy as np
from skimage.morphology import skeletonize
try:
    import fitz
except Exception:
    fitz = None

BBox = Tuple[int,int,int,int]
Interval = Tuple[int,int]

@dataclass
class Run:
    id:int; o:str; axis:int; start:int; end:int; c0:bool=False; c1:bool=False
    @property
    def length(self): return self.end-self.start+1
    @property
    def structural(self): return self.c0 and self.c1

@dataclass
class Event:
    o:str; axis:int; before:int; after:int; source:str; span_gaps:int=1

@dataclass
class Component:
    id:int; o:str; axis:int; kind:str; before:Optional[int]; after:Optional[int]
    source:str; bbox:BBox; track_count:int; node_count:int
    longitudinal_start:int; longitudinal_end:int


def norm(g):
    if g.dtype != np.uint8: g=np.clip(g,0,255).astype(np.uint8)
    return 255-g if g.mean()<127 else g

def load(p:Path):
    if p.suffix.lower() != '.pdf':
        g=cv2.imread(str(p),0)
        if g is None: raise RuntimeError(p)
        return norm(g)
    if fitz is None: raise RuntimeError('PyMuPDF is required for PDF input')
    d=fitz.open(str(p)); pg=d[0]; best=None; area=-1
    for inf in pg.get_images(full=True):
        try:
            dat=d.extract_image(inf[0]); a=np.frombuffer(dat['image'],np.uint8)
            im=cv2.imdecode(a,0)
            if im is not None and im.size>area: best=im; area=im.size
        except Exception: pass
    if best is not None: return norm(best)
    pix=pg.get_pixmap(matrix=fitz.Matrix(600/72,600/72),alpha=False,colorspace=fitz.csGRAY)
    return norm(np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width).copy())

def binarize(g):
    vals=np.unique(g)
    if len(vals)<=4: return g < ((float(vals.min())+float(vals.max()))/2)
    _,b=cv2.threshold(g,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return b>0

def runs1(a, keep_single=False):
    z=np.flatnonzero(a)
    if not len(z): return []
    out=[]; s=p=int(z[0])
    for q0 in z[1:]:
        q=int(q0)
        if q==p+1: p=q
        else:
            if keep_single or p>s: out.append((s,p))
            s=p=q
    if keep_single or p>s: out.append((s,p))
    return out

def extract(sk):
    H,W=sk.shape; rs=[]; hb={}; vb={}
    hid=np.full((H,W),-1,np.int32); vid=np.full((H,W),-1,np.int32); rid=0
    for y in range(H):
        ids=[]
        for x0,x1 in runs1(sk[y]):
            rs.append(Run(rid,'H',y,x0,x1)); ids.append(rid); hid[y,x0:x1+1]=rid; rid+=1
        if ids: hb[y]=ids
    for x in range(W):
        ids=[]
        for y0,y1 in runs1(sk[:,x]):
            rs.append(Run(rid,'V',x,y0,y1)); ids.append(rid); vid[y0:y1+1,x]=rid; rid+=1
        if ids: vb[x]=ids
    return rs,hb,vb,hid,vid

def endpoint(r,side):
    q=r.start if side==0 else r.end
    return (q,r.axis) if r.o=='H' else (r.axis,q)

def contact(r,side,hid,vid):
    x0,y0=endpoint(r,side); H,W=hid.shape; opp=vid if r.o=='H' else hid
    for yy in range(max(0,y0-1),min(H,y0+2)):
        for xx in range(max(0,x0-1),min(W,x0+2)):
            if opp[yy,xx]>=0: return True
    return False

def annotate(rs,hid,vid):
    for r in rs:
        r.c0=contact(r,0,hid,vid); r.c1=contact(r,1,hid,vid)

def perp_runs(parent_o,axis,pos,rs,hid,vid):
    opp=vid if parent_o=='H' else hid; H,W=opp.shape
    x0,y0=((pos,axis) if parent_o=='H' else (axis,pos)); ids=set()
    for yy in range(max(0,y0-1),min(H,y0+2)):
        for xx in range(max(0,x0-1),min(W,x0+2)):
            rid=int(opp[yy,xx])
            if rid>=0: ids.add(rid)
    return [rs[i] for i in ids]

# ---------------- Axis-relative local connectivity ----------------

def offaxis_endpoint_seeds(r:Run, side:int, sk:np.ndarray):
    """Foreground neighbours entering the candidate gap, excluding parent axis."""
    x0,y0=endpoint(r,side); H,W=sk.shape; out=[]
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if dx==0 and dy==0: continue
            x=x0+dx; y=y0+dy
            if not (0<=x<W and 0<=y<H) or not sk[y,x]: continue
            if r.o=='H':
                if y==r.axis: continue
                if side==1 and x<x0: continue
                if side==0 and x>x0: continue
            else:
                if x==r.axis: continue
                if side==1 and y<y0: continue
                if side==0 and y>y0: continue
            out.append((x,y))
    return tuple(dict.fromkeys(out))

_CONN_CACHE={}

def axis_relative_connected(a:Run,b:Run,sk:np.ndarray,return_seen=False):
    """Connectivity between outer carrier endpoints with ONLY parent axis forbidden.

    Search is constrained longitudinally to [a.end,b.start].  Normal displacement
    is unrestricted, so a complete valve outline can be followed, but the search
    cannot escape before/after the candidate component along the carrier.
    """
    if a.o!=b.o or a.axis!=b.axis or b.start<=a.end: return (False,set()) if return_seen else False
    key=(a.id,b.id)
    if not return_seen and key in _CONN_CACHE: return _CONN_CACHE[key]
    # Scale-free locality prerequisite: the unsupported longitudinal gap may not
    # exceed the observed carrier support immediately bracketing it.  This is not
    # a pixel threshold; it is normalized by the two actual run lengths.
    gap=b.start-a.end-1
    if gap>a.length+b.length:
        if not return_seen: _CONN_CACHE[key]=False
        return (False,set()) if return_seen else False
    S=offaxis_endpoint_seeds(a,1,sk); T=set(offaxis_endpoint_seeds(b,0,sk))
    if not S or not T:
        if not return_seen: _CONN_CACHE[key]=False
        return (False,set()) if return_seen else False
    for z in S:
        if z in T:
            if not return_seen: _CONN_CACHE[key]=True
            return (True,set(S)) if return_seen else True
    H,W=sk.shape; lo=a.end; hi=b.start; axis=a.axis
    dq=deque(S); seen=set(S)
    while dq:
        x,y=dq.popleft()
        if (x,y) in T:
            if not return_seen: _CONN_CACHE[key]=True
            return (True,seen) if return_seen else True
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dx==0 and dy==0: continue
                xx=x+dx; yy=y+dy
                if not (0<=xx<W and 0<=yy<H) or not sk[yy,xx]: continue
                if a.o=='H':
                    if xx<lo or xx>hi or yy==axis: continue
                else:
                    if yy<lo or yy>hi or xx==axis: continue
                z=(xx,yy)
                if z not in seen:
                    seen.add(z); dq.append(z)
    if not return_seen: _CONN_CACHE[key]=False
    return (False,seen) if return_seen else False

def disconnected_boundary_event(a:Run,b:Run,rs,hid,vid):
    """Local disconnected two-boundary case, mainly orifice plates.

    No fixed pixel threshold: the longitudinal gap must be supported by the
    observed normal spans and by the two neighbouring carrier spans themselves.
    """
    LP=perp_runs(a.o,a.axis,a.end,rs,hid,vid)
    RP=perp_runs(a.o,a.axis,b.start,rs,hid,vid)
    if not LP or not RP: return False
    gap=b.start-a.end-1
    normal_support=max(r.length for r in LP)+max(r.length for r in RP)
    return gap>=0 and gap<=normal_support and gap<=a.length+b.length

def base_adjacent_events(rs,by,o,sk,hid,vid):
    per_axis={}; connected_count=0; disconnected_count=0
    for axis,ids0 in by.items():
        ids=sorted(ids0,key=lambda i:rs[i].start)
        flags=[]; src=[]
        for a_id,b_id in zip(ids,ids[1:]):
            a,b=rs[a_id],rs[b_id]
            if b.start<=a.end+1:
                flags.append(False); src.append(''); continue
            if axis_relative_connected(a,b,sk):
                flags.append(True); src.append('AXIS_REL_CONNECTED'); connected_count+=1
            elif disconnected_boundary_event(a,b,rs,hid,vid):
                flags.append(True); src.append('DISCONNECTED_TWO_BOUNDARY'); disconnected_count+=1
            else:
                flags.append(False); src.append('')
        per_axis[axis]=(ids,flags,src)
    return per_axis,connected_count,disconnected_count

def merge_internal_axis_fragments(per_axis,rs,sk,o):
    """Merge adjacent FSM gaps only when OUTER endpoints remain displacement-connected.

    This is the key v34 FSM rule: a same-axis run inside a valve does not close
    COMPONENT if the displacement continuum still connects around it.  A true
    carrier run between two separate components does close the first component,
    because the outer displacement paths are not connected after the parent axis
    is removed.
    """
    out=[]; merge_tests=0; merges=0
    for axis,(ids,flags,src) in per_axis.items():
        n=len(ids); i=0
        while i<n-1:
            if not flags[i]: i+=1; continue
            start=i; end=i+1; sources=[src[i]]
            k=i+1
            while k<n-1 and flags[k]:
                merge_tests+=1
                if axis_relative_connected(rs[ids[start]],rs[ids[k+1]],sk):
                    end=k+1; sources.append(src[k]); merges+=1; k+=1
                else:
                    break
            source='FSM_MERGED_AXIS_REL' if end>start+1 else sources[0]
            out.append(Event(o,axis,ids[start],ids[end],source,end-start))
            # If extension failed at gap k, let that gap become the next event.
            i=end if end>start+1 else start+1
    return out,merge_tests,merges

def sequences(events,rs):
    grouped=defaultdict(list)
    for i,e in enumerate(events): grouped[(e.o,e.axis)].append((i,e))
    seq=[]
    for key,arr in grouped.items():
        adj=defaultdict(list)
        for ei,e in arr:
            adj[e.before].append((e.after,ei)); adj[e.after].append((e.before,ei))
        seen=set()
        for rid in list(adj):
            if rid in seen: continue
            st=[rid]; seen.add(rid); runs=[]; eids=set()
            while st:
                u=st.pop(); runs.append(u)
                for v,ei in adj[u]:
                    eids.add(ei)
                    if v not in seen: seen.add(v); st.append(v)
            runs.sort(key=lambda z:rs[z].start)
            seq.append((key,runs,sorted(eids)))
    return seq

# ---------------- Normal-run graph on RAW skeleton ----------------

def build_normal_run_index(mask,parent_o):
    H,W=mask.shape; idx={}
    if parent_o=='H':
        for x in range(W):
            rr=runs1(mask[:,x],keep_single=True)
            if rr: idx[x]=rr
    else:
        for y in range(H):
            rr=runs1(mask[y,:],keep_single=True)
            if rr: idx[y]=rr
    return idx

def split_axis(iv:Interval,axis:int):
    a,b=iv
    if b<axis or a>axis: return [iv]
    out=[]
    if a<=axis-1: out.append((a,axis-1))
    if axis+1<=b: out.append((axis+1,b))
    return out

def axis_removed_intervals(index,s,axis):
    out=[]
    for iv in index.get(s,()): out.extend(split_axis(iv,axis))
    return out

def interval_touch(a:Interval,b:Interval):
    return max(a[0],b[0]) <= min(a[1],b[1]) + 1

def axis_seed_split(iv:Interval,axis:int):
    return iv[1]==axis-1 or iv[0]==axis+1

def middle_run_graph_bbox(o,axis,s0,s1,index):
    """Run-graph envelope with ONLY selected parent axis removed."""
    lo,hi=sorted((int(s0),int(s1))); nodes={}; bys={}; seeds=[]
    for s in range(lo,hi+1):
        rr=axis_removed_intervals(index,s,axis)
        if not rr: continue
        bys[s]=rr
        for k,iv in enumerate(rr):
            nodes[(s,k)]=iv
            if axis_seed_split(iv,axis): seeds.append((s,k))
    if not seeds: return None,0,0
    seen=set(); tracks=0
    for seed in seeds:
        if seed in seen: continue
        tracks+=1; dq=deque([seed]); seen.add(seed)
        while dq:
            s,k=dq.popleft(); iv=nodes[(s,k)]
            for ns in (s-1,s+1):
                if ns<lo or ns>hi: continue
                for nk,niv in enumerate(bys.get(ns,())):
                    key=(ns,nk)
                    if key not in seen and interval_touch(iv,niv):
                        seen.add(key); dq.append(key)
    if not seen: return None,0,0
    ss=[]; n0=[]; n1=[]
    for s,k in seen:
        iv=nodes[(s,k)]; ss.append(s); n0.append(iv[0]); n1.append(iv[1])
    bb=(min(ss),min(n0),max(ss),max(n1)) if o=='H' else (min(n0),min(ss),max(n1),max(ss))
    return bb,tracks,len(seen)

def terminal_run_graph_bbox(o,axis,s0,direction,index,limit):
    """Axis-relative directional terminal recovery on raw run index."""
    step=1 if direction>0 else -1; s0=int(s0)
    lo,hi=(s0,limit) if step>0 else (limit,s0)
    seeds=[]
    for s in (s0,s0+step):
        if not(lo<=s<=hi): continue
        rr=axis_removed_intervals(index,s,axis)
        for k,iv in enumerate(rr):
            if axis_seed_split(iv,axis): seeds.append((s,k,iv))
        if seeds: break
    if not seeds: return None,0,0
    # Node identity carries the interval itself because split indices are local.
    frontier=seeds[:]; seen={(s,iv) for s,_,iv in seeds}; tracks=len(seeds)
    while frontier:
        nxt=[]
        for cs,_,iv in frontier:
            ns=cs+step
            if ns<lo or ns>hi: continue
            for niv in axis_removed_intervals(index,ns,axis):
                key=(ns,niv)
                if key not in seen and interval_touch(iv,niv):
                    seen.add(key); nxt.append((ns,0,niv))
        frontier=nxt
    ss=[s for s,_ in seen]; n0=[iv[0] for _,iv in seen]; n1=[iv[1] for _,iv in seen]
    bb=(min(ss),min(n0),max(ss),max(n1)) if o=='H' else (min(n0),min(ss),max(n1),max(ss))
    return bb,tracks,len(seen)

def recover_components(events,seqs,rs,indexH,indexV,shape):
    comps=[]; cid=0; H,W=shape
    for e in events:
        a,b=rs[e.before],rs[e.after]; s0=a.end; s1=b.start
        idx=indexH if e.o=='H' else indexV
        bb,tr,nodes=middle_run_graph_bbox(e.o,e.axis,s0,s1,idx)
        if bb is None: continue
        comps.append(Component(cid,e.o,e.axis,'MIDDLE',e.before,e.after,
                    'AXIS_REL_RUN_GRAPH:'+e.source,bb,tr,nodes,min(s0,s1),max(s0,s1))); cid+=1
    for (o,axis),runs,_ in seqs:
        if not runs: continue
        idx=indexH if o=='H' else indexV; first,last=rs[runs[0]],rs[runs[-1]]
        bb,tr,nodes=terminal_run_graph_bbox(o,axis,first.start,-1,idx,0)
        if bb is not None:
            comps.append(Component(cid,o,axis,'TERMINAL_START',None,first.id,'AXIS_REL_TERMINAL',bb,tr,nodes,(bb[0] if o=='H' else bb[1]),first.start)); cid+=1
        lim=W-1 if o=='H' else H-1
        bb,tr,nodes=terminal_run_graph_bbox(o,axis,last.end,+1,idx,lim)
        if bb is not None:
            comps.append(Component(cid,o,axis,'TERMINAL_END',last.id,None,'AXIS_REL_TERMINAL',bb,tr,nodes,last.end,(bb[2] if o=='H' else bb[3]))); cid+=1
    return comps

def draw(g,rs,seqs,comps):
    im=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR); car=set()
    for _,runs,_ in seqs: car.update(runs)
    for rid in car:
        r=rs[rid]
        if r.o=='H': cv2.line(im,(r.start,r.axis),(r.end,r.axis),(0,170,0),1)
        else: cv2.line(im,(r.axis,r.start),(r.axis,r.end),(0,170,0),1)
    for c in comps:
        x1,y1,x2,y2=c.bbox; col=(255,0,255) if c.kind=='MIDDLE' else (255,255,0)
        cv2.rectangle(im,(x1,y1),(x2,y2),col,1)
    return im

def save(out,rs,events,seqs,comps):
    with open(out/'component_candidates.csv','w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['id','orientation','axis','kind','before_run','after_run','source','track_count','run_graph_nodes','longitudinal_start','longitudinal_end','x1','y1','x2','y2','width','height'])
        for c in comps:
            x1,y1,x2,y2=c.bbox
            w.writerow([c.id,c.o,c.axis,c.kind,c.before,c.after,c.source,c.track_count,c.node_count,c.longitudinal_start,c.longitudinal_end,x1,y1,x2,y2,x2-x1+1,y2-y1+1])
    with open(out/'fsm_events.csv','w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['orientation','axis','before_id','before_start','before_end','after_id','after_start','after_end','source','merged_gap_count'])
        for e in events:
            a,b=rs[e.before],rs[e.after]
            w.writerow([e.o,e.axis,a.id,a.start,a.end,b.id,b.start,b.end,e.source,e.span_gaps])
    with open(out/'carrier_sequences.csv','w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['orientation','axis','run_ids','spans','component_transitions'])
        for (o,axis),runs,eids in seqs:
            w.writerow([o,axis,';'.join(map(str,runs)),';'.join(f'{rs[r].start}-{rs[r].end}' for r in runs),len(eids)])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('input'); ap.add_argument('-o','--output',default='AP1000_v34_AXIS_RELATIVE_RESULT')
    a=ap.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    t0=time.time(); g=load(Path(a.input)); ink=binarize(g); sk=skeletonize(ink)
    rs,hb,vb,hid,vid=extract(sk); annotate(rs,hid,vid)

    ph,hc,hd=base_adjacent_events(rs,hb,'H',sk,hid,vid)
    pv,vc,vd=base_adjacent_events(rs,vb,'V',sk,hid,vid)
    eh,mt_h,mg_h=merge_internal_axis_fragments(ph,rs,sk,'H')
    ev,mt_v,mg_v=merge_internal_axis_fragments(pv,rs,sk,'V')
    events=eh+ev; seqs=sequences(events,rs)

    # Raw skeleton indexes: parent axis is removed lazily per event, never globally.
    indexH=build_normal_run_index(sk,'H'); indexV=build_normal_run_index(sk,'V')
    comps=recover_components(events,seqs,rs,indexH,indexV,sk.shape)

    cv2.imwrite(str(out/'00_input.png'),g)
    cv2.imwrite(str(out/'01_binary.png'),ink.astype(np.uint8)*255)
    cv2.imwrite(str(out/'02_skeleton.png'),(~sk).astype(np.uint8)*255)
    middle=[c for c in comps if c.kind=='MIDDLE']
    cv2.imwrite(str(out/'03_middle_axis_relative_components.png'),draw(g,rs,seqs,middle))
    cv2.imwrite(str(out/'04_all_fsm_components_debug.png'),draw(g,rs,seqs,comps))
    save(out,rs,events,seqs,comps)
    with open(out/'middle_component_candidates.csv','w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['id','orientation','axis','before_run','after_run','source','x1','y1','x2','y2','width','height'])
        for c in middle:
            x1,y1,x2,y2=c.bbox
            w.writerow([c.id,c.o,c.axis,c.before,c.after,c.source,x1,y1,x2,y2,x2-x1+1,y2-y1+1])
    summ={
      'image_size':[g.shape[1],g.shape[0]], 'all_hv_runs':len(rs),
      'base_connected_adjacent_events':hc+vc, 'base_disconnected_boundary_events':hd+vd,
      'merge_tests':mt_h+mt_v, 'successful_internal_run_merges':mg_h+mg_v,
      'final_fsm_events':len(events), 'carrier_sequences':len(seqs),
      'middle_components':sum(c.kind=='MIDDLE' for c in comps),
      'terminal_components':sum(c.kind!='MIDDLE' for c in comps), 'total_components':len(comps),
      'runtime_sec':round(time.time()-t0,3),
      'rules':{
        'global_structural_hv_removal':False,
        'displacement_reference':'selected parent axis only',
        'same_axis_internal_run_closes_component':False,
        'merge_rule':'merge adjacent FSM gaps iff outer endpoints remain connected with parent axis forbidden',
        'bbox_recovery':'raw-skeleton normal RLE graph with parent axis split out',
        'global_pixel_flood_fill_for_bbox':False,
        'text_filter':False,'length_cluster':False,'hough_lsd':False,'bbox_margin':0
      }
    }
    (out/'summary.json').write_text(json.dumps(summ,indent=2),encoding='utf-8')
    print(json.dumps(summ,indent=2))

if __name__=='__main__': main()

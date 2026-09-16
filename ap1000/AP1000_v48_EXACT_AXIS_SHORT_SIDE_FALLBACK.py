#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AP1000 v48 — Original Core + Exact-Axis Near-Witness Short-Side Fallback
=====================================================================
Goals
-----
1) A symbol may contain two or more carrier discontinuities.  Short carrier-like
   islands inside one local symbol do NOT split it into multiple components.
   The outer pipe witnesses define ONE component interval.
2) Raster stair-steps / stroke-scaled axis drift (about 5 px on this sheet) are consolidated into one tolerant H/V
   witness before displacement detection (e.g. TE017B-style 26 px + 25 px pieces).
3) v45/v46 high-precision connected and multipart geometry are preserved first;
   a whole-carrier fallback handles locally structured but internally disconnected
   symbols such as check-valve-like forms.
4) No OCR/tag/class-specific detection logic.

The elevated line tolerance is used for carrier identity only.  Actual displacement
coordinates always come from the ORIGINAL extreme endpoint runs.
"""
from __future__ import annotations
import argparse, bisect, csv, json, math, sys, time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
import AP1000_v34_AXIS_RELATIVE_FSM_RUN_GRAPH as v34
import AP1000_v37_TOLERANT_LONGEST_TRACK_HIERARCHY as v37
import AP1000_v44_LATENT_CARRIER_DILATION_BBOX as v44
import AP1000_v45_PRECISION_PIPE_ATTACHED_BBOX as v45
import AP1000_v46_MULTIPART_PIPE_ATTACHED_BBOX as v46

BBox=Tuple[int,int,int,int]

@dataclass
class LaneWitness:
    id:int
    o:str
    member_ids:List[int]
    start:int
    end:int
    axis:float
    span:int
    ink:int
    density:float
    left_run:int
    right_run:int

@dataclass
class Proposal:
    o:str
    before_group:int
    after_group:int
    before_run:int
    after_run:int
    gap_count:int
    internal_group_ids:List[int]
    source:str
    geometry_mode:str
    symbol_bbox:BBox
    symbol_area:int
    symbol_offaxis_extent:int
    fragment_count:int=0
    graph_hops:int=0
    side_balance:float=1.0

@dataclass
class Candidate:
    id:int; o:str
    before_group:int; after_group:int
    before_run:int; after_run:int
    before_axis:int; after_axis:int
    before_support:int; after_support:int
    displacement_start:int; displacement_end:int
    gap_count:int; internal_group_count:int
    geometry_mode:str; proposal_source:str
    symbol_area:int; symbol_offaxis_extent:int
    fragment_count:int; graph_hops:int; side_balance:float
    bbox:BBox
    status:str='PENDING'; owner:Optional[int]=None; reason:str=''
    @property
    def displacement_span(self): return self.displacement_end-self.displacement_start+1
    @property
    def bbox_area(self):
        x1,y1,x2,y2=self.bbox; return (x2-x1+1)*(y2-y1+1)


def stroke_width(median_radius:float)->float:
    return max(1.0,2.0*float(median_radius)+1.0)

def pipe_witness_min_px(median_radius:float,factor:float)->int:
    return max(8,int(math.ceil(float(factor)*stroke_width(median_radius))))

def effective_axis_tolerance(base_margin:int, sw:float, factor:float)->int:
    return max(int(base_margin), int(math.ceil(float(factor)*float(sw))))


def _uf_find(parent,x):
    while parent[x]!=x:
        parent[x]=parent[parent[x]]; x=parent[x]
    return x

def _uf_union(parent,a,b):
    a=_uf_find(parent,a); b=_uf_find(parent,b)
    if a!=b: parent[b]=a


def build_tolerant_lane_witnesses(rs, active, axis_tol:int, join_gap:int,
                                   min_raw_len:int, witness_min:int,
                                   min_density:float=0.72, base_axis_margin:int=1):
    """Consolidate H/V stair-step pieces before pipe-witness qualification.

    Only longitudinally overlapping/touching pieces within axis_tol are unioned.
    Component-sized gaps are never bridged here, so dilation does not erase the
    displacement we are trying to detect.
    """
    ids=[rid for rid in active if rs[rid].length>=min_raw_len]
    parent={rid:rid for rid in ids}
    bins={'H':defaultdict(list),'V':defaultdict(list)}; starts={'H':{},'V':{}}
    for rid in ids:
        r=rs[rid]; bins[r.o][int(r.axis)].append(rid)
    for o in ('H','V'):
        for ax,L in bins[o].items():
            L.sort(key=lambda i:rs[i].start); starts[o][ax]=[rs[i].start for i in L]

    for rid in ids:
        r=rs[rid]
        for ax in range(int(r.axis)-axis_tol,int(r.axis)+axis_tol+1):
            L=bins[r.o].get(ax)
            if not L: continue
            ss=starts[r.o][ax]
            k=bisect.bisect_left(ss,int(r.start)-join_gap)
            while k<len(L) and rs[L[k]].start<=int(r.end)+join_gap:
                qid=L[k]; q=rs[qid]
                if qid!=rid:
                    gap=max(0,max(int(r.start),int(q.start))-min(int(r.end),int(q.end))-1)
                    overlap=min(int(r.end),int(q.end))-max(int(r.start),int(q.start))+1
                    axis_diff=abs(int(r.axis)-int(q.axis))
                    # Same-axis fragments can overlap freely.  For shifted axes,
                    # merge only endpoint-adjacent stair steps; do NOT merge long
                    # parallel lines merely because they fall inside the tolerance.
                    stair_step = (axis_diff <= int(base_axis_margin)) or (overlap <= max(1,join_gap))
                    if gap<=join_gap and stair_step:
                        _uf_union(parent,rid,qid)
                k+=1

    raw_groups=defaultdict(list)
    for rid in ids: raw_groups[_uf_find(parent,rid)].append(rid)
    groups=[]
    for mem in raw_groups.values():
        mem=sorted(mem,key=lambda i:(rs[i].start,rs[i].end,rs[i].axis))
        o=rs[mem[0]].o; st=min(rs[i].start for i in mem); en=max(rs[i].end for i in mem)
        ints=sorted((int(rs[i].start),int(rs[i].end)) for i in mem)
        cov=0; s,e=ints[0]
        for a,b in ints[1:]:
            if a<=e+1: e=max(e,b)
            else: cov+=e-s+1; s,e=a,b
        cov+=e-s+1
        span=en-st+1; weights=[rs[i].length for i in mem]
        axis=sum(float(rs[i].axis)*w for i,w in zip(mem,weights))/max(1,sum(weights))
        left=min(mem,key=lambda i:(rs[i].start,abs(rs[i].axis-axis),-rs[i].end))
        right=max(mem,key=lambda i:(rs[i].end,-abs(rs[i].axis-axis),-rs[i].start))
        groups.append(LaneWitness(len(groups),o,mem,int(st),int(en),float(axis),int(span),int(cov),float(cov/max(1,span)),left,right))

    eligible=[g.id for g in groups if g.span>=witness_min and g.density>=min_density]
    return groups,eligible,{
        'tolerant_lane_groups':len(groups),'pipe_witness_groups':len(eligible),
        'lane_join_gap_px':int(join_gap),'lane_min_density':float(min_density),
        'pipe_witness_min_px':int(witness_min)
    }


def _boundary_runs(group:LaneWitness, rs, side:int):
    # side 1 = right/end boundary of before group; side 0 = left/start boundary of after group
    return rs[group.right_run] if side==1 else rs[group.left_run]


def tolerant_endpoint_seeds(r, side:int, ink, tol:int):
    exact=list(v34.offaxis_endpoint_seeds(r,side,ink))
    if exact: return tuple(exact)
    H,W=ink.shape
    if r.o=='H':
        x0=int(r.end if side==1 else r.start); y0=int(r.axis)
        xs=range(max(0,x0-tol),min(W-1,x0+tol)+1)
        ys=range(max(0,y0-tol),min(H-1,y0+tol)+1)
        out=[]
        for y in ys:
            for x in xs:
                if not ink[y,x]: continue
                if abs(y-y0)<=max(0,tol//2): continue
                if side==1 and x<x0-tol: continue
                if side==0 and x>x0+tol: continue
                out.append((x,y))
        return tuple(dict.fromkeys(out))
    x0=int(r.axis); y0=int(r.end if side==1 else r.start)
    xs=range(max(0,x0-tol),min(W-1,x0+tol)+1)
    ys=range(max(0,y0-tol),min(H-1,y0+tol)+1)
    out=[]
    for y in ys:
        for x in xs:
            if not ink[y,x]: continue
            if abs(x-x0)<=max(0,tol//2): continue
            if side==1 and y<y0-tol: continue
            if side==0 and y>y0+tol: continue
            out.append((x,y))
    return tuple(dict.fromkeys(out))


def _gap_has_component_evidence(a,b,ink,axis_tol:int):
    if b.start<=a.end+1: return False
    if not tolerant_endpoint_seeds(a,1,ink,axis_tol): return False
    if not tolerant_endpoint_seeds(b,0,ink,axis_tol): return False
    return v44._offaxis_ink_in_gap(a.o,a,b,ink,axis_tol)


def _local_axis(a,b,s):
    s0=float(a.end); s1=float(b.start)
    if s1<=s0: return int(round((a.axis+b.axis)/2))
    t=(float(s)-s0)/(s1-s0)
    return int(round((1-t)*a.axis+t*b.axis))


def _mask_foreign_boundary_orthogonal(work, o, a, b, rs, active, witness_min,
                                      axis_tol, sw, x0,y0):
    """Clip long orthogonal lines entering through an OUTER boundary.

    A valve actuator in the middle of a displacement is retained.  Only a long
    orthogonal H/V line whose longitudinal coordinate sits at one of the outer
    pipe endpoints is clipped away from the local baseline neighborhood.
    """
    H,W=work.shape
    lo,hi=int(a.end),int(b.start)
    keep=int(math.ceil(max(axis_tol+1,2.5*sw)))
    if o=='H':
        base=int(round((a.axis+b.axis)/2))
        for rid in active:
            r=rs[rid]
            if r.o!='V' or r.length<witness_min: continue
            gx=int(r.axis)
            if min(abs(gx-lo),abs(gx-hi))>axis_tol+1: continue
            if r.end<base or r.start>base: continue
            xx=gx-x0
            if not (0<=xx<W): continue
            ya=max(0,int(r.start)-y0); yb=min(H-1,int(r.end)-y0)
            for yy in range(ya,yb+1):
                gy=y0+yy
                if abs(gy-base)>keep: work[yy,xx]=0
    else:
        base=int(round((a.axis+b.axis)/2))
        for rid in active:
            r=rs[rid]
            if r.o!='H' or r.length<witness_min: continue
            gy=int(r.axis)
            if min(abs(gy-lo),abs(gy-hi))>axis_tol+1: continue
            if r.end<base or r.start>base: continue
            yy=gy-y0
            if not (0<=yy<H): continue
            xa=max(0,int(r.start)-x0); xb=min(W-1,int(r.end)-x0)
            for xx in range(xa,xb+1):
                gx=x0+xx
                if abs(gx-base)>keep: work[yy,xx]=0


def _longest_runs(mask:np.ndarray):
    mh=0; mv=0
    for row in mask:
        z=np.flatnonzero(row)
        if len(z):
            s=p=int(z[0])
            for q0 in z[1:]:
                q=int(q0)
                if q==p+1: p=q
                else: mh=max(mh,p-s+1); s=p=q
            mh=max(mh,p-s+1)
    for col in mask.T:
        z=np.flatnonzero(col)
        if len(z):
            s=p=int(z[0])
            for q0 in z[1:]:
                q=int(q0)
                if q==p+1: p=q
                else: mv=max(mv,p-s+1); s=p=q
            mv=max(mv,p-s+1)
    return mh,mv


def _bbox_gap(A,B):
    ax1,ay1,ax2,ay2=A; bx1,by1,bx2,by2=B
    dx=max(0,max(ax1,bx1)-min(ax2,bx2)-1)
    dy=max(0,max(ay1,by1)-min(ay2,by2)-1)
    return dx,dy


def whole_carrier_geometry(o,a,b,ink,base_margin,axis_tol,sw,rs,active,witness_min):
    """Recover ONE bbox for the entire outer-carrier interval.

    It does not require all symbol fragments to be connected.  Baseline-touching
    structural fragments are primary.  Nearby strong H/V satellite fragments are
    added, while long orthogonal lines entering at an outer boundary are clipped.
    """
    H0,W0=ink.shape; lo=int(a.end); hi=int(b.start)
    if hi<=lo: return None
    span=hi-lo+1; center=int(round((a.axis+b.axis)/2))
    lim=max(int(math.ceil(2.0*span)),int(math.ceil(8.0*sw)))
    if o=='H': x0=max(0,lo);x1=min(W0-1,hi);y0=max(0,center-lim);y1=min(H0-1,center+lim)
    else: y0=max(0,lo);y1=min(H0-1,hi);x0=max(0,center-lim);x1=min(W0-1,center+lim)
    crop=ink[y0:y1+1,x0:x1+1].astype(np.uint8)
    if crop.size==0 or not np.any(crop): return None
    work=crop.copy()
    _mask_foreign_boundary_orthogonal(work,o,a,b,rs,active,witness_min,axis_tol,sw,x0,y0)

    k=max(1,2*int(base_margin)+1)
    repair=cv2.dilate(work,np.ones((k,k),np.uint8),1) if k>1 else work
    n,labels,stats,_=cv2.connectedComponentsWithStats(repair,8)
    if n<=1: return None

    corridor=max(axis_tol+1,int(math.ceil(1.5*sw)))
    min_off=max(base_margin+1,int(math.ceil(0.75*sw)))
    info={}
    seed=set()
    for z in range(1,n):
        m=(labels==z)&(work>0)
        pts=np.argwhere(m)
        if len(pts)==0: continue
        yy0,xx0=pts.min(axis=0); yy1,xx1=pts.max(axis=0)
        bb=(x0+int(xx0),y0+int(yy0),x0+int(xx1),y0+int(yy1))
        neg=pos=0; baseline_hit=False
        for yy,xx in pts:
            gx=x0+int(xx); gy=y0+int(yy); s=gx if o=='H' else gy; normal=gy if o=='H' else gx
            d=normal-_local_axis(a,b,s)
            if abs(d)<=corridor: baseline_hit=True
            if d<0: neg=max(neg,-d)
            else: pos=max(pos,d)
        mh,mv=_longest_runs(m)
        info[z]={'bbox':bb,'area':int(len(pts)),'neg':int(neg),'pos':int(pos),'off':int(max(neg,pos)),
                 'mh':int(mh),'mv':int(mv),'baseline':baseline_hit,'mask':m}
        if baseline_hit and max(neg,pos)>=min_off:
            seed.add(z)
    if not seed: return None

    chosen=set(seed)
    satellite_count=0
    structural_min=max(int(math.ceil(4.0*sw)),int(math.ceil(0.30*span)))
    sat_gap=max(axis_tol+1,int(math.ceil(2.0*sw)))
    # Add at most two structural satellite layers. This recovers split plates /
    # triangle bases but not nearby text, whose straight H/V strokes are short.
    for _ in range(2):
        add=[]
        for z,d in info.items():
            if z in chosen: continue
            if max(d['mh'],d['mv'])<structural_min: continue
            for q in chosen:
                dx,dy=_bbox_gap(d['bbox'],info[q]['bbox'])
                if dx<=sat_gap and dy<=sat_gap:
                    add.append(z); break
        if not add: break
        chosen.update(add); satellite_count += len(add)

    if satellite_count < 1:
        return None

    mask=np.zeros_like(work,dtype=bool)
    for z in chosen: mask |= info[z]['mask']
    pts=np.argwhere(mask)
    if len(pts)==0: return None
    yy0,xx0=pts.min(axis=0); yy1,xx1=pts.max(axis=0)
    if o=='H':
        bb=(lo,y0+int(yy0),hi,y0+int(yy1))
        off=max(abs(bb[1]-int(a.axis)),abs(bb[3]-int(a.axis)),abs(bb[1]-int(b.axis)),abs(bb[3]-int(b.axis)))
    else:
        bb=(x0+int(xx0),lo,x0+int(xx1),hi)
        off=max(abs(bb[0]-int(a.axis)),abs(bb[2]-int(a.axis)),abs(bb[0]-int(b.axis)),abs(bb[2]-int(b.axis)))
    if off<min_off: return None
    return bb,int(np.count_nonzero(mask)),int(off),int(len(chosen))


def _candidate_geometry(o,a,b,ink,base_margin,axis_tol,sw,rs,active,witness_min,
                        fragment_gap_strokes,min_side_balance,two_fragment_ratio):
    # Preserve high-precision v45 first.
    x=v45.bridge_component(o,a,b,ink,base_margin,sw)
    if x is not None:
        bb,area,off=x
        # force exact outer longitudinal interval
        if o=='H': bb=(int(a.end),bb[1],int(b.start),bb[3])
        else: bb=(bb[0],int(a.end),bb[2],int(b.start))
        return 'CONNECTED',bb,area,off,1,0,1.0
    x=v46.multipart_component(o,a,b,ink,base_margin,sw,fragment_gap_strokes,min_side_balance,two_fragment_ratio)
    if x is not None:
        bb,area,off,nfrag,hops,balance=x
        if o=='H': bb=(int(a.end),bb[1],int(b.start),bb[3])
        else: bb=(bb[0],int(a.end),bb[2],int(b.start))
        return 'MULTIPART',bb,area,off,nfrag,hops,balance
    x=whole_carrier_geometry(o,a,b,ink,base_margin,axis_tol,sw,rs,active,witness_min)
    if x is None: return None
    bb,area,off,nfrag=x
    return 'WHOLE_CARRIER',bb,area,off,nfrag,0,1.0


def build_whole_carrier_proposals(rs,active,ink,base_margin,median_radius,
                                  axis_tolerance_strokes=1.25,witness_factor=8.0,
                                  max_displacement_strokes=30.0,max_gap_fraction=0.05,
                                  lane_join_strokes=0.75,lane_min_density=0.72,
                                  internal_island_max_strokes=24.0,
                                  fragment_gap_strokes=6.0,min_side_balance=0.30,
                                  two_fragment_ratio=1.50,min_raw_len=3,
                                  embedded_fraction=0.55):
    """Build component intervals from CLEAN outer carrier witnesses.

    Critical rule: one bounded carrier interval == one component, even if the
    symbol contains 2+ baseline discontinuities and/or carrier-like islands.
    Internal witnesses are NOT turned into separate components when their middle
    is embedded in off-axis symbol geometry.  The first subsequent CLEAN pipe
    witness closes the interval; this keeps two nearby valves separate when a
    real pipe segment exists between them.
    """
    sw=stroke_width(median_radius)
    axis_tol=effective_axis_tolerance(base_margin,sw,axis_tolerance_strokes)
    join_gap=max(base_margin,int(math.ceil(lane_join_strokes*sw)))
    witness=pipe_witness_min_px(median_radius,witness_factor)
    groups,eligible,gstats=build_tolerant_lane_witnesses(
        rs,active,axis_tol,join_gap,min_raw_len,witness,lane_min_density,base_margin)
    eligible_set=set(eligible)

    # Classify eligible H/V witnesses.  A true outer pipe is mostly straight in
    # its central span.  A carrier-like island inside a symbol has substantial
    # off-axis ink through the middle and is therefore 'embedded'.
    embedded={}
    emb_detail={}
    for gid in eligible:
        frac,off=_central_embeddedness(groups[gid],ink,sw,axis_tol)
        is_emb=(frac>=float(embedded_fraction) and off>=int(math.ceil(2.0*sw)))
        embedded[gid]=bool(is_emb)
        emb_detail[gid]=(float(frac),int(off))

    gbins={'H':defaultdict(list),'V':defaultdict(list)}
    for gid in eligible:
        g=groups[gid]; gbins[g.o][int(round(g.axis))].append(gid)
    for o in ('H','V'):
        for ax,L in gbins[o].items(): L.sort(key=lambda z:groups[z].start)

    horizon=max(witness*3,int(round(min(ink.shape)*max_gap_fraction)))
    max_total=max(witness,int(math.ceil(max_displacement_strokes*sw)))

    def candidate_successors(G):
        ids=[]
        for ax in range(int(round(G.axis))-axis_tol,int(round(G.axis))+axis_tol+1):
            ids.extend(gbins[G.o].get(ax,()))
        ids=[gid for gid in set(ids) if groups[gid].start>G.end]
        ids.sort(key=lambda gid:(groups[gid].start,abs(groups[gid].axis-G.axis),-groups[gid].span))
        return ids

    def whole_interval_evidence(a,b):
        if b.start<=a.end+1:return False
        if not tolerant_endpoint_seeds(a,1,ink,axis_tol):return False
        if not tolerant_endpoint_seeds(b,0,ink,axis_tol):return False
        return v44._offaxis_ink_in_gap(a.o,a,b,ink,axis_tol)

    out=[]; reject_geom=0; tol_candidates=0
    # Start only from clean pipe witnesses.  Embedded witnesses are internal
    # symbol structure, never outer boundaries on their own.
    for aid in eligible:
        if embedded.get(aid,False):
            continue
        A=groups[aid]; a=_boundary_runs(A,rs,1)
        internals=[]; Bgid=None
        for gid in candidate_successors(A):
            G=groups[gid]
            b0=_boundary_runs(G,rs,0)
            total=int(b0.start)-int(a.end)+1
            if total>max_total:
                break
            # Ignore overlapping/near-touching witness fragments; lane grouping
            # should already have merged stair steps at <= join_gap.
            if b0.start<=a.end+1:
                continue
            # An embedded carrier-like island belongs to the SAME component.
            if embedded.get(gid,False):
                internals.append(gid)
                continue
            # First clean witness after any internal islands closes the component.
            # Require local symbol evidence at BOTH outer pipe endpoints.
            if 2<=b0.start-a.end-1<=horizon and whole_interval_evidence(a,b0):
                Bgid=gid
            break
        if Bgid is None:
            continue
        B=groups[Bgid]; b=_boundary_runs(B,rs,0)
        geom=_candidate_geometry(A.o,a,b,ink,base_margin,axis_tol,sw,rs,active,witness,
                                 fragment_gap_strokes,min_side_balance,two_fragment_ratio)
        if geom is None:
            reject_geom+=1; continue
        mode,bb,area,off,nfrag,hops,balance=geom
        # Count actual baseline blank islands inside the whole outer interval,
        # rather than equating 'component' with just one gap.
        tmp=Candidate(-1,A.o,A.id,B.id,a.id,b.id,int(a.axis),int(b.axis),A.span,B.span,
                      int(a.end),int(b.start),1,len(internals),mode,'TMP',area,off,nfrag,hops,balance,bb)
        gap_count=max(1,_count_baseline_blank_runs(tmp,ink,axis_tol))
        if abs(a.axis-b.axis)>base_margin or len(A.member_ids)>1 or len(B.member_ids)>1:
            tol_candidates+=1
        out.append(Proposal(A.o,A.id,B.id,a.id,b.id,gap_count,list(internals),
                            'TOLERANT_WHOLE_CARRIER',mode,bb,area,off,nfrag,hops,balance))

    # Same outer interval should appear once. Prefer the candidate that explains
    # more internal discontinuities, then stronger geometry mode.
    mode_rank={'CONNECTED':3,'MULTIPART':2,'WHOLE_CARRIER':1}
    uniq={}
    for p in out:
        a=rs[p.before_run];b=rs[p.after_run]
        key=(p.o,int(a.end),int(b.start),round((a.axis+b.axis)/2))
        cur=uniq.get(key)
        if cur is None or (p.gap_count,mode_rank.get(p.geometry_mode,0),p.symbol_area)>(cur.gap_count,mode_rank.get(cur.geometry_mode,0),cur.symbol_area):
            uniq[key]=p
    out=list(uniq.values())
    out.sort(key=lambda p:(p.o,rs[p.before_run].axis,rs[p.before_run].start,rs[p.after_run].start))
    stats={**gstats,
           'measured_stroke_width_px':round(sw,3),'base_line_margin_px':int(base_margin),
           'effective_axis_tolerance_px':int(axis_tol),'axis_tolerance_strokes':float(axis_tolerance_strokes),
           'lane_join_strokes':float(lane_join_strokes),'max_displacement_strokes':float(max_displacement_strokes),
           'max_displacement_px':int(max_total),'internal_island_max_strokes':float(internal_island_max_strokes),
           'internal_island_max_px':int(math.ceil(internal_island_max_strokes*sw)),
           'embedded_carrier_fraction':float(embedded_fraction),
           'embedded_pipe_witness_groups':int(sum(embedded.values())),
           'whole_carrier_proposals':len(out),
           'multi_gap_proposals':int(sum(p.gap_count>1 for p in out)),
           'proposals_using_tolerant_or_merged_witness':int(tol_candidates),
           'rejected_geometry_after_carrier_evidence':int(reject_geom)}
    return out,groups,eligible,stats

def build_candidates(ps,groups,rs):
    out=[]
    for p in ps:
        A=groups[p.before_group];B=groups[p.after_group];a=rs[p.before_run];b=rs[p.after_run]
        c=Candidate(len(out),p.o,p.before_group,p.after_group,p.before_run,p.after_run,
                    int(a.axis),int(b.axis),int(A.span),int(B.span),int(a.end),int(b.start),
                    int(p.gap_count),len(p.internal_group_ids),p.geometry_mode,p.source,
                    int(p.symbol_area),int(p.symbol_offaxis_extent),int(p.fragment_count),int(p.graph_hops),float(p.side_balance),p.symbol_bbox)
        c.reason=('WHOLE_CARRIER_MULTI_GAP' if p.gap_count>1 else 'WHOLE_CARRIER_SINGLE_INTERVAL')+';'+p.geometry_mode
        out.append(c)
    return out


def _intersects(A,B):
    return not(A[2]<B[0] or B[2]<A[0] or A[3]<B[1] or B[3]<A[1])
def _center(c): return ((c.bbox[0]+c.bbox[2])//2,(c.bbox[1]+c.bbox[3])//2)
def duplicate(a,b,axis_tol):
    if a.o==b.o and max(abs(a.before_axis-b.before_axis),abs(a.after_axis-b.after_axis))<=axis_tol:
        alo,ahi=sorted((a.displacement_start,a.displacement_end));blo,bhi=sorted((b.displacement_start,b.displacement_end))
        inter=max(0,min(ahi,bhi)-max(alo,blo)+1)
        if inter>0 and inter>=0.6*min(ahi-alo+1,bhi-blo+1): return True
    if not _intersects(a.bbox,b.bbox): return False
    ax,ay=_center(a);bx,by=_center(b);A=a.bbox;B=b.bbox
    return B[0]<=ax<=B[2] and B[1]<=ay<=B[3] and A[0]<=bx<=A[2] and A[1]<=by<=A[3]

def resolve(cands,axis_tol):
    mode_rank={'CONNECTED':3,'MULTIPART':2,'WHOLE_CARRIER':1}
    order=sorted(range(len(cands)),key=lambda i:(cands[i].gap_count,mode_rank.get(cands[i].geometry_mode,0),min(cands[i].before_support,cands[i].after_support),cands[i].symbol_area,-cands[i].bbox_area),reverse=True)
    acc=[]
    for i in order:
        c=cands[i];owner=None
        for j in acc:
            if duplicate(c,cands[j],axis_tol):owner=j;break
        if owner is None:
            c.status='ACCEPTED';c.reason+=';ACCEPTED';acc.append(i)
        else:
            c.status='DUPLICATE';c.owner=cands[owner].id;c.reason='SAME_LOCAL_WHOLE_CARRIER_COMPONENT'
    return acc



def _central_embeddedness(G:LaneWitness, ink, sw:float, axis_tol:int):
    """How strongly a carrier witness is embedded inside symbol geometry.

    Clean pipe normally has off-axis ink only near its endpoints.  A carrier-like
    island inside a valve/instrument has off-axis geometry through much of its
    middle.  The score therefore uses only the central part of the witness.
    """
    H,W=ink.shape
    trim=max(int(math.ceil(2.0*sw)), int(round(0.20*G.span)))
    s0=int(G.start+trim); s1=int(G.end-trim)
    if s1<s0:
        s0,s1=int(G.start),int(G.end)
    corridor=max(int(axis_tol)+1,int(math.ceil(1.25*sw)))
    normal_limit=max(corridor+1,int(math.ceil(8.0*sw)))
    hit_cols=0; max_off=0
    n=max(1,s1-s0+1)
    ax=int(round(G.axis))
    if G.o=='H':
        for x in range(max(0,s0),min(W-1,s1)+1):
            ya=max(0,ax-normal_limit); yb=min(H-1,ax+normal_limit)
            ys=np.flatnonzero(ink[ya:yb+1,x])
            ds=[abs((ya+int(yy))-ax) for yy in ys if abs((ya+int(yy))-ax)>corridor]
            if ds:
                hit_cols+=1; max_off=max(max_off,max(ds))
    else:
        for y in range(max(0,s0),min(H-1,s1)+1):
            xa=max(0,ax-normal_limit); xb=min(W-1,ax+normal_limit)
            xs=np.flatnonzero(ink[y,xa:xb+1])
            ds=[abs((xa+int(xx))-ax) for xx in xs if abs((xa+int(xx))-ax)>corridor]
            if ds:
                hit_cols+=1; max_off=max(max_off,max(ds))
    return float(hit_cols/max(1,n)),int(max_off)


def _group_component_bbox(G:LaneWitness, rs, ink, sw:float, base_margin:int):
    """BBox of foreground actually connected to one tolerant carrier island.

    The crop is limited to the island's own longitudinal span.  This deliberately
    prevents a nearby annotation or long branch outside that span from growing the
    bbox.  A tiny repair dilation only heals raster gaps; it is not global flood.
    """
    H,W=ink.shape; pad=max(4,int(math.ceil(10.0*sw))); ax=int(round(G.axis))
    if G.o=='H':
        x0=max(0,int(G.start)); x1=min(W-1,int(G.end)); y0=max(0,ax-pad); y1=min(H-1,ax+pad)
    else:
        y0=max(0,int(G.start)); y1=min(H-1,int(G.end)); x0=max(0,ax-pad); x1=min(W-1,ax+pad)
    crop=ink[y0:y1+1,x0:x1+1].astype(np.uint8)
    if crop.size==0 or not np.any(crop): return None
    k=max(1,2*int(base_margin)+1)
    rep=cv2.dilate(crop,np.ones((k,k),np.uint8),1) if k>1 else crop
    n,lab,_,_=cv2.connectedComponentsWithStats(rep,8)
    labels=set()
    for rid in G.member_ids:
        r=rs[rid]
        if G.o=='H':
            yy=int(r.axis)-y0
            if not(0<=yy<crop.shape[0]): continue
            for gx in range(max(int(r.start),x0),min(int(r.end),x1)+1):
                xx=gx-x0
                if crop[yy,xx]:
                    z=int(lab[yy,xx])
                    if z: labels.add(z)
        else:
            xx=int(r.axis)-x0
            if not(0<=xx<crop.shape[1]): continue
            for gy in range(max(int(r.start),y0),min(int(r.end),y1)+1):
                yy=gy-y0
                if crop[yy,xx]:
                    z=int(lab[yy,xx])
                    if z: labels.add(z)
    if not labels: return None
    mask=np.isin(lab,list(labels))&(crop>0)
    pts=np.argwhere(mask)
    if len(pts)==0:return None
    yy0,xx0=pts.min(axis=0); yy1,xx1=pts.max(axis=0)
    return (x0+int(xx0),y0+int(yy0),x0+int(xx1),y0+int(yy1)),int(len(pts))


def _union_bbox(A:BBox,B:BBox)->BBox:
    return (min(A[0],B[0]),min(A[1],B[1]),max(A[2],B[2]),max(A[3],B[3]))


def _count_baseline_blank_runs(c:Candidate, ink, axis_tol:int):
    lo=int(c.displacement_start); hi=int(c.displacement_end)
    if hi<lo:return 0
    H,W=ink.shape; occ=[]
    for s in range(lo,hi+1):
        t=0.0 if hi==lo else (s-lo)/(hi-lo)
        ax=int(round((1.0-t)*c.before_axis+t*c.after_axis))
        if c.o=='H':
            ya=max(0,ax-axis_tol);yb=min(H-1,ax+axis_tol)
            occ.append(bool(0<=s<W and np.any(ink[ya:yb+1,s])))
        else:
            xa=max(0,ax-axis_tol);xb=min(W-1,ax+axis_tol)
            occ.append(bool(0<=s<H and np.any(ink[s,xa:xb+1])))
    gaps=0;i=0
    while i<len(occ):
        if occ[i]: i+=1; continue
        gaps+=1
        while i<len(occ) and not occ[i]: i+=1
    return gaps


def expand_internal_carrier_islands(c:Candidate, groups, eligible, rs, ink,
                                    base_margin:int, axis_tol:int, sw:float,
                                    max_island_strokes:float=24.0,
                                    embedded_fraction:float=0.55,
                                    max_steps:int=4):
    """Absorb carrier-like islands that are visually part of ONE component.

    This fixes the key multi-gap case: A --gap-- island --gap-- B is one symbol,
    not two detections.  A clean pipe segment is never absorbed merely because it
    is long: the island must be local AND have strong off-axis geometry through
    its central span.
    """
    run_to_group={rid:G.id for G in groups for rid in G.member_ids}
    eligible_set=set(eligible)
    island_max=max(1,int(math.ceil(max_island_strokes*sw)))
    cache={}
    def emb(gid):
        if gid is None:return False
        if gid not in cache:
            G=groups[gid]; frac,off=_central_embeddedness(G,ink,sw,axis_tol)
            cache[gid]=(G.span<=island_max and frac>=embedded_fraction and off>=int(math.ceil(2.0*sw)),frac,off)
        return cache[gid][0]
    def nearest(gid,direction):
        if gid is None:return None
        G=groups[gid]; best=None
        for Q in groups:
            if Q.id==gid or Q.o!=G.o or abs(Q.axis-G.axis)>axis_tol: continue
            if direction<0 and Q.end<G.start:
                gap=G.start-Q.end-1
            elif direction>0 and Q.start>G.end:
                gap=Q.start-G.end-1
            else: continue
            if gap<1: continue
            key=(gap,abs(Q.axis-G.axis),-Q.span)
            if best is None or key<best[0]: best=(key,Q.id)
        return None if best is None else best[1]

    left=run_to_group.get(int(c.before_run)); right=run_to_group.get(int(c.after_run))
    old_bbox=c.bbox; bb=old_bbox; changed=False; absorbed=[]

    # Absorb an embedded boundary witness itself only when the original
    # displacement endpoint lies *inside* that tolerant witness.  This is the
    # signature of a raster-fragmented carrier island that belongs to the
    # component (V044-style), rather than a clean outer pipe that simply has
    # nearby off-axis geometry (R01A-style).  The tolerance is scale-normalized.
    boundary_slack=max(int(base_margin)+1,int(math.ceil(1.5*sw)))
    left_absorbed=False; right_absorbed=False
    if left is not None and emb(left):
        G=groups[left]
        inward_extra=int(G.end)-int(c.displacement_start)
        if inward_extra>=boundary_slack:
            c.displacement_start=min(c.displacement_start,int(G.start)); changed=True; absorbed.append(left); left_absorbed=True
            gb=_group_component_bbox(G,rs,ink,sw,base_margin)
            if gb: bb=_union_bbox(bb,gb[0])
    if right is not None and emb(right):
        G=groups[right]
        inward_extra=int(c.displacement_end)-int(G.start)
        if inward_extra>=boundary_slack:
            c.displacement_end=max(c.displacement_end,int(G.end)); changed=True; absorbed.append(right); right_absorbed=True
            gb=_group_component_bbox(G,rs,ink,sw,base_margin)
            if gb: bb=_union_bbox(bb,gb[0])

    # Walk outward through additional embedded islands only when the boundary
    # group was actually absorbed above.  Stop at the first clean pipe witness,
    # using its actual endpoint as the outer component boundary.
    cur=left
    if cur is not None and left_absorbed:
        for _ in range(max_steps):
            prev=nearest(cur,-1)
            if prev is None: break
            A=groups[prev];B=groups[cur];a=_boundary_runs(A,rs,1);b=_boundary_runs(B,rs,0)
            if b.start-a.end-1>max(2,int(math.ceil(30.0*sw))): break
            if not _gap_has_component_evidence(a,b,ink,axis_tol): break
            if emb(prev):
                c.displacement_start=min(c.displacement_start,int(A.start)); absorbed.append(prev);changed=True
                gb=_group_component_bbox(A,rs,ink,sw,base_margin)
                if gb:bb=_union_bbox(bb,gb[0])
                cur=prev;continue
            if prev in eligible_set:
                c.displacement_start=min(c.displacement_start,int(a.end));changed=True
            break

    cur=right
    if cur is not None and right_absorbed:
        for _ in range(max_steps):
            nxt=nearest(cur,+1)
            if nxt is None: break
            A=groups[cur];B=groups[nxt];a=_boundary_runs(A,rs,1);b=_boundary_runs(B,rs,0)
            if b.start-a.end-1>max(2,int(math.ceil(30.0*sw))): break
            if not _gap_has_component_evidence(a,b,ink,axis_tol): break
            if emb(nxt):
                c.displacement_end=max(c.displacement_end,int(B.end));absorbed.append(nxt);changed=True
                gb=_group_component_bbox(B,rs,ink,sw,base_margin)
                if gb:bb=_union_bbox(bb,gb[0])
                cur=nxt;continue
            if nxt in eligible_set:
                c.displacement_end=max(c.displacement_end,int(b.start));changed=True
            break

    if changed:
        # Longitudinal extent is the whole carrier interval; normal extent is the
        # union of the original high-precision box and absorbed-island geometry.
        if c.o=='H': bb=(int(c.displacement_start),bb[1],int(c.displacement_end),bb[3])
        else: bb=(bb[0],int(c.displacement_start),bb[2],int(c.displacement_end))
        c.bbox=bb
        n_absorbed=len(set(absorbed))
        c.internal_group_count=max(c.internal_group_count,n_absorbed)
        # One embedded carrier island implies at least two discontinuity gaps
        # inside the single outer-carrier component interval.  Preserve that
        # structural fact even if tolerant axis occupancy visually bridges one.
        c.gap_count=max(c.gap_count,1+n_absorbed,_count_baseline_blank_runs(c,ink,axis_tol))
        c.reason += ';ABSORBED_INTERNAL_CARRIER_ISLANDS'
    return changed,cache

def convert_v46_base_candidates(base_cands, base_accepted):
    out=[]
    for j in base_accepted:
        q=base_cands[j]
        c=Candidate(len(out),q.o,-1,-1,q.before_run,q.after_run,
                    int(q.before_axis),int(q.after_axis),int(q.before_length),int(q.after_length),
                    int(q.displacement_start),int(q.displacement_end),1,0,q.geometry_mode,
                    'V46_BASE_'+q.proposal_source,int(q.symbol_area),int(q.symbol_offaxis_extent),
                    int(q.fragment_count),int(q.graph_hops),float(q.side_balance),q.bbox)
        c.status='ACCEPTED'; c.reason='V46_PRECISION_BASE_PRESERVED'
        out.append(c)
    return out


def build_exact_axis_near_witness_fallback(rs, active, ink, base_margin, median_radius,
                                           witness_factor=8.0,
                                           max_displacement_strokes=30.0,
                                           near_witness_fraction=0.70,
                                           compact_gap_witness_factor=2.5):
    """Add only a narrowly-scoped fallback for a short carrier side.

    The original v48 path is not changed.  This fallback operates on RAW H/V runs
    after v48 has finished and accepts a pair only when all of the following hold:

      1) the two boundary runs are on the EXACT same raster axis;
      2) they are adjacent runs on that exact axis (nothing same-axis is skipped);
      3) one side already satisfies the original pipe witness threshold;
      4) the other side is just below that threshold, but still has at least
         near_witness_fraction * witness straight support;
      5) the interruption is component-sized: >= 1 witness and <= 2.5 witnesses;
      6) exact endpoint off-axis seeds exist at both boundaries; and
      7) v45's original CONNECTED bridge geometry proves that one local foreground
         component physically bridges both pipe endpoints.

    This is deliberately NOT a relaxed axis-tolerance rule and does not lower the
    original v48 pipe-witness threshold globally.  It only recovers a short exact-
    axis continuation that v48 could not promote to an outer witness.
    """
    sw=stroke_width(median_radius)
    witness=pipe_witness_min_px(median_radius,witness_factor)
    near_min=max(3,int(math.ceil(float(near_witness_fraction)*float(witness))))
    max_local=max(witness,int(math.ceil(float(max_displacement_strokes)*sw)))
    gap_min=int(witness)
    gap_max=min(max_local,int(math.ceil(float(compact_gap_witness_factor)*float(witness))))

    bins={'H':defaultdict(list),'V':defaultdict(list)}
    for rid in active:
        r=rs[rid]
        bins[r.o][int(r.axis)].append(rid)
    for o in ('H','V'):
        for ax,L in bins[o].items():
            L.sort(key=lambda i:(rs[i].start,rs[i].end))

    out=[]
    tested=near_pairs=endpoint_ok=bridge_ok=0
    for o in ('H','V'):
        for ax,L in bins[o].items():
            # Exact-axis adjacency is essential: never skip an intervening same-axis run.
            for aid,bid in zip(L,L[1:]):
                a,b=rs[aid],rs[bid]
                gap=int(b.start)-int(a.end)-1
                if gap<gap_min or gap>gap_max:
                    continue
                tested+=1
                la,lb=int(a.length),int(b.length)
                short=min(la,lb); strong=max(la,lb)
                if strong<witness or short>=witness or short<near_min:
                    continue
                near_pairs+=1
                # Exact endpoint attachment, not tolerant endpoint search.
                if not v34.offaxis_endpoint_seeds(a,1,ink):
                    continue
                if not v34.offaxis_endpoint_seeds(b,0,ink):
                    continue
                endpoint_ok+=1
                x=v45.bridge_component(o,a,b,ink,base_margin,sw)
                if x is None:
                    continue
                bb,area,off=x
                bridge_ok+=1
                c=Candidate(len(out),o,-1,-1,int(a.id),int(b.id),
                            int(a.axis),int(b.axis),la,lb,
                            int(a.end),int(b.start),1,0,'CONNECTED',
                            'EXACT_AXIS_NEAR_WITNESS_FALLBACK',int(area),int(off),
                            1,0,1.0,bb)
                c.reason=('EXACT_SAME_AXIS;ONE_ORIGINAL_PIPE_WITNESS;'
                          'ONE_NEAR_WITNESS_SHORT_SIDE;COMPACT_COMPONENT_GAP;'
                          'ONE_LOCAL_COMPONENT_BRIDGES_BOTH_ENDPOINTS')
                out.append(c)
    return out,{
        'exact_axis_fallback_witness_min_px':int(witness),
        'exact_axis_fallback_short_min_px':int(near_min),
        'exact_axis_fallback_gap_min_px':int(gap_min),
        'exact_axis_fallback_gap_max_px':int(gap_max),
        'exact_axis_fallback_pairs_in_compact_gap':int(tested),
        'exact_axis_fallback_one_strong_one_near_witness':int(near_pairs),
        'exact_axis_fallback_endpoint_verified':int(endpoint_ok),
        'exact_axis_fallback_bridge_verified':int(bridge_ok),
        'exact_axis_fallback_raw_candidates':int(len(out)),
    }


def supplement_not_duplicate(c, bases, axis_tol):
    for b in bases:
        if duplicate(c,b,axis_tol): return False
    return True

def draw_raw(g,rs,active): return v44.draw_raw_hv(g,rs,active)
def draw_witnesses(g,groups,eligible,rs):
    im=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)
    for gid in eligible:
        G=groups[gid]
        for rid in G.member_ids:
            r=rs[rid]
            if r.o=='H': cv2.line(im,(r.start,r.axis),(r.end,r.axis),(0,180,0),1)
            else: cv2.line(im,(r.axis,r.start),(r.axis,r.end),(0,180,0),1)
    return im

def draw_proposals(g,ps,rs):
    im=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)
    for p in ps:
        a=rs[p.before_run];b=rs[p.after_run]
        col=(255,0,0) if p.gap_count>1 else ((0,0,255) if p.geometry_mode=='WHOLE_CARRIER' else (0,160,255))
        if p.o=='H': cv2.line(im,(a.end,a.axis),(b.start,b.axis),col,1)
        else: cv2.line(im,(a.axis,a.end),(b.axis,b.start),col,1)
    return im

def draw_final(g,cands,accepted,rs):
    im=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)
    for i in accepted:
        c=cands[i];a=rs[c.before_run];b=rs[c.after_run]
        if c.o=='H':
            cv2.line(im,(a.start,a.axis),(a.end,a.axis),(0,180,0),1);cv2.line(im,(b.start,b.axis),(b.end,b.axis),(0,180,0),1)
        else:
            cv2.line(im,(a.axis,a.start),(a.axis,a.end),(0,180,0),1);cv2.line(im,(b.axis,b.start),(b.axis,b.end),(0,180,0),1)
        x1,y1,x2,y2=c.bbox
        col=(255,0,0) if c.gap_count>1 else ((0,0,255) if c.geometry_mode=='WHOLE_CARRIER' else ((255,0,255) if c.geometry_mode=='CONNECTED' else (0,128,255)))
        cv2.rectangle(im,(x1,y1),(x2,y2),col,1)
    return im


def save_csv(out,cands):
    with open(out/'whole_carrier_components.csv','w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['candidate_id','status','owner','reason','proposal_source','geometry_mode','orientation','before_group','after_group','before_run','after_run','before_axis','after_axis','axis_shift_px','before_support','after_support','gap_count','internal_group_count','displacement_start','displacement_end','displacement_span','symbol_area','symbol_offaxis_extent','fragment_count','graph_hops','side_balance','x1','y1','x2','y2','width','height','bbox_area'])
        for c in cands:
            x1,y1,x2,y2=c.bbox
            w.writerow([c.id,c.status,c.owner,c.reason,c.proposal_source,c.geometry_mode,c.o,c.before_group,c.after_group,c.before_run,c.after_run,c.before_axis,c.after_axis,abs(c.before_axis-c.after_axis),c.before_support,c.after_support,c.gap_count,c.internal_group_count,c.displacement_start,c.displacement_end,c.displacement_span,c.symbol_area,c.symbol_offaxis_extent,c.fragment_count,c.graph_hops,round(c.side_balance,4),x1,y1,x2,y2,x2-x1+1,y2-y1+1,c.bbox_area])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('input');ap.add_argument('-o','--output',default='AP1000_v48_WHOLE_CARRIER_MULTI_GAP_TOLERANT_RESULT')
    ap.add_argument('--min-line-length',type=int,default=3)
    ap.add_argument('--pipe-witness-factor',type=float,default=8.0)
    ap.add_argument('--axis-tolerance-strokes',type=float,default=1.25)
    ap.add_argument('--lane-join-strokes',type=float,default=0.75)
    ap.add_argument('--lane-min-density',type=float,default=0.72)
    ap.add_argument('--max-displacement-strokes',type=float,default=30.0)
    ap.add_argument('--internal-island-max-strokes',type=float,default=24.0)
    ap.add_argument('--fragment-gap-strokes',type=float,default=6.0)
    ap.add_argument('--multipart-min-side-balance',type=float,default=0.30)
    ap.add_argument('--two-fragment-max-normal-ratio',type=float,default=1.50)
    ap.add_argument('--embedded-carrier-fraction',type=float,default=0.55)
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True);t0=time.time()
    g=v34.load(Path(args.input));ink=v34.binarize(g);sk=v34.skeletonize(ink)
    base_margin,med=v37.estimate_gap_margin(ink,sk)
    rs,hb0,vb0,hid0,vid0=v34.extract(sk)
    hb,vb,hid,vid,active,lstats=v44.filter_line_entities(rs,hb0,vb0,hid0,vid0,args.min_line_length);v34.annotate(rs,hid,vid)
    # Preserve v46's precision set exactly, then add tolerant/whole-carrier misses only.
    base_ps,base_stats=v46.precision_proposals(
        rs,active,ink,base_margin,med,args.pipe_witness_factor,args.max_displacement_strokes,
        fragment_gap_strokes=args.fragment_gap_strokes,min_side_balance=args.multipart_min_side_balance,
        two_fragment_max_normal_ratio=args.two_fragment_max_normal_ratio)
    base_raw=v46.build_candidates(base_ps,rs,base_stats['pipe_witness_min_px'])
    base_acc=v46.resolve(base_raw,base_margin)
    base_cands=convert_v46_base_candidates(base_raw,base_acc)

    ps,groups,eligible,pstats=build_whole_carrier_proposals(
        rs,active,ink,base_margin,med,args.axis_tolerance_strokes,args.pipe_witness_factor,
        args.max_displacement_strokes,0.05,args.lane_join_strokes,args.lane_min_density,
        args.internal_island_max_strokes,args.fragment_gap_strokes,args.multipart_min_side_balance,
        args.two_fragment_max_normal_ratio,args.min_line_length,args.embedded_carrier_fraction)
    supp=build_candidates(ps,groups,rs);axis_tol=pstats['effective_axis_tolerance_px']
    # Expand carrier-like islands BEFORE duplicate resolution.  A local carrier
    # fragment embedded inside the symbol belongs to the same component interval.
    base_expanded=0
    for c in base_cands:
        changed,_=expand_internal_carrier_islands(
            c,groups,eligible,rs,ink,base_margin,axis_tol,stroke_width(med),
            args.internal_island_max_strokes,args.embedded_carrier_fraction)
        base_expanded += int(changed)
    supp_expanded=0
    for c in supp:
        changed,_=expand_internal_carrier_islands(
            c,groups,eligible,rs,ink,base_margin,axis_tol,stroke_width(med),
            args.internal_island_max_strokes,args.embedded_carrier_fraction)
        supp_expanded += int(changed)
    supp=[c for c in supp if supplement_not_duplicate(c,base_cands,axis_tol)]
    for i,c in enumerate(supp): c.id=len(base_cands)+i
    supp_acc_local=resolve(supp,axis_tol)
    # resolve() indexes into supp; accepted supplement candidates are those marked ACCEPTED.
    supp_accepted=[i for i,c in enumerate(supp) if c.status=='ACCEPTED']

    # IMPORTANT: original v48 is complete at this point.  The following is an
    # additive RAW-run fallback only; no v48 grouping/embeddedness/witness rule
    # above has been altered.
    existing_accepted=base_cands+[supp[i] for i in supp_accepted]
    exact_fb,exact_stats=build_exact_axis_near_witness_fallback(
        rs,active,ink,base_margin,med,args.pipe_witness_factor,args.max_displacement_strokes)
    exact_raw_count=len(exact_fb)
    exact_fb=[c for c in exact_fb if supplement_not_duplicate(c,existing_accepted,axis_tol)]
    for i,c in enumerate(exact_fb): c.id=len(base_cands)+len(supp)+i
    exact_acc_local=resolve(exact_fb,axis_tol)
    exact_accepted=[i for i,c in enumerate(exact_fb) if c.status=='ACCEPTED']

    cands=base_cands+supp+exact_fb
    accepted=(list(range(len(base_cands)))
              +[len(base_cands)+i for i in supp_accepted]
              +[len(base_cands)+len(supp)+i for i in exact_accepted])
    cv2.imwrite(str(out/'01_raw_HV_runs.png'),draw_raw(g,rs,active),[cv2.IMWRITE_PNG_COMPRESSION,1])
    cv2.imwrite(str(out/'02_tolerant_lane_witnesses.png'),draw_witnesses(g,groups,eligible,rs),[cv2.IMWRITE_PNG_COMPRESSION,1])
    cv2.imwrite(str(out/'03_whole_carrier_displacements.png'),draw_proposals(g,ps,rs),[cv2.IMWRITE_PNG_COMPRESSION,1])
    cv2.imwrite(str(out/'04_v48_whole_carrier_bbox.png'),draw_final(g,cands,accepted,rs),[cv2.IMWRITE_PNG_COMPRESSION,1])
    save_csv(out,cands)
    areas=[cands[i].bbox_area for i in accepted]
    modes=defaultdict(int)
    for i in accepted:modes[cands[i].geometry_mode]+=1
    summary={'version':'v48-exact-axis-short-side-fallback','mode':'ORIGINAL_V48_PLUS_EXACT_AXIS_NEAR_WITNESS_FALLBACK','image_size':[g.shape[1],g.shape[0]],
             'estimated_base_margin_px':int(base_margin),'median_skeleton_stroke_radius_px':round(float(med),3),**lstats,**pstats,
             'v46_base_accepted_preserved':len(base_cands),'base_candidates_expanded_to_whole_carrier':int(base_expanded),'supplement_candidates_expanded_to_whole_carrier':int(supp_expanded),'embedded_carrier_fraction':float(args.embedded_carrier_fraction),'supplement_candidates':len(supp),**exact_stats,'exact_axis_fallback_raw_candidates_before_existing_duplicate_filter':int(exact_raw_count),'exact_axis_fallback_candidates_after_existing_duplicate_filter':len(exact_fb),'exact_axis_fallback_accepted':len(exact_accepted),'candidates_built':len(cands),'accepted_components':len(accepted),'accepted_modes':dict(modes),
             'accepted_multi_gap_components':int(sum(cands[i].gap_count>1 for i in accepted)),
             'duplicates':int(sum(c.status=='DUPLICATE' for c in cands)),
             'accepted_bbox_area_median':float(np.median(areas)) if areas else 0.0,'accepted_bbox_area_max':int(max(areas)) if areas else 0,
             'ocr_or_tag_logic_used':False,'global_connectivity_used':False,
             'displacement_coordinates_from_original_endpoints':True,'runtime_sec':round(time.time()-t0,3)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

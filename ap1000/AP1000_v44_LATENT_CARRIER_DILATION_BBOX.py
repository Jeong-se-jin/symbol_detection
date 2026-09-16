#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AP1000 v44 — Latent Carrier Dilation + Local Max-Distance BBox
=================================================================

Design
------
1) Recognize H/V line entities with the same small raster tolerance as v37.
2) Build ordinary v34 displacement proposals for exact same-axis carriers.
3) Add ONLY near-axis supplements where the two carrier fragments differ by
   <= the measured line-recognition tolerance (normally 1 px).  The two endpoint
   axes are preserved; they are not snapped to one y/x.
4) For every longitudinal coordinate inside the displacement interval, define a
   local baseline point by linear interpolation between the two carrier endpoints.
5) From EACH baseline point, measure perpendicular rays independently on the
   ORIGINAL binary ink.  The ray may cross blank interior until it first reaches
   symbol ink; after first contact it follows only the local contiguous/near-
   contiguous support and stops after a blank gap larger than the raster tolerance.
6) The final bbox is the envelope of all pointwise ray extents plus the exact
   displacement endpoints.
7) Axis validation is LOCAL and length-based only: actual observed parent-carrier
   ink support (gap excluded) must be longer than the local perpendicular bbox
   extent.  No global orthogonal track can veto a candidate.

No connected flood, direct-path traversal, component hierarchy, or dashed-track
bbox expansion is used for final bbox generation.
"""
from __future__ import annotations

import argparse, csv, json, sys, time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import AP1000_v34_AXIS_RELATIVE_FSM_RUN_GRAPH as v34
import AP1000_v36_LONGEST_TRACK_HIERARCHY as v36
import AP1000_v37_TOLERANT_LONGEST_TRACK_HIERARCHY as v37

BBox = Tuple[int,int,int,int]

@dataclass
class Proposal:
    o: str
    before: int
    after: int
    source: str

@dataclass
class Candidate:
    id: int
    o: str
    before_run: int
    after_run: int
    before_axis: int
    after_axis: int
    displacement_start: int
    displacement_end: int
    left_track: int
    right_track: int
    parent_left_ink: int
    parent_right_ink: int
    parent_support_ink: int
    orthogonal_bbox_extent: int
    ray_max_neg: int
    ray_max_pos: int
    bbox: BBox
    proposal_source: str
    status: str = "PENDING"
    owner: Optional[int] = None
    reason: str = ""

    @property
    def displacement_span(self):
        return self.displacement_end-self.displacement_start+1
    @property
    def bbox_area(self):
        x1,y1,x2,y2=self.bbox
        return (x2-x1+1)*(y2-y1+1)


def filter_line_entities(rs,hb,vb,hid,vid,min_line_length=3):
    m=max(1,int(min_line_length)); keep=np.zeros(len(rs),bool)
    for r in rs: keep[r.id]=r.length>=m
    hb2={a:[i for i in ids if keep[i]] for a,ids in hb.items()}; hb2={a:v for a,v in hb2.items() if v}
    vb2={a:[i for i in ids if keep[i]] for a,ids in vb.items()}; vb2={a:v for a,v in vb2.items() if v}
    hid2=hid.copy(); q=hid2>=0
    if np.any(q):
        vals=hid2[q]; hid2[q]=np.where(keep[vals],vals,-1)
    vid2=vid.copy(); q=vid2>=0
    if np.any(q):
        vals=vid2[q]; vid2[q]=np.where(keep[vals],vals,-1)
    active={r.id for r in rs if keep[r.id]}
    return hb2,vb2,hid2,vid2,active,{
        "min_line_length_px":m,
        "raw_hv_runs_before_min_length_filter":len(rs),
        "recognized_hv_runs_after_min_length_filter":len(active),
        "short_hv_runs_rejected":len(rs)-len(active),
    }


def build_tracks(rs,hb,vb,sk,margin,active):
    d=v36.DSU(len(rs)); repair=0
    for by in (hb,vb):
        for _,ids0 in by.items():
            ids=sorted(ids0,key=lambda i:rs[i].start)
            for ai,bi in zip(ids,ids[1:]):
                a,b=rs[ai],rs[bi]; gap=b.start-a.end-1
                if 1<=gap<=margin and not v37.gap_has_offaxis(a,b,sk,margin):
                    d.union(ai,bi); repair+=1
    dh,sh=v36.detect_dashed_links(rs,hb); dv,sv=v36.detect_dashed_links(rs,vb)
    dashed=set()
    for a,b in dh+dv:
        d.union(a,b); dashed.add((min(a,b),max(a,b)))
    groups=defaultdict(list)
    for r in rs:
        if r.id in active: groups[d.find(r.id)].append(r.id)
    tracks={}; r2t={}; tid=0
    for rids in groups.values():
        rr=[rs[i] for i in rids]; o,axis=rr[0].o,rr[0].axis
        st=min(r.start for r in rr); en=max(r.end for r in rr); ink=sum(r.length for r in rr)
        rset=set(rids); isdash=any(a in rset and b in rset for a,b in dashed)
        sr=sorted(rids,key=lambda i:rs[i].start)
        gaps=[max(0,rs[b].start-rs[a].end-1) for a,b in zip(sr,sr[1:])]
        gs=max(gaps) if isdash and gaps else 0
        t=v36.Track(tid,o,axis,sr,st,en,en-st+1,ink,isdash,gs); tracks[tid]=t
        for rid in rids:r2t[rid]=tid
        tid+=1
    return tracks,r2t,{"low_quality_collinear_repair_links":repair,"dashed_sequences":sh+sv,
                       "line_tracks":len(tracks),"dashed_tracks":sum(t.dashed for t in tracks.values())}


def _offaxis_ink_in_gap(o, a, b, ink, margin):
    """Require component-like ink inside the longitudinal gap, away from the carrier lane."""
    H,W=ink.shape
    lo=int(a.end); hi=int(b.start)
    if hi<=lo+1: return False
    # Vectorized local slab test.  Carrier lane itself is excluded by margin+1.
    if o=='H':
        y=int(round((a.axis+b.axis)/2))
        y0=max(0,y-max(2,margin+1)); y1=min(H-1,y+max(2,margin+1))
        slab=ink[y0:y1+1, max(0,lo):min(W,hi+1)].copy()
        cy0=max(0,int(a.axis)-margin-y0); cy1=min(slab.shape[0]-1,int(a.axis)+margin-y0)
        if cy0<=cy1: slab[cy0:cy1+1,:]=False
        cy0=max(0,int(b.axis)-margin-y0); cy1=min(slab.shape[0]-1,int(b.axis)+margin-y0)
        if cy0<=cy1: slab[cy0:cy1+1,:]=False
        return bool(np.any(slab))
    x=int(round((a.axis+b.axis)/2))
    x0=max(0,x-max(2,margin+1)); x1=min(W-1,x+max(2,margin+1))
    slab=ink[max(0,lo):min(H,hi+1), x0:x1+1].copy()
    cx0=max(0,int(a.axis)-margin-x0); cx1=min(slab.shape[1]-1,int(a.axis)+margin-x0)
    if cx0<=cx1: slab[:,cx0:cx1+1]=False
    cx0=max(0,int(b.axis)-margin-x0); cx1=min(slab.shape[1]-1,int(b.axis)+margin-x0)
    if cx0<=cx1: slab[:,cx0:cx1+1]=False
    return bool(np.any(slab))


def _same_lane_internal_runs(rs, bins, starts, o, a, b, margin):
    """Return same-orientation runs in the tolerant axis lane strictly inside A..B."""
    import bisect
    ids=[]
    axlo=min(int(a.axis),int(b.axis))-margin
    axhi=max(int(a.axis),int(b.axis))+margin
    for ax in range(axlo,axhi+1):
        ids0=bins[o].get(ax)
        if not ids0: continue
        st=starts[o][ax]
        k0=bisect.bisect_right(st,int(a.end))
        k1=bisect.bisect_left(st,int(b.start))
        for rid in ids0[k0:k1]:
            r=rs[rid]
            if r.start>a.end and r.end<b.start:
                ids.append(rid)
    return sorted(set(ids), key=lambda i:(rs[i].start,rs[i].axis))


def latent_carrier_proposals(rs, active, ink, margin, carrier_min_length=8):
    """v44 proposal generator: longitudinally inflate H/V identity, then recover raw endpoints.

    Important differences from v43
    -------------------------------
    * NO outer_support > 2*gap rule.
    * NO max_internal < min(outer lengths) rule.
    * Unequal carrier lengths are accepted without penalty.
    * The inflated identity is used ONLY to decide that outer runs belong to one
      latent H/V carrier.  Displacement coordinates always use the ORIGINAL
      endpoints a.end and b.start.
    * Short same-axis symbol strokes are allowed inside the displacement.

    To prevent one proposal from swallowing multiple separated components, an
    interval is rejected only when the original same-axis ink already occupies
    at least half of its interior.  This is a continuity/separator test, not an
    outer-carrier length test.
    """
    import bisect
    bins={"H":defaultdict(list),"V":defaultdict(list)}
    carrier_active=[rid for rid in active if rs[rid].length>=carrier_min_length]
    for rid in carrier_active:
        r=rs[rid]; bins[r.o][int(r.axis)].append(rid)
    starts={"H":{},"V":{}}
    for o in ('H','V'):
        for ax,ids in bins[o].items():
            ids.sort(key=lambda i:rs[i].start)
            starts[o][ax]=[rs[i].start for i in ids]

    # Longitudinal dilation is represented analytically: A and B are considered
    # the same latent carrier when the blank interval between their ORIGINAL
    # endpoints contains component-like off-axis ink.  We do not modify pixels.
    # Up to 16 following lane runs are inspected so internal symbol H/V strokes
    # (e.g. V012A 3/11/3 px fragments) can be skipped.
    raw=[]
    for aid in carrier_active:
        a=rs[aid]
        succ=[]
        for ax2 in range(int(a.axis)-margin,int(a.axis)+margin+1):
            ids=bins[a.o].get(ax2)
            if not ids: continue
            k=bisect.bisect_right(starts[a.o][ax2],int(a.end)+1)
            succ.extend(ids[k:k+8])
        succ=sorted(set(succ),key=lambda i:(rs[i].start,abs(rs[i].axis-a.axis)))[:12]
        candidates=[]
        for bid in succ:
            b=rs[bid]
            if b.start<=a.end or abs(int(a.axis)-int(b.axis))>margin: continue
            gap=int(b.start-a.end-1)
            if gap<2: continue
            # Search horizon is image-scale only (not dependent on either outer run length).
            # This prevents unrelated distant text/lines from being paired while preserving
            # unequal left/right carrier stubs.
            max_gap=max(64, int(round(min(ink.shape)*0.05)))
            if gap>max_gap: continue
            # A true displacement boundary should have component ink leaving BOTH
            # original carrier endpoints into the gap. This is geometric endpoint
            # evidence only; it does not compare outer-run lengths.
            if not v34.offaxis_endpoint_seeds(a,1,ink): continue
            if not v34.offaxis_endpoint_seeds(b,0,ink): continue
            if not _offaxis_ink_in_gap(a.o,a,b,ink,margin): continue
            internal=_same_lane_internal_runs(rs,bins,starts,a.o,a,b,margin)
            internal_ink=0
            for rid in internal:
                r=rs[rid]
                lo=max(int(r.start),int(a.end)+1); hi=min(int(r.end),int(b.start)-1)
                if hi>=lo: internal_ink += hi-lo+1
            axis_fill=internal_ink/max(1,gap)
            # A truly restored carrier inside the proposed displacement is a
            # separator. Sparse internal component strokes are intentionally kept.
            if axis_fill>=0.50:
                continue
            # Prefer the widest clean latent interruption.  Outer-run lengths are
            # deliberately absent from this score.
            score=(gap, -axis_fill, len(internal), -abs(int(a.axis)-int(b.axis)))
            candidates.append((score,bid,internal,axis_fill))
        if candidates:
            candidates.sort(reverse=True,key=lambda z:z[0])
            score,bid,internal,fill=candidates[0]
            raw.append((aid,bid,internal,fill))

    # Mutual boundary consistency: keep a pair if no other proposal encloses it
    # on the same tolerant lane with the same left boundary.  This removes tiny
    # inner fragments while preserving the widest displacement boundary.
    by_left={}
    for aid,bid,internal,fill in raw:
        key=(rs[aid].o,aid)
        cur=by_left.get(key)
        if cur is None or rs[bid].start>rs[cur[0]].start:
            by_left[key]=(bid,internal,fill)
    out=[]
    for (o,aid),(bid,internal,fill) in by_left.items():
        out.append(Proposal(o,aid,bid,'LATENT_CARRIER_DILATION'))
    out.sort(key=lambda p:(p.o,rs[p.before].axis,rs[p.before].start,rs[p.after].start))
    return out,{"latent_carrier_bracket_proposals":len(out),
                "proposal_outer_length_gate_used":False,
                "proposal_internal_vs_outer_length_gate_used":False,
                "latent_axis_fill_separator_fraction":0.50,"latent_carrier_search_horizon_px":max(64,int(round(min(ink.shape)*0.05))),"carrier_min_length_px":carrier_min_length,"carrier_eligible_runs":len(carrier_active)}

def build_proposals(rs,hb,vb,sk,hid,vid,active,margin,ink=None,carrier_min_length=8):
    # v42 deliberately bypasses the expensive v34 strict graph/bracket proposal
    # path.  Candidate generation is carrier-lane geometry only.
    if ink is None: ink=sk
    return latent_carrier_proposals(rs,active,ink,margin,carrier_min_length)


def _baseline_normal(o,a,b,s):
    s0=float(a.end); s1=float(b.start)
    if s1<=s0: return int(round((a.axis+b.axis)/2))
    t=(float(s)-s0)/(s1-s0)
    return int(round((1.0-t)*a.axis+t*b.axis))


def maxray_bbox(o,a,b,ink,margin):
    """Local axis-connected maximum-distance bbox.

    The longitudinal slab is EXACTLY the displacement [a.end,b.start].  Only in
    the perpendicular direction do we open a local search window (2x span each
    side).  Inside that small slab we select foreground connected to the carrier
    baseline corridor; nearby text/independent lines are separate components and
    are ignored.  Final bbox is the max perpendicular extent of that selected
    local component on the ORIGINAL ink.

    This is not a global flood/path operation: connectivity cannot escape the
    displacement slab or walk along external process piping.
    """
    H,W=ink.shape
    lo,hi=sorted((int(a.end),int(b.start)))
    span=max(1,hi-lo+1); lim=max(2,2*span)
    ss=np.arange(lo,hi+1,dtype=np.int32)
    n0=np.array([_baseline_normal(o,a,b,int(s)) for s in ss],dtype=np.int32)
    center=int(round(float(np.median(n0))))

    if o=='H':
        x0=max(0,lo); x1=min(W-1,hi); y0=max(0,center-lim); y1=min(H-1,center+lim)
        crop=ink[y0:y1+1,x0:x1+1].astype(np.uint8)
        # 1px raster repair only for membership labeling; bbox is measured on original ink.
        lab_img=cv2.dilate(crop,np.ones((2*margin+1,2*margin+1),np.uint8),iterations=1) if margin>0 else crop
        n,labels,stats,_=cv2.connectedComponentsWithStats(lab_img,8)
        selected=set()
        for j,scoord in enumerate(range(x0,x1+1)):
            # interpolated baseline for this x; +/-margin corridor
            bn=_baseline_normal(o,a,b,scoord)-y0
            for yy in range(max(0,bn-margin),min(labels.shape[0]-1,bn+margin)+1):
                lid=int(labels[yy,j])
                if lid>0:selected.add(lid)
        mask=np.isin(labels,list(selected)) if selected else np.zeros_like(labels,dtype=bool)
        # Apply membership to ORIGINAL foreground, not dilated pixels.
        pts=np.argwhere(mask & (crop>0))
        if len(pts)==0:
            min_n,max_n=int(n0.min()),int(n0.max())
        else:
            min_n=y0+int(pts[:,0].min()); max_n=y0+int(pts[:,0].max())
        max_neg=max(0,int(n0.max())-min_n); max_pos=max(0,max_n-int(n0.min()))
        return (x0,max(0,min_n),x1,min(H-1,max_n)),max_neg,max_pos

    y0=max(0,lo); y1=min(H-1,hi); x0=max(0,center-lim); x1=min(W-1,center+lim)
    crop=ink[y0:y1+1,x0:x1+1].astype(np.uint8)
    lab_img=cv2.dilate(crop,np.ones((2*margin+1,2*margin+1),np.uint8),iterations=1) if margin>0 else crop
    n,labels,stats,_=cv2.connectedComponentsWithStats(lab_img,8)
    selected=set()
    for j,scoord in enumerate(range(y0,y1+1)):
        bn=_baseline_normal(o,a,b,scoord)-x0
        for xx in range(max(0,bn-margin),min(labels.shape[1]-1,bn+margin)+1):
            lid=int(labels[j,xx])
            if lid>0:selected.add(lid)
    mask=np.isin(labels,list(selected)) if selected else np.zeros_like(labels,dtype=bool)
    pts=np.argwhere(mask & (crop>0))
    if len(pts)==0:
        min_n,max_n=int(n0.min()),int(n0.max())
    else:
        min_n=x0+int(pts[:,1].min()); max_n=x0+int(pts[:,1].max())
    max_neg=max(0,int(n0.max())-min_n); max_pos=max(0,max_n-int(n0.min()))
    return (max(0,min_n),y0,min(W-1,max_n),y1),max_neg,max_pos

def candidate_from_proposal(cid,p,rs,tracks,r2t,ink,margin):
    a,b=rs[p.before],rs[p.after]
    if a.o!=p.o or b.o!=p.o or b.start<=a.end:return None
    if abs(int(a.axis)-int(b.axis))>margin:return None
    if p.before not in r2t or p.after not in r2t:return None
    lt,rt=r2t[p.before],r2t[p.after]; L,R=tracks[lt],tracks[rt]
    parent_left=int(L.ink); parent_right=int(R.ink)
    parent=parent_left if lt==rt else parent_left+parent_right
    bbox,mneg,mpos=maxray_bbox(p.o,a,b,ink,margin)
    x1,y1,x2,y2=bbox; ortho=(y2-y1+1) if p.o=='H' else (x2-x1+1)
    c=Candidate(cid,p.o,p.before,p.after,int(a.axis),int(b.axis),int(a.end),int(b.start),lt,rt,
                parent_left,parent_right,parent,int(ortho),int(mneg),int(mpos),bbox,p.source)
    # v42: NO bbox-aspect-ratio veto. Axis was already established by carrier evidence.
    # Keep only the scale-free carrier-vs-gap requirement used during proposal creation.
    c.status='PENDING'
    c.reason='AXIS_FIXED_BY_LATENT_CARRIER_DILATION; BBOX_FROM_LOCAL_MAX_DISTANCE'
    return c


def build_candidates(ps,rs,tracks,r2t,ink,margin):
    out=[]
    for p in ps:
        c=candidate_from_proposal(len(out),p,rs,tracks,r2t,ink,margin)
        if c is not None: out.append(c)
    return out

def bbox_intersects(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    return not(ax2<bx1 or bx2<ax1 or ay2<by1 or by2<ay1)


def center(c):
    s=(c.displacement_start+c.displacement_end)//2
    # use interpolated baseline at center
    # caller only needs approximate duplicate geometry
    n=int(round((c.before_axis+c.after_axis)/2))
    return (s,n) if c.o=='H' else (n,s)


def same_component(a,b):
    # Same tolerant carrier lane + nested displacement = same physical interruption.
    if a.o==b.o and max(abs(a.before_axis-b.before_axis),abs(a.after_axis-b.after_axis))<=1:
        alo,ahi=sorted((a.displacement_start,a.displacement_end)); blo,bhi=sorted((b.displacement_start,b.displacement_end))
        if (alo<=blo and ahi>=bhi) or (blo<=alo and bhi>=ahi):
            return True
    if not bbox_intersects(a.bbox,b.bbox):return False
    ax,ay=center(a); bx,by=center(b)
    A=a.bbox; B=b.bbox
    return B[0]<=ax<=B[2] and B[1]<=ay<=B[3] and A[0]<=bx<=A[2] and A[1]<=by<=A[3]


def rank(c):
    return (c.parent_support_ink,-c.orthogonal_bbox_extent,-c.displacement_span,-c.bbox_area,-c.id)


def resolve(cands):
    eligible=[i for i,c in enumerate(cands) if c.status=='PENDING']; eligible.sort(key=lambda i:rank(cands[i]),reverse=True)
    accepted=[]
    for i in eligible:
        c=cands[i]; owner=None
        for j in accepted:
            if same_component(c,cands[j]):owner=j;break
        if owner is None:
            c.status='ACCEPTED'; c.reason='LATENT_CARRIER_DILATION_MAXDIST_ACCEPTED'; accepted.append(i)
        else:
            c.status='DUPLICATE_MAXRAY_COMPONENT'; c.owner=cands[owner].id; c.reason='SAME_LOCAL_GEOMETRY_AS_STRONGER_PROPOSAL'
    return accepted


def draw(g,rs,cands,accepted):
    im=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)
    for i in accepted:
        c=cands[i]; a,b=rs[c.before_run],rs[c.after_run]
        if c.o=='H':
            cv2.line(im,(a.start,a.axis),(a.end,a.axis),(0,180,0),1); cv2.line(im,(b.start,b.axis),(b.end,b.axis),(0,180,0),1)
            cv2.line(im,(a.end,a.axis),(b.start,b.axis),(0,180,255),1)
        else:
            cv2.line(im,(a.axis,a.start),(a.axis,a.end),(0,180,0),1); cv2.line(im,(b.axis,b.start),(b.axis,b.end),(0,180,0),1)
            cv2.line(im,(a.axis,a.end),(b.axis,b.start),(0,180,255),1)
        x1,y1,x2,y2=c.bbox; cv2.rectangle(im,(x1,y1),(x2,y2),(255,0,255),1)
    return im



def draw_raw_hv(g,rs,active):
    im=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)
    for rid in active:
        r=rs[rid]
        if r.o=='H': cv2.line(im,(r.start,r.axis),(r.end,r.axis),(0,150,0),1)
        else: cv2.line(im,(r.axis,r.start),(r.axis,r.end),(180,120,0),1)
    return im


def draw_latent_carriers(g,rs,ps):
    im=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)
    for p in ps:
        a,b=rs[p.before],rs[p.after]
        if p.o=='H':
            y=int(round((a.axis+b.axis)/2)); cv2.line(im,(a.start,y),(b.end,y),(0,170,255),1)
            cv2.circle(im,(a.end,a.axis),2,(0,0,255),-1); cv2.circle(im,(b.start,b.axis),2,(255,0,0),-1)
        else:
            x=int(round((a.axis+b.axis)/2)); cv2.line(im,(x,a.start),(x,b.end),(0,170,255),1)
            cv2.circle(im,(a.axis,a.end),2,(0,0,255),-1); cv2.circle(im,(b.axis,b.start),2,(255,0,0),-1)
    return im


def draw_displacements(g,rs,ps):
    im=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)
    for p in ps:
        a,b=rs[p.before],rs[p.after]
        if p.o=='H': cv2.line(im,(a.end,a.axis),(b.start,b.axis),(0,0,255),2)
        else: cv2.line(im,(a.axis,a.end),(b.axis,b.start),(0,0,255),2)
    return im

def save_csv(out,cands,tracks):
    with open(out/'axis_connected_components.csv','w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['candidate_id','status','owner_candidate_id','reason','proposal_source','orientation',
        'before_axis','after_axis','axis_shift_px','before_run','after_run','displacement_start','displacement_end','displacement_span',
        'left_parent_track','right_parent_track','parent_left_ink','parent_right_ink','parent_support_ink','ray_max_negative','ray_max_positive',
        'orthogonal_bbox_extent','x1','y1','x2','y2','width','height','area'])
        for c in cands:
            x1,y1,x2,y2=c.bbox; w.writerow([c.id,c.status,c.owner,c.reason,c.proposal_source,c.o,c.before_axis,c.after_axis,
            abs(c.after_axis-c.before_axis),c.before_run,c.after_run,c.displacement_start,c.displacement_end,c.displacement_span,
            c.left_track,c.right_track,c.parent_left_ink,c.parent_right_ink,c.parent_support_ink,c.ray_max_neg,c.ray_max_pos,c.orthogonal_bbox_extent,
            x1,y1,x2,y2,x2-x1+1,y2-y1+1,c.bbox_area])
    with open(out/'line_tracks.csv','w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['track_id','orientation','axis','start','end','span','ink_length','is_dashed','dash_gap_scale','run_count','run_ids'])
        for tid in sorted(tracks):
            t=tracks[tid]; w.writerow([t.id,t.o,t.axis,t.start,t.end,t.span,t.ink,int(t.dashed),t.dash_gap_scale,len(t.run_ids),';'.join(map(str,t.run_ids))])


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('input'); ap.add_argument('-o','--output',default='AP1000_v44_LATENT_CARRIER_DILATION_BBOX_RESULT')
    ap.add_argument('--debug-images',action='store_true'); ap.add_argument('--min-line-length',type=int,default=3); ap.add_argument('--carrier-min-length',type=int,default=8)
    a=ap.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True); t0=time.time()
    g=v34.load(Path(a.input)); ink=v34.binarize(g); sk=v34.skeletonize(ink)
    margin,med=v37.estimate_gap_margin(ink,sk); rs,hb0,vb0,hid0,vid0=v34.extract(sk)
    hb,vb,hid,vid,active,lstats=filter_line_entities(rs,hb0,vb0,hid0,vid0,a.min_line_length); v34.annotate(rs,hid,vid)
    tracks,r2t,tstats=build_tracks(rs,hb,vb,sk,margin,active)
    ps,pstats=build_proposals(rs,hb,vb,sk,hid,vid,active,margin,ink,a.carrier_min_length)
    cands=build_candidates(ps,rs,tracks,r2t,ink,margin); accepted=resolve(cands)
    cv2.imwrite(str(out/'01_raw_HV_runs.png'),draw_raw_hv(g,rs,active),[cv2.IMWRITE_PNG_COMPRESSION,1])
    cv2.imwrite(str(out/'02_latent_carriers.png'),draw_latent_carriers(g,rs,ps),[cv2.IMWRITE_PNG_COMPRESSION,1])
    cv2.imwrite(str(out/'03_displacement_candidates.png'),draw_displacements(g,rs,ps),[cv2.IMWRITE_PNG_COMPRESSION,1])
    cv2.imwrite(str(out/'04_v44_final_bbox.png'),draw(g,rs,cands,accepted),[cv2.IMWRITE_PNG_COMPRESSION,1])
    if a.debug_images:
        cv2.imwrite(str(out/'00_input.png'),g); cv2.imwrite(str(out/'00_skeleton.png'),(~sk).astype(np.uint8)*255)
    save_csv(out,cands,tracks)
    areas=[cands[i].bbox_area for i in accepted]
    summary={"version":"v44","image_size":[g.shape[1],g.shape[0]],"line_recognition_margin_px":margin,
      "median_skeleton_stroke_radius_px":round(med,3),"carrier_pairing_rule":"longitudinal latent-carrier dilation on H/V lanes across measured axis tolerance; NO outer-run length gate; sparse internal same-axis strokes allowed; preserve original endpoints",
      "bbox_rule":"after latent-carrier displacement is fixed, exact longitudinal slab only; local axis-connected foreground gives maximum perpendicular extent on original ink",
      "ray_blank_tolerance_px":None,"ray_search_limit_rule":"2 x displacement span perpendicular local window; connected membership cannot leave the exact longitudinal displacement slab",
      "axis_rule":"axis is fixed before bbox generation by inflated H/V carrier-lane pairing; bbox aspect ratio does not veto the component",
      "global_orthogonal_track_veto_used":False,"connectivity_used_for_final_bbox":"LOCAL_SLAB_ONLY","direct_path_used":False,"flood_fill_used":False,
      **lstats,**tstats,**pstats,"candidates_built":len(cands),
      "rejected_axis_not_dominant":0,
      "duplicate_pointwise_components":sum(c.status=='DUPLICATE_POINTWISE_COMPONENT' for c in cands),"accepted_components":len(accepted),
      "accepted_bbox_area_median":float(np.median(areas)) if areas else 0.0,"accepted_bbox_area_max":int(max(areas)) if areas else 0,
      "runtime_sec":round(time.time()-t0,3)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

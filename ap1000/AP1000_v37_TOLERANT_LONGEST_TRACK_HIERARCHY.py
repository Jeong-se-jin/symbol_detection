#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AP1000 v37 — Low-Quality-Tolerant Longest-Track Hierarchy
==========================================================

Changes from v36
----------------
* Keep the orientation-relative longest-line-track hierarchy unchanged.
* Estimate a minimal raster tolerance from the observed stroke radius.
* Line recognition: repair only tiny COLLINEAR H/V dropouts. A dropout is not
  repaired when off-axis foreground is present nearby, so a real component
  displacement is not erased.
* Shape recovery: replace strict 8-neighbour motion by tolerant 8-neighbour
  motion. If the data-driven gap margin is m, foreground pixels within
  Chebyshev distance 1+m are connectable, allowing m missing raster pixels.
* Final bboxes are NOT padded: they are the min/max of recovered ORIGINAL
  foreground pixels (plus explicitly attached dashed-track pixels).

No manual pixel margin is required; on the current AP1000 raster the estimated
missing-pixel margin is 1 px.
"""
from __future__ import annotations
import argparse, csv, json, time, sys
from pathlib import Path
from collections import defaultdict, deque
from typing import Set

import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
import AP1000_v34_AXIS_RELATIVE_FSM_RUN_GRAPH as v34
import AP1000_v36_LONGEST_TRACK_HIERARCHY as v36


def estimate_gap_margin(ink:np.ndarray, sk:np.ndarray):
    """Estimate minimal missing-pixel tolerance from modal skeleton stroke radius."""
    dt=cv2.distanceTransform(ink.astype(np.uint8),cv2.DIST_L2,5)
    vals=np.rint(dt[sk]).astype(np.int32)
    vals=vals[vals>0]
    if vals.size==0:
        return 1,1.0
    bc=np.bincount(vals)
    mode=int(np.argmax(bc))
    # A modal radius r supports r missing raster pixels as the minimal tolerance.
    return max(1,mode), float(np.median(dt[sk]))


def _shift_bool(a,dy,dx):
    H,W=a.shape; out=np.zeros_like(a,bool)
    ys0=max(0,-dy); ys1=min(H,H-dy); yd0=ys0+dy; yd1=ys1+dy
    xs0=max(0,-dx); xs1=min(W,W-dx); xd0=xs0+dx; xd1=xs1+dx
    if ys1>ys0 and xs1>xs0:
        out[yd0:yd1,xd0:xd1]=a[ys0:ys1,xs0:xs1]
    return out


def repair_collinear_dropouts(sk:np.ndarray, margin:int):
    """Repair tiny H/V raster dropouts without filling true off-axis displacement.

    Directional closing proposes virtual pixels.  Proposed H pixels are accepted
    only when no original foreground exists in neighbouring rows within the
    local tolerance; V is symmetric.  Thus an isolated scan dropout is repaired,
    while a valve/orifice stroke at the gap protects the displacement event.
    """
    u=sk.astype(np.uint8)
    k=2*margin+1
    hc=cv2.morphologyEx(u,cv2.MORPH_CLOSE,np.ones((1,k),np.uint8)).astype(bool)
    vc=cv2.morphologyEx(u,cv2.MORPH_CLOSE,np.ones((k,1),np.uint8)).astype(bool)
    cand_h=hc & ~sk; cand_v=vc & ~sk
    off_h=np.zeros_like(sk,bool); off_v=np.zeros_like(sk,bool)
    # Include one extra normal pixel: a true displacement normally touches the
    # carrier at or immediately beside the dropout.
    for d in range(1,margin+2):
        off_h |= _shift_bool(sk,d,0) | _shift_bool(sk,-d,0)
        off_v |= _shift_bool(sk,0,d) | _shift_bool(sk,0,-d)
    fill_h=cand_h & ~off_h
    fill_v=cand_v & ~off_v
    repaired=sk | fill_h | fill_v
    return repaired,fill_h,fill_v


def offsets_for_margin(margin:int):
    r=1+margin
    # nearest first makes the traversal behave like ordinary 8-neighbour unless
    # a tiny raster dropout actually needs to be crossed.
    off=[]
    for dy in range(-r,r+1):
        for dx in range(-r,r+1):
            if dx==0 and dy==0: continue
            d=max(abs(dx),abs(dy))
            if d<=r: off.append((d,dy,dx))
    off.sort()
    return [(dy,dx) for _,dy,dx in off]


def tolerant_endpoint_seeds(r,side,sk,margin):
    x0,y0=v34.endpoint(r,side); H,W=sk.shape; R=1+margin; out=[]
    for dy in range(-R,R+1):
        for dx in range(-R,R+1):
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
    # deterministic nearest-first unique order
    out=sorted(set(out),key=lambda p:(max(abs(p[0]-x0),abs(p[1]-y0)),abs(p[0]-x0)+abs(p[1]-y0),p[1],p[0]))
    return tuple(out)


_CONN={}
def axis_relative_connected_tol(a,b,sk,margin,return_seen=False):
    if a.o!=b.o or a.axis!=b.axis or b.start<=a.end:
        return (False,set()) if return_seen else False
    key=(a.id,b.id,margin)
    if not return_seen and key in _CONN: return _CONN[key]
    gap=b.start-a.end-1
    # Retain v34's scale-free locality guard, enlarged only by the learned gap tolerance.
    if gap>a.length+b.length+margin:
        if not return_seen: _CONN[key]=False
        return (False,set()) if return_seen else False
    S=tolerant_endpoint_seeds(a,1,sk,margin); T=set(tolerant_endpoint_seeds(b,0,sk,margin))
    if not S or not T:
        if not return_seen: _CONN[key]=False
        return (False,set()) if return_seen else False
    H,W=sk.shape; lo=a.end; hi=b.start; axis=a.axis; offsets=offsets_for_margin(margin)
    dq=deque(S); seen=set(S)
    while dq:
        x,y=dq.popleft()
        if (x,y) in T:
            if not return_seen: _CONN[key]=True
            return (True,seen) if return_seen else True
        for dy,dx in offsets:
            xx=x+dx; yy=y+dy
            if not (0<=xx<W and 0<=yy<H) or not sk[yy,xx]: continue
            if a.o=='H':
                if xx<lo or xx>hi or yy==axis: continue
            else:
                if yy<lo or yy>hi or xx==axis: continue
            z=(xx,yy)
            if z not in seen:
                seen.add(z); dq.append(z)
    if not return_seen: _CONN[key]=False
    return (False,seen) if return_seen else False


def base_adjacent_events_tol(rs,by,o,shape_sk,hid,vid,margin):
    per_axis={}; cc=dc=0
    for axis,ids0 in by.items():
        ids=sorted(ids0,key=lambda i:rs[i].start); flags=[]; src=[]
        for ai,bi in zip(ids,ids[1:]):
            a,b=rs[ai],rs[bi]
            if b.start<=a.end+1:
                flags.append(False); src.append(''); continue
            if axis_relative_connected_tol(a,b,shape_sk,margin):
                flags.append(True); src.append('TOL_AXIS_REL_CONNECTED'); cc+=1
            elif v34.disconnected_boundary_event(a,b,rs,hid,vid):
                flags.append(True); src.append('DISCONNECTED_TWO_BOUNDARY'); dc+=1
            else:
                flags.append(False); src.append('')
        per_axis[axis]=(ids,flags,src)
    return per_axis,cc,dc


def merge_internal_tol(per_axis,rs,shape_sk,o,margin):
    out=[]; tests=merges=0
    for axis,(ids,flags,src) in per_axis.items():
        n=len(ids); i=0
        while i<n-1:
            if not flags[i]: i+=1; continue
            start=i; end=i+1; sources=[src[i]]; k=i+1
            while k<n-1 and flags[k]:
                tests+=1
                if axis_relative_connected_tol(rs[ids[start]],rs[ids[k+1]],shape_sk,margin):
                    end=k+1; sources.append(src[k]); merges+=1; k+=1
                else: break
            source='TOL_FSM_MERGED_AXIS_REL' if end>start+1 else sources[0]
            out.append(v34.Event(o,axis,ids[start],ids[end],source,end-start))
            i=end if end>start+1 else start+1
    return out,tests,merges


def raw_event_pixels_tol(e,rs,sk,margin):
    """Recover ORIGINAL foreground with tolerant 8-neighbour motion.

    The event slab and forbidden parent axis are unchanged from v36.  Only the
    neighbourhood radius is enlarged enough to cross the learned missing-pixel
    margin.  Virtual bridge pixels are never returned in the component set.
    """
    H,W=sk.shape; a,b=rs[e.before],rs[e.after]
    lo=a.end; hi=b.start; axis=e.axis
    seeds=list(tolerant_endpoint_seeds(a,1,sk,margin))+list(tolerant_endpoint_seeds(b,0,sk,margin))
    seeds=list(dict.fromkeys(seeds))
    if not seeds: return set()
    offsets=offsets_for_margin(margin); seen=set(); out=set(); dq=deque()
    for z in seeds:
        x,y=z
        if e.o=='H':
            if x<lo or x>hi or y==axis: continue
        else:
            if y<lo or y>hi or x==axis: continue
        f=y*W+x
        if f not in seen: seen.add(f); dq.append(z)
    while dq:
        x,y=dq.popleft()
        if not sk[y,x]: continue
        f=y*W+x; out.add(f)
        for dy,dx in offsets:
            xx=x+dx; yy=y+dy
            if not (0<=xx<W and 0<=yy<H) or not sk[yy,xx]: continue
            if e.o=='H':
                if xx<lo or xx>hi or yy==axis: continue
            else:
                if yy<lo or yy>hi or xx==axis: continue
            ff=yy*W+xx
            if ff not in seen:
                seen.add(ff); dq.append((xx,yy))
    return out


def build_candidates_tol(events,middle,rs,shape_sk,hid,vid,tracks,run_to_track,margin):
    H,W=shape_sk.shape; em={v36.event_key(e):e for e in events}; out=[]
    for c in middle:
        e=em.get((c.o,c.axis,c.before,c.after))
        if e is None: continue
        pix=raw_event_pixels_tol(e,rs,shape_sk,margin)
        own=run_to_track[e.before]
        same=v36.same_orientation_tracks_in_pixels(e,pix,own,run_to_track,hid,vid,W)
        dom=max(same,key=lambda tid:(tracks[tid].span,tracks[tid].ink,-tid))
        dashed=v36.touching_dashed_tracks(pix,tracks,run_to_track,hid,vid,W)
        for tid in dashed:
            if tid!=dom: pix.update(v36.track_pixels(tracks[tid],rs,W))
        bb=v36.pix_bbox(pix,W,c.bbox)
        cand=v36.Candidate(len(out),e,c,own,dom,tracks[own].span,tracks[dom].span,len(same),pix,bb,dashed)
        if own!=dom:
            cand.status='ORIENTATION_CHILD'; cand.reason='LONGER_SAME_ORIENTATION_TRACK_IN_DISPLACEMENT'
        out.append(cand)
    return out



def gap_has_offaxis(a,b,sk,margin):
    """True if a tiny same-axis gap contains nearby off-axis geometry."""
    if a.o!=b.o or a.axis!=b.axis: return True
    H,W=sk.shape
    if a.o=='H':
        x0=a.end+1; x1=b.start-1
        if x1<x0: return False
        y=a.axis
        yy0=max(0,y-margin-1); yy1=min(H,y+margin+2)
        block=sk[yy0:yy1,max(0,x0):min(W,x1+1)].copy()
        # parent axis itself does not count as off-axis
        if yy0<=y<yy1: block[y-yy0,:]=False
        return bool(block.any())
    else:
        y0=a.end+1; y1=b.start-1
        if y1<y0: return False
        x=a.axis
        xx0=max(0,x-margin-1); xx1=min(W,x+margin+2)
        block=sk[max(0,y0):min(H,y1+1),xx0:xx1].copy()
        if xx0<=x<xx1: block[:,x-xx0]=False
        return bool(block.any())


def build_tracks_tol(rs,hb,vb,events,sk,margin):
    """v36 tracks + minimal collinear dropout links for low-quality raster lines."""
    d=v36.DSU(len(rs)); event_links=0; repair_links=0
    for e in events:
        d.union(e.before,e.after); event_links+=1
    # Only bridge tiny same-axis gaps that contain no off-axis geometry.
    for by in (hb,vb):
        for axis,ids0 in by.items():
            ids=sorted(ids0,key=lambda i:rs[i].start)
            for ai,bi in zip(ids,ids[1:]):
                a,b=rs[ai],rs[bi]; gap=b.start-a.end-1
                if 1<=gap<=margin and not gap_has_offaxis(a,b,sk,margin):
                    d.union(ai,bi); repair_links+=1
    dh,sh=v36.detect_dashed_links(rs,hb); dv,sv=v36.detect_dashed_links(rs,vb)
    dashed_edges=set()
    for a,b in dh+dv:
        d.union(a,b); dashed_edges.add((min(a,b),max(a,b)))
    groups=defaultdict(list)
    for r in rs: groups[d.find(r.id)].append(r.id)
    tracks={}; run_to_track={}; tid=0
    for _,rids in groups.items():
        rr=[rs[i] for i in rids]; o=rr[0].o; axis=rr[0].axis
        st=min(r.start for r in rr); en=max(r.end for r in rr); ink=sum(r.length for r in rr)
        rset=set(rids); dashed=any(a in rset and b in rset for a,b in dashed_edges)
        sr=sorted(rids,key=lambda i:rs[i].start)
        gaps=[max(0,rs[b].start-rs[a].end-1) for a,b in zip(sr,sr[1:])]
        gs=max(gaps) if dashed and gaps else 0
        t=v36.Track(tid,o,axis,sr,st,en,en-st+1,ink,dashed,gs)
        tracks[tid]=t
        for rid in rids: run_to_track[rid]=tid
        tid+=1
    return tracks,run_to_track,{
        'event_track_links':event_links,'low_quality_collinear_repair_links':repair_links,
        'dashed_sequences':sh+sv,'dashed_pair_links':len(dh)+len(dv),
        'line_tracks':len(tracks),'dashed_tracks':sum(t.dashed for t in tracks.values())}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('-o','--output',default='AP1000_v37_TOLERANT_LONGEST_TRACK_RESULT')
    ap.add_argument('--reuse-v34-dir',default=str(HERE/'AP1000_v34_AXIS_RELATIVE_RESULT'))
    a=ap.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    t0=time.time()
    g=v34.load(Path(a.input)); ink=v34.binarize(g); raw_sk=v34.skeletonize(ink)
    margin,median_radius=estimate_gap_margin(ink,raw_sk)
    rs,hb,vb,hid,vid=v34.extract(raw_sk); v34.annotate(rs,hid,vid)
    events,middle=v36.load_reused_v34(Path(a.reuse_v34_dir),rs)
    tracks,run_to_track,tstats=build_tracks_tol(rs,hb,vb,events,raw_sk,margin)
    cands=build_candidates_tol(events,middle,rs,raw_sk,hid,vid,tracks,run_to_track,margin)
    accepted,owned_supp=v36.resolve_global_ownership(cands,tracks,raw_sk.shape)

    cv2.imwrite(str(out/'00_input.png'),g)
    cv2.imwrite(str(out/'01_raw_skeleton.png'),(~raw_sk).astype(np.uint8)*255)
    cv2.imwrite(str(out/'02_v36_reference.png'),cv2.imread(str(HERE/'AP1000_v36_LONGEST_TRACK_RESULT'/'03_v36_longest_track_hierarchy.png')))
    cv2.imwrite(str(out/'03_v37_tolerant_hierarchy.png'),v36.draw(g,rs,cands,accepted))
    v36.save_csv(out,cands,tracks,rs)
    summ={
      'image_size':[g.shape[1],g.shape[0]],
      'estimated_missing_pixel_margin_px':margin,
      'median_skeleton_stroke_radius_px':round(median_radius,3),
      'all_hv_runs':len(rs),'v34_middle_candidates':len(middle),
      'orientation_children_suppressed':sum(c.status=='ORIENTATION_CHILD' for c in cands),
      'pixel_ownership_children_suppressed':owned_supp,'accepted_components':len(accepted),
      'recognized_dashed_tracks_touching_components':len(set(t for c in cands for t in c.dashed_tracks)),
      **tstats,'runtime_sec':round(time.time()-t0,3),
      'line_tolerance':'same-axis gaps <= data-derived margin are linked only when no off-axis geometry occupies the gap',
      'shape_tolerance':'8-neighbour extended by data-derived missing-pixel margin on ORIGINAL skeleton',
      'bbox_padding_px':0,
      'hierarchy':'parent orientation only; longest longitudinal line-track dominant',
      'manual_margin':False
    }
    (out/'summary.json').write_text(json.dumps(summ,indent=2),encoding='utf-8')
    print(json.dumps(summ,indent=2))

if __name__=='__main__': main()

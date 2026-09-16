#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AP1000 v36 — Orientation-Relative Longest-Line-Track Hierarchy
===============================================================

Unified rules
-------------
1. Extract every horizontal/vertical raster run from the raw skeleton.
2. Build H/V line-tracks.  A track joins:
   - same-axis carrier fragments already linked by the v34 FSM event graph; and
   - conservative, scale-free dashed-line sequences (>=3 collinear runs whose
     intervening gaps do not exceed the adjacent dash support).
3. For a component event, hierarchy competes ONLY in the parent orientation:
      H event -> H tracks only; V event -> V tracks only.
   Among those touching the event displacement pixels, the longest longitudinal
   track is dominant.  If the event's own track is not dominant, it is child
   geometry and is suppressed.
4. Recover component shape from RAW skeleton pixels by 8-neighbour connectivity
   with only the selected parent axis forbidden inside the event's longitudinal
   slab.  H/V/diagonal strokes inside the symbol are never globally removed.
5. If a recognized dashed track touches the recovered component, all dashes of
   that track are included in the component pixel set/bbox, but the far end is
   not recursively flood-filled into unrelated structures.
6. Remaining candidates are processed in descending dominant-track span.  If a
   shorter candidate's exact component pixel set intersects an already-owned
   longer component pixel set, it is a child/duplicate and is suppressed.

No IoU, no overlap percentage, no text filter, no Hough/LSD, no manual line
length threshold.  Dashed recognition uses only relative dash/gap geometry.
"""
from __future__ import annotations
import argparse, csv, json, time, sys
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional

import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
import AP1000_v34_AXIS_RELATIVE_FSM_RUN_GRAPH as v34


class DSU:
    def __init__(self,n):
        self.p=list(range(n)); self.sz=[1]*n
    def find(self,a):
        p=self.p
        while p[a]!=a:
            p[a]=p[p[a]]; a=p[a]
        return a
    def union(self,a,b):
        a=self.find(a); b=self.find(b)
        if a==b: return a
        if self.sz[a]<self.sz[b]: a,b=b,a
        self.p[b]=a; self.sz[a]+=self.sz[b]
        return a

@dataclass
class Track:
    id:int
    o:str
    axis:int
    run_ids:List[int]
    start:int
    end:int
    span:int
    ink:int
    dashed:bool
    dash_gap_scale:int=0

@dataclass
class Candidate:
    id:int
    event:v34.Event
    source_component:v34.Component
    own_track:int
    dominant_track:int
    own_span:int
    dominant_span:int
    competing_tracks:int
    pixels:Set[int]
    bbox:Tuple[int,int,int,int]
    dashed_tracks:Set[int]
    status:str='PENDING'
    owner:Optional[int]=None
    reason:str=''


def event_key(e): return (e.o,e.axis,e.before,e.after)


def detect_dashed_links(rs, by):
    """Return run-pairs that form repeated collinear dash sequences.

    A pair is locally dash-compatible when its positive gap is no larger than
    the longer of its two observed dash supports.  Only maximal sequences with
    at least THREE runs (two compatible gaps) are accepted.  Thus a single
    component interruption is never called a dashed line merely because it is a
    gap.  There is no absolute pixel threshold.
    """
    links=[]; seq_count=0
    for axis,ids0 in by.items():
        ids=sorted(ids0,key=lambda i:rs[i].start)
        if len(ids)<3: continue
        compat=[]
        for a_id,b_id in zip(ids,ids[1:]):
            a,b=rs[a_id],rs[b_id]
            gap=b.start-a.end-1
            compat.append(gap>0 and gap<=max(a.length,b.length))
        i=0
        while i<len(compat):
            if not compat[i]: i+=1; continue
            j=i
            while j+1<len(compat) and compat[j+1]: j+=1
            # gaps i..j => runs i..j+1, require >=3 runs => >=2 gaps
            if j-i+1>=2:
                seq_count+=1
                for k in range(i,j+1): links.append((ids[k],ids[k+1]))
            i=j+1
    return links,seq_count


def build_tracks(rs,hb,vb,events):
    d=DSU(len(rs)); event_links=0
    for e in events:
        d.union(e.before,e.after); event_links+=1
    dh,sh=detect_dashed_links(rs,hb); dv,sv=detect_dashed_links(rs,vb)
    dashed_edges=set()
    for a,b in dh+dv:
        d.union(a,b); dashed_edges.add((min(a,b),max(a,b)))
    groups=defaultdict(list)
    for r in rs: groups[d.find(r.id)].append(r.id)

    tracks={}; run_to_track={}; tid=0
    for root,rids in groups.items():
        # DSU only joins same orientation/axis by construction.
        rr=[rs[i] for i in rids]; o=rr[0].o; axis=rr[0].axis
        s=min(r.start for r in rr); e=max(r.end for r in rr)
        ink=sum(r.length for r in rr)
        rset=set(rids)
        dashed=any(a in rset and b in rset for a,b in dashed_edges)
        sr=sorted(rids,key=lambda i:rs[i].start)
        gaps=[max(0,rs[b].start-rs[a].end-1) for a,b in zip(sr,sr[1:])]
        gap_scale=max(gaps) if dashed and gaps else 0
        t=Track(tid,o,axis,sr,s,e,e-s+1,ink,dashed,gap_scale)
        tracks[tid]=t
        for rid in rids: run_to_track[rid]=tid
        tid+=1
    return tracks,run_to_track,{
        'event_track_links':event_links,
        'dashed_sequences':sh+sv,
        'dashed_pair_links':len(dh)+len(dv),
        'line_tracks':len(tracks),
        'dashed_tracks':sum(t.dashed for t in tracks.values())
    }


def raw_event_pixels(e,rs,sk):
    """Full 8-neighbour displacement in the event longitudinal slab.

    Only the selected parent axis itself is forbidden.  Search is unbounded in
    the normal direction, so valve outlines/branches are fully retained.  The
    longitudinal slab prevents escape into unrelated upstream/downstream network.
    Both boundaries are seeded; disconnected pieces such as orifice plates are
    unioned under the same FSM event identity.
    """
    H,W=sk.shape; a,b=rs[e.before],rs[e.after]
    lo=a.end; hi=b.start; axis=e.axis
    seeds=list(v34.offaxis_endpoint_seeds(a,1,sk))+list(v34.offaxis_endpoint_seeds(b,0,sk))
    seeds=list(dict.fromkeys(seeds))
    if not seeds: return set()
    seen=set(); out=set()
    for sx,sy in seeds:
        sf=sy*W+sx
        if sf in seen: continue
        dq=deque([(sx,sy)]); seen.add(sf)
        while dq:
            x,y=dq.popleft()
            if e.o=='H':
                if x<lo or x>hi or y==axis: continue
            else:
                if y<lo or y>hi or x==axis: continue
            if not sk[y,x]: continue
            f=y*W+x; out.add(f)
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    if dx==0 and dy==0: continue
                    xx=x+dx; yy=y+dy
                    if not (0<=xx<W and 0<=yy<H): continue
                    if e.o=='H':
                        if xx<lo or xx>hi or yy==axis: continue
                    else:
                        if yy<lo or yy>hi or xx==axis: continue
                    ff=yy*W+xx
                    if ff in seen or not sk[yy,xx]: continue
                    seen.add(ff); dq.append((xx,yy))
    return out


def pix_bbox(pix,W,fallback):
    if not pix: return fallback
    xs=[]; ys=[]
    for f in pix:
        y,x=divmod(f,W); xs.append(x); ys.append(y)
    return (min(xs),min(ys),max(xs),max(ys))


def track_pixels(track:Track,rs,W):
    out=set()
    if track.o=='H':
        for rid in track.run_ids:
            r=rs[rid]; base=r.axis*W
            out.update(base+x for x in range(r.start,r.end+1))
    else:
        for rid in track.run_ids:
            r=rs[rid]
            out.update(y*W+r.axis for y in range(r.start,r.end+1))
    return out


def touching_dashed_tracks(pix,tracks,run_to_track,hid,vid,W):
    """Recognized dashed tracks that actually START/END at the component.

    A component-derived dashed line is directional: one longitudinal END of the
    dashed track must meet the component (or begin within that track's own
    observed dash-gap scale).  A middle dash merely crossing/touching the
    component is not enough.  This avoids absorbing unrelated dashed/text
    structures while still bridging the natural first dash gap.
    """
    found=set()
    if not pix: return found
    xs=[]; ys=[]
    for f in pix:
        y,x=divmod(f,W); xs.append(x); ys.append(y)
    x1,x2=min(xs),max(xs); y1,y2=min(ys),max(ys)
    for tid,t in tracks.items():
        if not t.dashed or t.dash_gap_scale<=0: continue
        if t.o=='H':
            if not (y1<=t.axis<=y2): continue
            # left end starts just to the right of C, or right end terminates just left of C;
            # endpoint already inside C projection counts as distance zero.
            d_start = max(0, t.start-x2-1) if t.start>=x1 else None
            d_end   = max(0, x1-t.end-1) if t.end<=x2 else None
            endpoint_attached = ((d_start is not None and d_start<=t.dash_gap_scale) or
                                 (d_end is not None and d_end<=t.dash_gap_scale))
        else:
            if not (x1<=t.axis<=x2): continue
            d_start = max(0, t.start-y2-1) if t.start>=y1 else None
            d_end   = max(0, y1-t.end-1) if t.end<=y2 else None
            endpoint_attached = ((d_start is not None and d_start<=t.dash_gap_scale) or
                                 (d_end is not None and d_end<=t.dash_gap_scale))
        if endpoint_attached: found.add(tid)
    return found

def same_orientation_tracks_in_pixels(e,pix,own_tid,run_to_track,hid,vid,W):
    tids={own_tid}; arr=hid if e.o=='H' else vid
    for f in pix:
        y,x=divmod(f,W); rid=int(arr[y,x])
        if rid>=0 and rid in run_to_track: tids.add(run_to_track[rid])
    return tids


def build_candidates(events,middle,rs,sk,hid,vid,tracks,run_to_track):
    H,W=sk.shape
    em={event_key(e):e for e in events}
    out=[]
    for c in middle:
        e=em.get((c.o,c.axis,c.before,c.after))
        if e is None: continue
        pix=raw_event_pixels(e,rs,sk)
        own=run_to_track[e.before]
        same=same_orientation_tracks_in_pixels(e,pix,own,run_to_track,hid,vid,W)
        # Longitudinal span is primary hierarchy; ink is deterministic tie-break.
        dom=max(same,key=lambda tid:(tracks[tid].span,tracks[tid].ink,-tid))
        dashed=touching_dashed_tracks(pix,tracks,run_to_track,hid,vid,W)
        # Include recognized component-derived dashed line itself in geometry/bbox.
        for tid in dashed:
            if tid==dom: continue
            pix.update(track_pixels(tracks[tid],rs,W))
        bb=pix_bbox(pix,W,c.bbox)
        cand=Candidate(len(out),e,c,own,dom,tracks[own].span,tracks[dom].span,len(same),pix,bb,dashed)
        if own!=dom:
            cand.status='ORIENTATION_CHILD'
            cand.reason='LONGER_SAME_ORIENTATION_TRACK_IN_DISPLACEMENT'
        out.append(cand)
    return out


def sets_touch(a:Set[int],b:Set[int],W:int):
    """Exact 8-neighbour pixel-set contact/intersection, no overlap ratio."""
    if not a or not b: return False
    if len(a)>len(b): a,b=b,a
    if any(f in b for f in a): return True
    # Only boundary adjacency; useful when parent-axis removal leaves one raster-pixel split.
    for f in a:
        y,x=divmod(f,W)
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dx==0 and dy==0: continue
                if (y+dy)*W+(x+dx) in b: return True
    return False


def resolve_global_ownership(cands,tracks,shape):
    H,W=shape
    eligible=[i for i,c in enumerate(cands) if c.status=='PENDING']
    eligible.sort(key=lambda i:(cands[i].dominant_span,tracks[cands[i].dominant_track].ink,len(cands[i].pixels)),reverse=True)
    accepted=[]; suppressed=0
    for i in eligible:
        c=cands[i]; owner=None
        for j in accepted:
            p=cands[j]
            # Cheap bbox rejection before exact pixel ownership test.
            ax1,ay1,ax2,ay2=p.bbox; bx1,by1,bx2,by2=c.bbox
            if ax2<bx1-1 or bx2<ax1-1 or ay2<by1-1 or by2<ay1-1: continue
            if sets_touch(p.pixels,c.pixels,W): owner=j; break
        if owner is not None:
            c.status='PIXEL_OWNERSHIP_CHILD'; c.owner=cands[owner].id
            c.reason='TOUCHES_COMPONENT_OWNED_BY_LONGER_DOMINANT_TRACK'; suppressed+=1
        else:
            c.status='ACCEPTED'; accepted.append(i)
    return accepted,suppressed


def draw(g,rs,cands,accepted):
    im=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)
    for i in accepted:
        c=cands[i]
        # Show the bracketing carrier runs, not all hypotheses.
        for rid in (c.event.before,c.event.after):
            r=rs[rid]
            if r.o=='H': cv2.line(im,(r.start,r.axis),(r.end,r.axis),(0,180,0),1)
            else: cv2.line(im,(r.axis,r.start),(r.axis,r.end),(0,180,0),1)
        x1,y1,x2,y2=c.bbox; cv2.rectangle(im,(x1,y1),(x2,y2),(255,0,255),1)
    return im


def save_csv(out,cands,tracks,rs):
    with open(out/'component_hierarchy.csv','w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['candidate_id','status','owner_candidate_id','reason','orientation','axis','before_run','after_run','own_track','dominant_track','own_track_span','dominant_track_span','same_orientation_track_count','dashed_tracks_included','pixel_count','x1','y1','x2','y2','width','height'])
        for c in cands:
            x1,y1,x2,y2=c.bbox
            w.writerow([c.id,c.status,c.owner,c.reason,c.event.o,c.event.axis,c.event.before,c.event.after,c.own_track,c.dominant_track,c.own_span,c.dominant_span,c.competing_tracks,';'.join(map(str,sorted(c.dashed_tracks))),len(c.pixels),x1,y1,x2,y2,x2-x1+1,y2-y1+1])
    with open(out/'line_tracks.csv','w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['track_id','orientation','axis','start','end','span','ink_length','is_dashed','dash_gap_scale','run_count','run_ids'])
        for tid in sorted(tracks):
            t=tracks[tid]; w.writerow([t.id,t.o,t.axis,t.start,t.end,t.span,t.ink,int(t.dashed),t.dash_gap_scale,len(t.run_ids),';'.join(map(str,t.run_ids))])



def load_reused_v34(reuse_dir:Path, rs):
    events=[]; middle=[]
    with open(reuse_dir/'fsm_events.csv',encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            events.append(v34.Event(row['orientation'],int(row['axis']),int(row['before_id']),int(row['after_id']),row['source'],int(row['merged_gap_count'])))
    with open(reuse_dir/'middle_component_candidates.csv',encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            bb=(int(row['x1']),int(row['y1']),int(row['x2']),int(row['y2']))
            middle.append(v34.Component(int(row['id']),row['orientation'],int(row['axis']),'MIDDLE',int(row['before_run']),int(row['after_run']),row['source'],bb,0,0,
                                        (rs[int(row['before_run'])].end), (rs[int(row['after_run'])].start)))
    return events,middle

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('-o','--output',default='AP1000_v36_LONGEST_TRACK_RESULT')
    ap.add_argument('--reuse-v34-dir',default=None,help='Reuse v34 fsm_events/middle_component_candidates for the same raster input')
    a=ap.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    t0=time.time()
    g=v34.load(Path(a.input)); ink=v34.binarize(g); sk=v34.skeletonize(ink)
    rs,hb,vb,hid,vid=v34.extract(sk); v34.annotate(rs,hid,vid)
    if a.reuse_v34_dir:
        events,middle=load_reused_v34(Path(a.reuse_v34_dir),rs)
        seqs=v34.sequences(events,rs)
    else:
        v34._CONN_CACHE.clear()
        ph,hc,hd=v34.base_adjacent_events(rs,hb,'H',sk,hid,vid)
        pv,vc,vd=v34.base_adjacent_events(rs,vb,'V',sk,hid,vid)
        eh,mth,mgh=v34.merge_internal_axis_fragments(ph,rs,sk,'H')
        ev,mtv,mgv=v34.merge_internal_axis_fragments(pv,rs,sk,'V')
        events=eh+ev; seqs=v34.sequences(events,rs)
        idxH=v34.build_normal_run_index(sk,'H'); idxV=v34.build_normal_run_index(sk,'V')
        comps=v34.recover_components(events,seqs,rs,idxH,idxV,sk.shape)
        middle=[c for c in comps if c.kind=='MIDDLE']

    tracks,run_to_track,tstats=build_tracks(rs,hb,vb,events)
    cands=build_candidates(events,middle,rs,sk,hid,vid,tracks,run_to_track)
    accepted,owned_supp=resolve_global_ownership(cands,tracks,sk.shape)

    cv2.imwrite(str(out/'00_input.png'),g)
    cv2.imwrite(str(out/'01_skeleton.png'),(~sk).astype(np.uint8)*255)
    cv2.imwrite(str(out/'02_v34_before_hierarchy.png'),v34.draw(g,rs,seqs,middle))
    cv2.imwrite(str(out/'03_v36_longest_track_hierarchy.png'),draw(g,rs,cands,accepted))
    save_csv(out,cands,tracks,rs)

    summ={
      'image_size':[g.shape[1],g.shape[0]],
      'all_hv_runs':len(rs),
      'v34_middle_candidates':len(middle),
      'orientation_children_suppressed':sum(c.status=='ORIENTATION_CHILD' for c in cands),
      'pixel_ownership_children_suppressed':owned_supp,
      'accepted_components':len(accepted),
      'recognized_dashed_tracks_touching_components':len(set(t for c in cands for t in c.dashed_tracks)),
      **tstats,
      'runtime_sec':round(time.time()-t0,3),
      'hierarchy':'within each event, compare only parent orientation; longest longitudinal line-track is dominant',
      'shape_recovery':'8-neighbour raw-skeleton connectivity in event slab with only parent axis forbidden',
      'dashed_component_rule':'recognized dashed track touching component is included in bbox geometry',
      'global_child_rule':'exact component pixel-set contact with longer dominant-track owner; no percentage/IoU',
      'text_filter':False,'hough_lsd':False,'manual_length_threshold':False,'iou_threshold':False,'overlap_percentage':False
    }
    (out/'summary.json').write_text(json.dumps(summ,indent=2),encoding='utf-8')
    print(json.dumps(summ,indent=2))

if __name__=='__main__': main()

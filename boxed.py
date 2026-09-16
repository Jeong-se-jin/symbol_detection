"""박스 태그·커넥터 — pkg9/symbol_boxed.py 를 그대로 옮겼다.

가져온 것 (13행 이후 전부)
  connector_flags · _end_taper   오프페이지 커넥터, 장축 단면 테이퍼 프로파일
  boxed_tags · _grow · _cells    닫힌 윤곽 + 내부 잉크, 칸막이 넘어 확장

import 만 shape_geom → geom 으로 바꿨고 함수 본문은 손대지 않았다.
'TAG' / 'TANK' 라벨은 분류가 아니라 이 검출기가 짧은 변 길이로 가른 갈래의
이름이다 — detect.py 가 geometry 안에 추적용으로만 담는다.
"""
from __future__ import annotations

import cv2
import numpy as np

from geom import iom, iom_nms


# ---------------------------------------------------------------- connectors.py (전부)

"""Off-page connector flags: length-invariant taper-profile geometry.

These symbols carry the destination drawing number, so their body length grows
with the text inside. Reference-template ROI-IoU cannot cover that: the search
only spans three scales (.95/1/1.05), so a fixed-length template misses long
instances and fires on partial matches along the parallel body edges. Detection
therefore uses the cross-section profile along the long axis, whose plateau and
tip taper are unchanged by length.

Constants are AP1000 page_001 measurements (31 instances, tip depth ratio
.4262-.4426, body height 61 px), not universal symbol geometry. No OCR of the
interior drawing number is performed here, so the destination is unresolved.
"""


def _end_taper(hs, slack=.06, straight=.90):
    """Taper length in samples at the high end of a normalized profile.

    Returns 0 for a flat end, None when the end is not a straight monotonic
    taper. Walking inward from the end keeps interior profile noise, such as a
    gap in the outline or the pipe stub, out of the measurement.
    """
    i = len(hs) - 1
    while i >= 0 and hs[i] < .90:
        i -= 1
    d = len(hs) - 1 - i
    if d < 1:
        return 0
    seg = hs[i + 1:]
    if np.any(np.diff(seg) > slack):
        return None
    if len(seg) >= 3:
        ramp = np.linspace(1., 0., len(seg))
        if np.corrcoef(seg, ramp)[0, 1] < straight:
            return None
    return int(d)


def connector_flags(raw, min_short_frac=.008, max_long_frac=.20,
                    depth=(.35, .60), aspect=(1.2, 12.), nms=.70,
                    dominant_tol=.12):
    """Locate pentagon/hexagon off-page connectors in the raw binary page.

    CHAIN_APPROX_NONE is required: the profile needs every boundary pixel.
    RETR_LIST returns both the outer and the hole boundary of the outline, which
    complement each other. An attached pipe merges into the outer boundary but
    leaves the hole intact; interior text touching the outline breaks the hole
    but leaves the outer boundary intact. Duplicates are removed by IoM NMS.
    """
    h, w = raw.shape
    lo = max(16, round(min(h, w) * min_short_frac))
    hi = round(max(h, w) * max_long_frac)
    rows = []
    for c in cv2.findContours(raw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]:
        x, y, cw, ch = cv2.boundingRect(c)
        short, long = min(cw, ch), max(cw, ch)
        if short < lo or long > hi:
            continue
        if not aspect[0] <= long / short <= aspect[1]:
            continue
        axis = 0 if cw >= ch else 1
        pts = c[:, 0, :]
        u = pts[:, axis] - (x if axis == 0 else y)
        v = pts[:, 1 - axis] - (y if axis == 0 else x)
        n = cw if axis == 0 else ch
        top = np.full(n, -1, np.int64); bot = np.full(n, 1 << 30, np.int64)
        np.maximum.at(top, u, v); np.minimum.at(bot, u, v)
        prof = np.where(top >= 0, top - bot + 1, 0).astype(float)
        body = float(np.percentile(prof, 90))
        if body <= 0:
            continue
        hs = prof / body
        d_hi = _end_taper(hs); d_lo = _end_taper(hs[::-1])
        if d_hi is None or d_lo is None:
            continue
        if d_hi == 0 and d_lo == 0:
            continue
        if any(d and not depth[0] <= d / short <= depth[1] for d in (d_hi, d_lo)):
            continue
        # A pure triangle has no body. Connectors always carry a flat run.
        if int(np.count_nonzero(hs >= .90)) < 2:
            continue
        if d_hi and d_lo:
            heading = 'BIDIRECTIONAL'
        elif axis == 0:
            heading = 'RIGHT' if d_hi else 'LEFT'
        else:
            heading = 'DOWN' if d_hi else 'UP'
        ratio = max(d_hi, d_lo) / short
        margin = max(3, round(short * .18))
        interior = raw[y + margin:y + ch - margin, x + margin:x + cw - margin]
        rows.append({'bbox': [int(x), int(y), int(x + cw), int(y + ch)],
            'score': round(float(ratio), 6), 'tip_depth_ratio': round(float(ratio), 4),
            'body_height': int(short), 'heading': heading,
            # Recorded, never gated: empty flags exist and their tag sits outside.
            'interior_ink': round(float(interior.mean()), 4) if interior.size else 0.,
            'label': 'OFF-PAGE CONNECTOR', 'family': 'CONNECTOR', 'subtype': None,
            'confidence': 'FAMILY_ONLY', 'classification_status': 'FAMILY_ONLY',
            'tag_text': None, 'method': 'taper_profile_length_invariant',
            'score_kind': 'tip_depth_ratio_not_class_probability'})
    rows = iom_nms(rows, nms)
    # One sheet draws its connectors at a single body height; only the length
    # varies. Rounded vessel ends and leader arrows do not share that height.
    if len(rows) >= 6:
        hs = np.array([r['body_height'] for r in rows], float)
        support = np.array([np.count_nonzero(np.abs(hs - v) <= dominant_tol * v) for v in hs])
        dominant = float(hs[support.argmax()])
        if int(support.max()) >= max(4, .50 * len(rows)):
            rows = [r for r, v in zip(rows, hs) if abs(v - dominant) <= dominant_tol * dominant]
            for r in rows:
                r['dominant_body_height'] = dominant
    return rows


# ---------------------------------------------------------------- boxed_tags.py (전부)

"""Boxed code tags: closed outlines whose interior text is instance-specific.

Detection is geometric for the same reason connectors.py is: the interior text
varies per instance, so a reference template would either match only the one
code it was cut from, or - with the code erased - lose precision to the code ink
that the convex-hull ROI still covers.
"""


def _grow(raw, bbox, limit=300):
    """Extend a cell across any divider it sits against, out to the far cap.

    Whether the two parallel edges carry on past an end is what tells a shared
    divider from the tag's own wall: a square tag's end wall is full-height ink
    just as a divider is, but nothing continues beyond it. The edges stop short
    of a rounded end, which curves inward, so a walk that did advance finishes
    by following whatever ink remains inside the band for one cap radius.
    """
    x0, y0, x1, y1 = bbox
    horiz = (x1 - x0) > (y1 - y0)
    cap = min(x1 - x0, y1 - y0) // 2 + 2
    span = raw.shape[1] if horiz else raw.shape[0]
    out = [x0, y0, x1, y1]
    for step in (1, -1):
        p = (x1 if step > 0 else x0 - 1) if horiz else (y1 if step > 0 else y0 - 1)
        n = 0
        while 0 <= p < span and n < limit:
            if horiz:
                edges = raw[max(0, y0 - 1):y0 + 2, p].any() and raw[y1 - 2:y1 + 1, p].any()
            else:
                edges = raw[p, max(0, x0 - 1):x0 + 2].any() and raw[p, x1 - 2:x1 + 1].any()
            if not edges:
                break
            p += step; n += 1
        if n == 0:
            continue
        c = 0
        while 0 <= p < span and c < cap:
            band = raw[y0:y1, p] if horiz else raw[p, x0:x1]
            if not band.any():
                break
            p += step; c += 1
        out[(0 if step < 0 else 2) + (0 if horiz else 1)] = p if step > 0 else p + 1
    return out


def _cells(raw, bbox, min_frac=.05):
    """Interior regions the dividers cut the tag into, and their extent.

    A cell is enclosed by the outline, so it never reaches the edge of the box.
    A pipe stub merged into the boundary does reach it, and the white it strands
    beside itself would otherwise count as a cell and stretch the measurement.
    Flooding from a padded corner clears the page outside the outline first, so
    the seed never lands on ink however square the corners are.

    The outline is closed by a pixel before flooding. Straight edges on this
    sheet stop a pixel or two short of the cap arcs they meet, and a diagonal
    step of that size does not stop a four-connected flood: without the close,
    the far half of every divided tag reads as open page. Cell extents are taken
    back out by the same pixel so the walls still land where they are drawn.
    """
    x0, y0, x1, y1 = bbox
    sub = raw[y0:y1, x0:x1]
    if sub.size == 0:
        return 0, None
    h, w = sub.shape
    closed = cv2.dilate(sub, np.ones((3, 3), np.uint8))
    ff = cv2.copyMakeBorder((1 - closed).astype('uint8'), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    cv2.floodFill(ff, None, (0, 0), 0)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ff[1:-1, 1:-1], 4)
    keep = []
    for i in range(1, n):
        L, T = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP]
        W, H = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        if st[i, cv2.CC_STAT_AREA] < min_frac * w * h:
            continue
        if L == 0 or T == 0 or L + W == w or T + H == h:
            continue
        keep.append((max(0, L - 1), max(0, T - 1), min(w, L + W + 1), min(h, T + H + 1)))
    if not keep:
        return 0, None
    L = min(c[0] for c in keep); T = min(c[1] for c in keep)
    R = max(c[2] for c in keep); B = max(c[3] for c in keep)
    # Step out onto the wall the outline is drawn on, no further: beyond a wall
    # of a couple of pixels there is only the stub this trim exists to shed.
    def out(lo, hi, axis, step, wall=2, solid=.5):
        """Step from a cell onto the wall, across it, and no further.

        A wall spans most of the cell, a stub spans a couple of pixels of it, and
        a rounded end spans little at any single line. So a line is crossed while
        it carries ink, except that once a solid line has been crossed only solid
        lines follow it: that is what stops the walk at the far face of the wall
        instead of running out along the stub beyond it.
        """
        p = lo if step < 0 else hi - 1
        crossed_wall = False
        for _ in range(wall):
            q = p + step
            if not 0 <= q < sub.shape[1 - axis]:
                break
            line = sub[T:B, q] if axis == 0 else sub[q, L:R]
            if not line.any():
                break
            filled = float(line.mean()) >= solid
            if crossed_wall and not filled:
                break
            p = q
            crossed_wall = crossed_wall or filled
        return p
    # int(): a width read off a connected component is numpy, which compares
    # equal to an int but is refused by the json enhanced.py writes.
    return len(keep), [int(x0 + out(L, R, 0, -1)), int(y0 + out(T, B, 1, -1)),
                       int(x0 + out(L, R, 0, 1)) + 1, int(y0 + out(T, B, 1, 1)) + 1]


def boxed_tags(raw, fill=.83, ink=(.01, .60), margin_frac=.20, nms=.70,
               min_aspect=1.6, min_short_frac=.004, tank_short_frac=.02,
               exclude=(), claimed=.50):
    """Locate closed outlines carrying interior ink in the raw binary page.

    A ring's outer boundary encloses nearly its whole bounding box, which is what
    separates an outline from open ink such as text or a pipe junction.

    Unlike connectors.py, interior ink is gated rather than merely recorded:
    these tags are defined by the code they carry, so an empty outline is a pipe
    end or a blank box. The upper bound rejects solid blobs, whose outer
    boundary also encloses its bounding box and so passes the fill test.

    Elongation is what separates these tags from the square and round frames of
    instrument_frames, which carry a code inside a closed outline too. That
    detector accepts .80-1.25, so this one starts above it and the two cannot
    both claim the same symbol.

    Small boxed codes share every other property of a tag and outnumber them on
    a real sheet, so a page-relative floor on the short side is what keeps the
    fill test loose enough to admit outlines nicked by an attached pipe. On
    page_001 the tags measure 31 and 46 px short; the codes measure 9 to 12.

    A tank is the same geometry at another scale, and it carries a name and
    nothing else, so its interior ink sits an order below a code tag's: .022 for
    the core makeup tanks against .05 upward for the tags. The floor is set below
    both rather than exempting size, because a blank outline measures exactly
    zero and the comparison is strict, so pipe ends stay out either way. Short
    side then separates the two: on page_001 the tags measure 31 and 46 px and
    the tanks 181, so a tank is reported under its own family rather than
    labelled a tag.

    TANK, not VESSEL: Figure 1.7-2 Note 16 reserves VESSEL (type code MV) for
    equipment holding internal equipment and TANK (MT) for equipment holding
    only process fluid, explicitly regardless of pressure rating. Everything
    this detector reaches on page_001 is tagged MT.

    A stub of pipe merged into the boundary stretches its box, so the box is
    finally cut back to one pixel outside the cells, which is the wall the
    outline is drawn on. Reporting the stub as tag length would put pipe into
    the size that tells the tag families apart.

    A divided tag is one symbol, but only its cells clear the fill test: the
    divider runs on past the outline as a stub, which drops the fill of the whole
    outline to .37-.73 on page_001. Each cell is therefore grown across its
    divider, and the grown boxes, now describing the same tag, collapse in NMS.

    An off-page connector is a closed outline carrying text too, so the rows of
    a pass that already claimed such outlines are passed in as exclude and their
    overlaps are dropped rather than reported twice.

    RETR_LIST returns both the outer and the hole boundary of an outline more
    than one pixel thick, and both pass the fill test. IoM NMS keeps one, ranked
    by how completely the boundary encloses its box, so the outer one wins.
    """
    lo = max(16, round(min(raw.shape) * min_short_frac))
    # An absolute floor as well as a page-relative one, so the split does not
    # collapse onto the tags when the page itself is small.
    tank = max(64, round(min(raw.shape) * tank_short_frac))
    rows = []
    for c in cv2.findContours(raw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area <= 0 or area / (cw * ch) < fill:
            continue
        if min(cw, ch) < lo:
            continue
        if max(cw, ch) / max(1, min(cw, ch)) < min_aspect:
            continue
        m = max(2, round(min(cw, ch) * margin_frac))
        interior = raw[y + m:y + ch - m, x + m:x + cw - m]
        if interior.size == 0 or not ink[0] < float(interior.mean()) < ink[1]:
            continue
        bbox = _grow(raw, [int(x), int(y), int(x + cw), int(y + ch)])
        cells, walls = _cells(raw, bbox)
        if walls:
            bbox = walls
        short = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
        big = short >= tank
        rows.append({'bbox': bbox,
            'score': round(float(area / (cw * ch)), 6),
            'body_height': int(min(bbox[2] - bbox[0], bbox[3] - bbox[1])),
            'body_length': int(max(bbox[2] - bbox[0], bbox[3] - bbox[1])),
            'cells': cells,
            'interior_ink': round(float(interior.mean()), 4),
            'label': 'TANK' if big else 'BOXED CODE TAG',
            'family': 'TANK' if big else 'TAG', 'subtype': None,
            'confidence': 'FAMILY_ONLY', 'classification_status': 'FAMILY_ONLY',
            'tag_text': None, 'method': 'closed_outline_with_interior_ink',
            'score_kind': 'boundary_box_fill_not_class_probability'})
    rows = iom_nms(rows, nms)
    return [r for r in rows
            if not any(iom(r['bbox'], e['bbox']) > claimed for e in exclude)]

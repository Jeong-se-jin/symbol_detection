"""닫힌/점선 윤곽 검출 — pkg9/symbol_outline.py 에서 기하 부분만 옮겼다.

가져온 것 (24-181행 그대로)
  instrument_frames   닫힌 사각/원 윤곽 + 내부 잉크
  _profile · dashed_circles   점선 큰 원 (Hough 후보 → 반지름별 잉크 프로파일)

가져오지 않은 것
  label_frames · FUNCTIONS · _tesseract · _read_words (182-245행). 버블 내부를
  크롭해 Tesseract 로 계기 코드를 읽어 subtype 을 붙이는 분류 코드다. 이 폴더는
  기하만 쓰고 클래스를 내지 않기로 했으므로 뺀다 — 덕분에 Tesseract 의존도 없다.
  검출 개수는 영향받지 않는다. label_frames 는 rows 를 제자리에서 고칠 뿐
  넣거나 빼지 않는다.

라벨 문자열은 남겨 둔다
  'INSTRUMENT FRAME' / 'TANK' / 'VESSEL' 은 분류가 아니라 이 검출기가 크기·형태로
  가른 갈래의 이름이다. detect.py 가 geometry 안에 추적용으로만 담는다.
"""
from __future__ import annotations

import cv2
import numpy as np

from geom import iom_nms

# ---------------------------------------------------------------- instrument_frames (native_symbols.py)


def instrument_frames(raw, frame_frac=.028, tank_frac=.07):
    """Closed square and round outlines carrying interior ink.

    Two size bands, not one. Up to frame_frac of the page an outline is an
    instrument frame; above it and up to tank_frac it is a tank, which on this
    sheet means the accumulator tanks at 303 px against bubbles at 91. The
    repeated-scale filter below keeps only sizes near the dominant one, so a
    tank has to sit outside that filter or the bubbles delete it.

    TANK, not VESSEL: Figure 1.7-2 Note 16 reserves VESSEL (type code MV) for
    equipment holding internal equipment; these are tagged MT.
    """
    h, w = raw.shape
    contours, _ = cv2.findContours(raw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rows = []
    lo = max(24, round(min(h, w) * .006))
    frame = max(lo + 1, round(min(h, w) * frame_frac))
    hi = max(frame + 1, round(min(h, w) * tank_frac))
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if not (lo <= min(cw, ch) <= max(cw, ch) <= hi and .80 <= cw / ch <= 1.25):
            continue
        area = cv2.contourArea(c)
        extent = area / (cw * ch)
        if area <= 0:
            continue
        approx = cv2.approxPolyDP(c, .02 * cv2.arcLength(c, True), True)
        form = None
        if len(approx) == 4 and extent > .88:
            form = 'RECTANGULAR'
            quality = min(1., extent)
        elif len(c) >= 8 and .68 < extent < .85:
            (cx, cy), (a, b), _ = cv2.fitEllipse(c)
            if min(a, b) <= 0 or max(a, b) / min(a, b) > 1.2:
                continue
            pts = c[:, 0, :].astype(float)
            radius = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
            error = float(np.std(radius) / max(1., np.mean(radius)))
            if error > .07:
                continue
            form = 'CIRCULAR'; quality = 1 - error
        if form is None:
            continue
        # Require interior ink: exclude empty pipe-end circles and empty boxes.
        margin = max(3, round(min(cw, ch) * .15))
        interior = raw[y + margin:y + ch - margin, x + margin:x + cw - margin]
        if interior.size == 0 or interior.mean() < .015 or interior.mean() > .45:
            continue
        big = min(cw, ch) > frame
        rows.append({'bbox': [x, y, x + cw, y + ch], 'score': round(float(quality), 6),
            'label': 'TANK' if big else 'INSTRUMENT FRAME',
            'family': 'TANK' if big else 'INSTRUMENT', 'subtype': None,
            'frame_form': form, 'confidence': 'FAMILY_ONLY', 'method': 'closed_outline_geometry',
            'score_kind': 'outline_fit_not_class_probability'})
    rows = iom_nms(rows, .70)
    # Repeated frame scale separates instrument bubbles from small boxed function
    # codes (OP, M, S). Keep all when no dominant repeated size exists.
    # Tanks are held out of this: the sheet draws only a couple of them, so the
    # filter that finds the dominant repeated size would always delete them.
    tanks = [r for r in rows if r['family'] == 'TANK']
    rows = [r for r in rows if r['family'] != 'TANK']
    if len(rows) >= 12:
        sizes = np.array([min(r['bbox'][2] - r['bbox'][0], r['bbox'][3] - r['bbox'][1]) for r in rows])
        support = np.array([np.count_nonzero(np.abs(sizes - s) <= .15 * s) for s in sizes])
        dominant = float(sizes[np.argmax(support)])
        if int(support.max()) >= max(10, .60 * len(rows)):
            rows = [r for r, s in zip(rows, sizes) if abs(s - dominant) <= .15 * dominant]
    return rows + tanks


# ---------------------------------------------------------------- dashed_circles.py (전부)


def _profile(raw, x, y, rmax):
    """Ink fraction at each radius from a centre, out to rmax."""
    h, w = raw.shape
    y0, y1 = max(0, y - rmax), min(h, y + rmax + 1)
    x0, x1 = max(0, x - rmax), min(w, x + rmax + 1)
    sub = raw[y0:y1, x0:x1]
    if sub.size == 0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2).astype(int)
    ok = d <= rmax
    total = np.bincount(d[ok], minlength=rmax + 1).astype(float)
    ink = np.bincount(d[ok], weights=sub[ok].astype(float), minlength=rmax + 1)
    return np.where(total > 0, ink / np.maximum(total, 1), 0.)


def dashed_circles(raw, min_frac=.03, max_frac=.07, min_peak=.05, max_base=.005,
                   half_width=40, min_dist=200, canny=100, votes=40):
    """Locate large dashed circular outlines in the raw binary page.

    The circle transform supplies candidate centres; the radial profile decides.
    A candidate is kept when some radius in the search band carries a real
    fraction of ink while the median radius carries almost none, and the peak is
    narrow rather than a broad smear of text. Concentric rings share a centre,
    so the outermost qualifying radius sets the extent: a vessel wall is drawn
    as two rings and its size is the outer one.

    Sizes are page-relative and were read off page_001, where the reactor vessel
    measures 428 px across the inner ring and 490 across the outer, against a
    page 4859 px short. No name is read, so nothing here claims which vessel it
    is; the family rests on this being the one dashed circular outline on the
    sheet and on Figure 1.7-2 Note 16, which reserves VESSEL (type code MV) for
    equipment holding internal equipment and TANK (MT) for equipment holding
    only process fluid.
    """
    short = min(raw.shape)
    rmin, rmax = round(short * min_frac), round(short * max_frac)
    if rmin < 8 or rmax <= rmin:
        return []
    found = cv2.HoughCircles((raw * 255).astype(np.uint8), cv2.HOUGH_GRADIENT, dp=1,
                             minDist=min_dist, param1=canny, param2=votes,
                             minRadius=rmin, maxRadius=rmax)
    if found is None:
        return []
    rows = []
    for x, y, _ in np.round(found[0]).astype(int):
        f = _profile(raw, int(x), int(y), rmax)
        if f is None:
            continue
        band = f[rmin:rmax + 1]
        if band.size == 0:
            continue
        peak = float(band.max())
        if peak < min_peak or float(np.median(band)) > max_base:
            continue
        # The extent is the outermost radius still carrying half the peak, so a
        # wall drawn as two rings is reported at its outer face.
        strong = np.flatnonzero(band >= peak * .5)
        if strong.size > half_width:
            continue
        r = int(rmin + strong.max())
        rows.append({'bbox': [int(x - r), int(y - r), int(x + r), int(y + r)],
            'score': round(peak, 6), 'ring_ink': round(peak, 4),
            'ring_baseline': round(float(np.median(band)), 6),
            'body_height': int(2 * r), 'body_length': int(2 * r),
            'label': 'VESSEL', 'family': 'VESSEL', 'subtype': None,
            'frame_form': 'DASHED CIRCULAR', 'confidence': 'FAMILY_ONLY',
            'classification_status': 'FAMILY_ONLY', 'tag_text': None,
            'method': 'radial_ink_profile_on_circle_transform_candidates',
            'score_kind': 'ring_ink_fraction_not_class_probability'})
    # Concentric rings and near-duplicate centres collapse to the largest.
    rows.sort(key=lambda r: -r['body_length'])
    keep = []
    for r in rows:
        cx, cy = (r['bbox'][0] + r['bbox'][2]) / 2, (r['bbox'][1] + r['bbox'][3]) / 2
        if any(abs(cx - (k['bbox'][0] + k['bbox'][2]) / 2) < min_dist
               and abs(cy - (k['bbox'][1] + k['bbox'][3]) / 2) < min_dist for k in keep):
            continue
        keep.append(r)
    return keep



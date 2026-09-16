"""선 검출 — 방향이 일치하는 잉크의 연속이 하나의 run 이다.

digitization-of-piping-and-instrument-diagrams 의 `tools/pid10/lines.py` 를
**함수 본문 그대로** 옮겼다. 우리 쪽에서 고친 것은 없다.

착상
  픽셀마다 얇은 선 커널 한 벌(기본 18방향)을 씌워 어느 방향에 잉크가 가장 많이
  실리는지 고르고, 같은 답을 낸 이웃끼리 묶어 하나의 run 으로 본다. 세선화를
  하지 않으므로 대각선이 계단으로 부서지지 않고, 접합점 검출기도 두지 않는다 —
  **답이 바뀌는 자리가 곧 접합점**이다.

고쳐야 할 것이 둘 있다
  선이 교차하면 교차당한 쪽이 공유 픽셀을 가져가므로 교차한 선이 두 토막으로
  들어온다. 다시 이어 주지 않으면 그 교차가 접합으로 읽힌다. 그리고 접촉은 두
  run 의 **끝**에서 일어났을 때만 연결이다 — 양쪽 다 중간이면 crossover 이고,
  배관 위를 지나가는 리더가 분기로 오인되지 않는 것이 이 구분 덕이다.

우리 도면에서 왜 필요한가
  carrier(v34/v48)는 H·V 두 축만 본다. 대각선 배관·리더·신호선은 그쪽 관심 밖이라
  배관 그래프를 만들려면 이 갈래가 따로 있어야 한다.
"""
import collections
import math

import cv2
import numpy as np

SHIFTS = ((0, 1), (1, 0), (1, 1), (1, -1))


def line_kernels(count, length):
    size = int(length) | 1
    c = size // 2
    out = []
    for i in range(count):
        theta = math.pi * i / count
        k = np.zeros((size, size), np.float32)
        dx, dy = math.cos(theta), math.sin(theta)
        cv2.line(k, (int(round(c - dx * c)), int(round(c - dy * c))),
                 (int(round(c + dx * c)), int(round(c + dy * c))), 1.0, 1)
        total = k.sum()
        out.append(k / (total if total else 1.0))
    return out


def orientation(ink, count, length):
    best = np.zeros(ink.shape, np.float32)
    which = np.zeros(ink.shape, np.uint8)
    src = ink.astype(np.float32)
    for i, k in enumerate(line_kernels(count, length)):
        r = cv2.filter2D(src, -1, k, borderType=cv2.BORDER_CONSTANT)
        better = r > best
        best[better] = r[better]
        which[better] = i
    return which, best


def _bin_gap(a, b, count):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return np.minimum(d, count - d)


def _geometry(labels, wanted):
    """Principal axis and extent for each run."""
    yy, xx = np.nonzero(labels > 0)
    vv = labels[yy, xx]
    order = np.argsort(vv, kind='stable')
    yy, xx, vv = yy[order], xx[order], vv[order]
    want = np.array(sorted(wanted))
    lo_i = np.searchsorted(vv, want)
    hi_i = np.searchsorted(vv, want, side='right')
    built = {}
    for rid, lo, hi in zip(want.tolist(), lo_i, hi_i):
        if hi <= lo:
            continue
        py = yy[lo:hi].astype(np.float64)
        px = xx[lo:hi].astype(np.float64)
        mx, my = px.mean(), py.mean()
        cxx = ((px - mx) ** 2).mean()
        cyy = ((py - my) ** 2).mean()
        cxy = ((px - mx) * (py - my)).mean()
        theta = 0.5 * math.atan2(2 * cxy, cxx - cyy)
        ux, uy = math.cos(theta), math.sin(theta)
        t = (px - mx) * ux + (py - my) * uy
        lo_t, hi_t = float(t.min()), float(t.max())
        angle = math.degrees(theta) % 180
        built[int(rid)] = {
            'id': int(rid), 'length': round(hi_t - lo_t, 1),
            'pixels': int(hi - lo), 'angle': round(angle, 1),
            'a': [round(mx + lo_t * ux, 1), round(my + lo_t * uy, 1)],
            'b': [round(mx + hi_t * ux, 1), round(my + hi_t * uy, 1)],
            'orient': 'h' if min(angle, 180 - angle) <= 45 else 'v',
        }
    return built


def extract_runs(ink, params, directions=18, tolerance=1, min_support=0.45,
                 min_pixels=4, bridge_degrees=14.0, bridge_offset=3.0):
    """Runs, and every place two of them touch."""
    H, W = ink.shape[:2]
    mask = (ink > 0)
    which, support = orientation(mask, directions, params.probe)
    weak = mask & (support < min_support)
    live = mask & ~weak

    offset, labels = 0, np.zeros(mask.shape, np.int32)
    for b in range(directions):
        m = (live & (which == b)).astype(np.uint8)
        if not m.any():
            continue
        n, lab = cv2.connectedComponents(m, 8)
        labels = np.where(m > 0, lab + offset, labels)
        offset += n

    parent = list(range(offset + 2))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    disagree = {}
    for dy, dx in SHIFTS:
        a = labels[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)]
        b = labels[max(-dy, 0):H + min(-dy, 0), max(-dx, 0):W + min(-dx, 0)]
        wa = which[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)]
        wb = which[max(-dy, 0):H + min(-dy, 0), max(-dx, 0):W + min(-dx, 0)]
        both = (a > 0) & (b > 0) & (a != b)
        gap = _bin_gap(wa, wb, directions)
        same = both & (gap <= tolerance)
        for u, v in zip(a[same].tolist(), b[same].tolist()):
            ra, rb = find(u), find(v)
            if ra != rb:
                parent[rb] = ra
        cut = both & (gap > tolerance)
        ys, xs = np.nonzero(cut)
        oy, ox = max(dy, 0), max(dx, 0)
        for u, v, py, px in zip(a[cut].tolist(), b[cut].tolist(),
                                (ys + oy).tolist(), (xs + ox).tolist()):
            disagree.setdefault((u, v), []).append((px, py))

    root = np.arange(offset + 2, dtype=np.int32)
    for i in range(1, offset + 1):
        root[i] = find(i)
    merged = np.where(labels > 0, root[labels], 0)
    counts = collections.Counter(merged[merged > 0].tolist())
    keep = {i for i, c in counts.items() if i and c >= min_pixels}
    runs = _geometry(merged, keep)

    # Rejoin runs that continue each other across the pixels a crossing stole.
    grid = collections.defaultdict(list)
    cell = max(int(params.bridge_gap * 2), 4)
    for k, r in runs.items():
        for pt in (r['a'], r['b']):
            grid[(int(pt[0] // cell), int(pt[1] // cell))].append((k, pt))
    link = list(range(max(runs) + 2)) if runs else [0, 1]

    def lfind(x):
        while link[x] != x:
            link[x] = link[link[x]]
            x = link[x]
        return x

    joined = 0
    for k, r in runs.items():
        ux = math.cos(math.radians(r['angle']))
        uy = math.sin(math.radians(r['angle']))
        for pt in (r['a'], r['b']):
            cx, cy = int(pt[0] // cell), int(pt[1] // cell)
            for gx in (cx - 1, cx, cx + 1):
                for gy in (cy - 1, cy, cy + 1):
                    for j, qt in grid.get((gx, gy), ()):
                        if j <= k:
                            continue
                        if math.hypot(pt[0] - qt[0], pt[1] - qt[1]) > params.bridge_gap:
                            continue
                        turn = abs(((r['angle'] - runs[j]['angle']) + 90) % 180 - 90)
                        if turn > bridge_degrees:
                            continue
                        off = abs(-(qt[0] - r['a'][0]) * uy + (qt[1] - r['a'][1]) * ux)
                        if off > bridge_offset:
                            continue
                        ra, rb = lfind(k), lfind(j)
                        if ra != rb:
                            link[rb] = ra
                            joined += 1
    if joined:
        remap = np.arange(offset + 2, dtype=np.int32)
        for k in runs:
            remap[k] = lfind(k)
        merged = np.where(merged > 0, remap[merged], 0)
        counts = collections.Counter(merged[merged > 0].tolist())
        keep = {i for i, c in counts.items() if i and c >= min_pixels}
        runs = _geometry(merged, keep)
        root = remap[root]

    contact = {}
    for (u, v), points in disagree.items():
        ru, rv = int(root[u]), int(root[v])
        if ru == rv or ru not in runs or rv not in runs:
            continue
        contact.setdefault((min(ru, rv), max(ru, rv)), []).extend(points)

    return {'labels': merged, 'runs': runs, 'contacts': contact,
            'weak': weak, 'rejoined': joined,
            'no_direction': int(weak.sum())}


def split_at_branches(labels, runs, edges, params):
    """Cut a run where another one tees into it.

    Without this a header is one run from end to end, and a label on its left
    half names its right half too -- everything that branches off anywhere
    along it becomes a neighbour of everything else. A tee is a junction, so the
    run is two runs: the part before it and the part after.

    The run that ends at the contact is left alone. Only the one the contact
    lands in the middle of is cut, and it is cut there.
    """
    cuts = collections.defaultdict(list)
    for (ra, rb), info in edges.items():
        px, py = info['at']
        for rid, at_its_end in ((ra, info['ends'][0]), (rb, info['ends'][1])):
            if at_its_end or rid not in runs:
                continue
            r = runs[rid]
            ux = math.cos(math.radians(r['angle']))
            uy = math.sin(math.radians(r['angle']))
            t = (px - r['a'][0]) * ux + (py - r['a'][1]) * uy
            cuts[rid].append(t)

    if not cuts:
        return labels, runs, 0

    out = labels.copy()
    nxt = int(labels.max()) + 1
    added = 0
    for rid, positions in cuts.items():
        r = runs[rid]
        ux = math.cos(math.radians(r['angle']))
        uy = math.sin(math.radians(r['angle']))
        edges_t = sorted({round(t, 1) for t in positions
                          if params.endpoint_radius < t
                          < r['length'] - params.endpoint_radius})
        if not edges_t:
            continue
        ys, xs = np.nonzero(labels == rid)
        if not len(ys):
            continue
        t = (xs - r['a'][0]) * ux + (ys - r['a'][1]) * uy
        piece = np.searchsorted(np.array(edges_t), t, side='right')
        for k in range(1, piece.max() + 1):
            sel = piece == k
            if not sel.any():
                continue
            out[ys[sel], xs[sel]] = nxt
            nxt += 1
            added += 1
    counts = collections.Counter(out[out > 0].tolist())
    keep = {i for i, c in counts.items() if i and c >= 1}
    return out, _geometry(out, keep), added


def contacts_from_labels(labels, runs):
    """Every place two runs touch, on a label image built from several passes.

    extract_runs works out its own contacts while it segments, but once the
    pipes from one pass and the offcuts from another live in the same label
    image the only question left is which pairs of labels are adjacent.
    """
    H, W = labels.shape
    found = {}
    for dy, dx in SHIFTS:
        a = labels[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)]
        b = labels[max(-dy, 0):H + min(-dy, 0), max(-dx, 0):W + min(-dx, 0)]
        hit = (a > 0) & (b > 0) & (a != b)
        ys, xs = np.nonzero(hit)
        oy, ox = max(dy, 0), max(dx, 0)
        for u, v, py, px in zip(a[hit].tolist(), b[hit].tolist(),
                                (ys + oy).tolist(), (xs + ox).tolist()):
            if u in runs and v in runs:
                found.setdefault((min(u, v), max(u, v)), []).append((px, py))
    return found


def contacts_from_endpoints(runs, radius, existing=None):
    """Join runs whose ends are close but whose pixels never touch.

    Pixel adjacency is exact and that is the trouble: a signal line stopping
    3px short of the pipe it feeds is connected on the drawing and adjacent to
    nothing in the image. Every dash gap, every stroke the binariser thinned
    away at a join, and every symbol box edge leaves that kind of hole.

    Only ends are joined. Two runs that pass close by in their middles are a
    crossover, and connecting those would wire the sheet to itself.
    """
    found = {}
    cell = max(radius * 2, 4)
    grid = collections.defaultdict(list)
    for k, r in runs.items():
        for pt in (r['a'], r['b']):
            grid[(int(pt[0] // cell), int(pt[1] // cell))].append((k, pt))
    for k, r in runs.items():
        for pt in (r['a'], r['b']):
            cx, cy = int(pt[0] // cell), int(pt[1] // cell)
            for gx in (cx - 1, cx, cx + 1):
                for gy in (cy - 1, cy, cy + 1):
                    for j, qt in grid.get((gx, gy), ()):
                        if j <= k:
                            continue
                        gap = math.hypot(pt[0] - qt[0], pt[1] - qt[1])
                        if gap > radius:
                            continue
                        key = (k, j)
                        if existing and key in existing:
                            continue
                        mid = ((pt[0] + qt[0]) / 2, (pt[1] + qt[1]) / 2)
                        best = found.get(key)
                        if best is None or gap < best[0]:
                            found[key] = (gap, mid)
    return {k: [v[1]] for k, v in found.items()}


def at_end(run, px, py, radius):
    return min(math.hypot(px - run['a'][0], py - run['a'][1]),
               math.hypot(px - run['b'][0], py - run['b'][1])) <= radius


def classify_contacts(runs, contact, radius, keep_crossovers=False):
    """end-to-end, tee, or crossover -- and drop the crossovers."""
    edges, kinds = {}, collections.Counter()
    for (ra, rb), points in contact.items():
        best = None
        for px, py in points:
            ea = at_end(runs[ra], px, py, radius)
            eb = at_end(runs[rb], px, py, radius)
            rank = 2 if (ea and eb) else 1 if (ea or eb) else 0
            if best is None or rank > best[0]:
                best = (rank, px, py, ea, eb)
        rank, px, py, ea, eb = best
        kind = ('end-to-end' if rank == 2 else 'tee' if rank == 1 else 'crossover')
        kinds[kind] += 1
        if kind == 'crossover' and not keep_crossovers:
            continue
        edges[(ra, rb)] = {'at': [round(px, 1), round(py, 1)], 'kind': kind,
                           'ends': [bool(ea), bool(eb)]}
    return edges, kinds


# ---------------------------------------------------------------- 우리 추가분

def contacts_end_to_middle(runs, radius, existing=None, end_guard=None):
    """내 **끝**이 상대의 **중간**에 닿는 자리. (위 이식분은 손대지 않았다.)

    왜 필요한가
      이식분에는 두 갈래만 있다 — 픽셀이 실제로 맞닿거나(contacts_from_labels),
      두 run 의 **끝끼리** 가깝거나(contacts_from_endpoints). 그런데 내 끝이 상대의
      중간에 2~3px 떠서 닿는 T 접합은 둘 다 못 잡는다. 픽셀은 안 닿고, 끝-대-끝도
      아니기 때문이다. page_001 에서 그런 끝점이 198개였다 — 이진화가 획 한 겹을
      깎았거나 심볼 bbox 를 채우며 접점이 지워진 자리다.

    왜 안전한가
      이식분이 중간끼리의 근접을 잇지 않는 것은 옳다 — 배관 위를 **지나가기만** 하는
      리더를 분기로 오인하면 도면이 자기 자신에 얽힌다. 하지만 "내 끝 vs 상대의
      중간" 은 그 경우가 아니다. 한쪽이 여기서 **끝난다**는 것 자체가 T 접합의
      증거이고, 버려야 할 crossover 는 양쪽 다 중간인 경우다.

    끝-대-끝을 먼저 돌린 뒤에 쓴다
      이미 이어진 쌍(existing)은 건드리지 않는다. 같은 자리를 두 규칙이 잡으면
      더 직접적인 증거인 끝-대-끝이 이겨야 한다.

    end_guard: 상대의 끝에서 이만큼 안쪽이어야 '중간' 으로 본다. 기본은 radius —
      그보다 끝에 가까우면 끝-대-끝이 다룰 영역이다.
    """
    import numpy as _np
    if end_guard is None:
        end_guard = radius
    ids = sorted(runs)
    if not ids:
        return {}
    A = _np.array([runs[i]['a'] for i in ids], float)
    B = _np.array([runs[i]['b'] for i in ids], float)
    D = B - A
    LL = (D * D).sum(1)
    LL[LL == 0] = 1.0
    found = {}
    for n, k in enumerate(ids):
        for pt in (runs[k]['a'], runs[k]['b']):
            p = _np.array(pt, float)
            t = ((p - A) * D).sum(1) / LL
            t = _np.clip(t, 0.0, 1.0)
            foot = A + t[:, None] * D
            dist = _np.hypot(*(p - foot).T)
            L = _np.sqrt(LL)
            inner = _np.minimum(t, 1 - t) * L          # 상대 끝까지의 거리
            ok = (dist <= radius) & (inner > end_guard)
            ok[n] = False
            for m in _np.nonzero(ok)[0]:
                j = ids[m]
                key = (min(k, j), max(k, j))
                if existing and key in existing:
                    continue
                mid = ((p[0] + foot[m][0]) / 2, (p[1] + foot[m][1]) / 2)
                best = found.get(key)
                if best is None or dist[m] < best[0]:
                    found[key] = (float(dist[m]), mid)
    return {k: [v[1]] for k, v in found.items()}


def symbol_links(runs, symbols, touch):
    """심볼마다 그 둘레에 끝이 닿은 run. (심볼 id → [run id], 접점 좌표).

    심볼은 bbox 째로 지워져 있으므로 거기 붙던 배관은 경계에서 **끝난다**. 그래서
    끝점이 bbox 에서 touch 안쪽이면 그 심볼에 물린 배관으로 본다. run 의 중간이
    bbox 를 스치는 것은 세지 않는다 — 그것은 옆을 지나가는 배관이다.
    """
    out = []
    for s in symbols:
        x0, y0, x1, y1 = s['bbox']
        hit = []
        for rid, r in runs.items():
            for side in ('a', 'b'):
                px, py = r[side]
                dx = max(x0 - touch - px, 0, px - (x1 + touch))
                dy = max(y0 - touch - py, 0, py - (y1 + touch))
                if dx == 0 and dy == 0:
                    hit.append({'run': rid, 'end': side,
                                'at': [round(px, 1), round(py, 1)]})
                    break
        if hit:
            out.append({'symbol': s['id'], 'bbox': s['bbox'],
                        'source': s['source'], 'shape': s['shape'],
                        'status': s['status'], 'runs': hit})
    return out

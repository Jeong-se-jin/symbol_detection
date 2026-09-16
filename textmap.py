"""textmap.py — OCR 텍스트를 심볼·배관 run 에 붙인다.

`pipeline.py` 4단계까지는 글자를 **지우는 대상**으로만 다룬다. 이 모듈은 같은 글자를
**붙이는 대상**으로 다시 읽어 세 규칙을 순서대로 적용한다. 순서가 곧 우선순위다 —
앞 규칙이 가져간 텍스트는 뒤 규칙의 후보에서 빠진다.

    2  심볼 내부   기하 도형 심볼 bbox 에 **완전히** 들어간 텍스트 → 그 심볼
    3  커넥터 이웃 커넥터 bbox 의 장축 양옆(방향에 맞춰 상하 또는 좌우) → 그 커넥터
    3.5 밸브 태그  `V###` 꼴 → 가장 가까운 심볼 (밸브 판정이 없어 급한 대로)
    4  라인 번호   L+3~4자리로 끝나는 텍스트 → 5px 안의 **아직 안 묶인** run 하나
    5  리더 추적   남은 텍스트의 리더 선을 잉크로 따라가 배관 run 또는 심볼에

**텍스트 방향은 `rot` 이 아니라 bbox 장단축으로 잰다.** `rot` 은 그 글자를 잡아낸 OCR
패스(0도/90도)이지 글자가 누운 방향이 아니다 — page_001 의 라인 번호 113개 중 40개가
어긋난다 (`1" BBB L011A` 는 `rot=90도` 인데 bbox 가 143×31 가로다).
"""
import json
import os
import re

#: 라인 번호 — 끝이 L + 3~4자리(+선택 접미 문자) 로 끝난다. page_001 은 전부 3자리다.
LINE_NO = re.compile(r'\bL(\d{3,4})[A-Z]?\s*$')

#: 커넥터 이웃으로 볼 최대 간격(px). page_001 실측은 위 3~8 · 아래 0~9 다. 12 로
#: 잡았더니 N0002 가 위 12px 의 `3" EBC L034A` 를 함께 집어갔다 — 그것은 옆 배관의
#: 라인 번호이고 N0002 의 짝은 8px 의 `3108` 이다. 진짜 이웃과 남의 글자 사이가
#: 9 와 12 로 좁으므로 10 에서 끊는다.
CONN_GAP = 10

#: 라인 번호와 run 사이 최대 간격(px). 5 → 8 → 10 으로 넓혔다. 8px 의
#: `8" BTA L025B`, 10px 의 `8" BTA L007A` 처럼 후보 run 이 하나뿐인 세로 라벨이
#: 문턱 바로 밖에서 떨어져 나갔다.
LINE_GAP = 10

#: 리더 추적. `tol` 은 텍스트 bbox 에서 리더 자유 끝점까지, `snap` 은 엣지 접점이
#: run 의 그 끝에 속한다고 볼 거리다. 5/6 으로 시작했다가 15/10 으로 넓혔다 —
#: 대각선 run 은 끝점이 격자에 안 떨어져 `at` 이 7~8px 어긋나고(`8" BTA L016A`),
#: 라벨과 리더 사이가 13~14px 벌어진 것이 있다(`1" EBC L053A`). 20 까지 올리면
#: 라인 번호는 안 늘고 잡동사니만 는다.
LEADER_TOL = 15
LEADER_SNAP = 10
LEADER_HOPS = 12

#: 커넥터 heading → 텍스트가 붙는 두 방향. 장축의 양옆이다.
CONN_SIDES = {
    'LEFT': ('above', 'below'), 'RIGHT': ('above', 'below'),
    'BIDIRECTIONAL': ('above', 'below'),
    'UP': ('right', 'left'), 'DOWN': ('right', 'left'),
}

#: 기하로 판정한 도형 심볼. carrier 는 '배관이 끊긴 구간' 이라 도형이 없으므로 뺀다.
GEOM_SOURCES = ('frame', 'ring', 'connector', 'boxed')

#: 심볼을 글자로 잘못 읽은 텍스트를 걷어내는 문턱. bbox 가 심볼과 거의 그대로 겹치면
#: OCR 이 밸브 그림을 글자로 읽은 것이다 (`D` `[` `10` `ema` `-RM` …).
#: **carrier 에만 건다** — 도형 심볼(boxed·frame)은 글자를 담는 그릇이라 태그 안
#: `PMS OP CMT` 가 bbox 를 거의 채운다. 그것은 오독이 아니라 읽어야 할 글자다.
#: page_001 은 carrier 쪽이 0.86~0.69 로 12개, 그 다음이 0.37 이라 사이가 비어 있다.
MISREAD_IOU = 0.5
MISREAD_SOURCES = ('carrier',)

#: 밸브 태그 — `V080A` `V041 (T)`. V 다음에 숫자가 와야 한다 (`VESSEL` 과 홑 `V` 를
#: 가른다). **가장 가까운 심볼에 그냥 붙인다** — 밸브를 가려내는 판정이 아직 없으니
#: 급한 대로다. 이 규칙이 라인 번호보다 **앞서야** 뒤의 리더 추적이 조용해진다:
#: 밸브 태그 옆 배관이 자유 끝으로 끝나면 그 배관이 통째로 리더로 읽힌다
#: (`V081A` 388px · `V080A` 380px).
VALVE_TAG = re.compile(r'^V\s?\d')


# ---------------------------------------------------------------- 기하 도우미

def _ov(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def _orient(bbox):
    """텍스트가 누운 방향. 'H' 가로 · 'V' 세로."""
    x0, y0, x1, y1 = bbox
    return 'H' if (x1 - x0) >= (y1 - y0) else 'V'


def _inside(inner, outer):
    """inner 가 outer 안에 **완전히** 들어갔는가. 걸치는 것은 제외한다."""
    a, b, c, d = inner
    x0, y0, x1, y1 = outer
    return a >= x0 and b >= y0 and c <= x1 and d <= y1


def _gap(box, side, target):
    """box 기준 side 방향으로 target 까지의 간격. 수직 방향으로 겹치지 않으면 None."""
    a, b, c, d = box
    t0, u0, t1, u1 = target
    if side in ('above', 'below'):
        if _ov(a, c, t0, t1) <= 0:
            return None
        return (u0 - d) if side == 'below' else (b - u1)
    if _ov(b, d, u0, u1) <= 0:
        return None
    return (t0 - c) if side == 'right' else (a - t1)


def _polyline(run):
    """run 을 꼭짓점 목록으로. 1패스 run 은 양 끝점뿐이고 2패스는 points 를 가진다."""
    return run.get('points') or [run['a'], run['b']]


def _closest_on_segment(p0, p1, box):
    """선분 p0→p1 위에서 box 에 가장 가까운 점과 그 거리. box 는 [x0,y0,x1,y1].

    라벨은 배관과 **나란히** 놓이므로 최소 거리를 내는 점이 겹침 구간 전체로
    줄줄이 나온다. 그중 처음 것을 집으면 접점이 겹침의 한쪽 끝에 붙어 실제로
    어디에 나란한지 안 보인다 — 동률이면 가운데를 쓴다.
    """
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    n = max(1, int(max(abs(dx), abs(dy))))
    lo, tie = None, []
    for i in range(n + 1):
        t = i / n
        x, y = x0 + dx * t, y0 + dy * t
        gx = max(box[0] - x, 0, x - box[2])     # 점에서 사각형까지 — 안이면 0
        gy = max(box[1] - y, 0, y - box[3])
        d = (gx * gx + gy * gy) ** .5
        if lo is None or d < lo - 1e-9:
            lo, tie = d, [(x, y)]
        elif d <= lo + 1e-9:
            tie.append((x, y))
    x, y = tie[len(tie) // 2]
    return lo, [round(x, 1), round(y, 1)]


def contact(box, run):
    """텍스트 상자에서 run 으로 가는 **연결점** — run 위에서 상자에 가장 가까운 점.

    symbol_links 의 `at` 과 같은 뜻이다. bbox 중심끼리 이으면 run 이 길 때
    (2803 은 1906px 다) 연결선이 도면을 가로질러 무엇에 붙었는지 안 보인다.
    """
    pts = _polyline(run)
    best = None
    for a, b in zip(pts, pts[1:]):
        c = _closest_on_segment(a, b, box)
        if best is None or c[0] < best[0]:
            best = c
    return best[1], round(best[0], 1)


def contact_box(box, target_box):
    """텍스트 상자에서 심볼 bbox 로 가는 연결점 — 마주보는 변 위의 가장 가까운 점.
    상자가 심볼 안에 들어가 있으면(규칙 2) 텍스트 자신의 중심을 쓴다."""
    x0, y0, x1, y1 = target_box
    a, b, c, d = box
    if a >= x0 and b >= y0 and c <= x1 and d <= y1:
        return [round((a + c) / 2, 1), round((b + d) / 2, 1)], 0.0
    px = min(max((a + c) / 2, x0), x1)
    py = min(max((b + d) / 2, y0), y1)
    gx = max(a - px, 0, px - c)
    gy = max(b - py, 0, py - d)
    return [round(px, 1), round(py, 1)], round((gx * gx + gy * gy) ** .5, 1)


def _iou(a, b):
    i = _ov(a[0], a[2], b[0], b[2]) * _ov(a[1], a[3], b[1], b[3])
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    return i / max(1, ua + ub - i)


def symbol_misreads(texts, symbols, iou=MISREAD_IOU):
    """심볼을 글자로 읽은 텍스트. {text_idx: (symbol_id, iou)}. 어느 규칙에도 안 쓴다."""
    cand = [s for s in symbols if s['source'] in MISREAD_SOURCES]
    out = {}
    for i, t in enumerate(texts):
        best, who = 0.0, None
        for s in cand:
            u = _iou(t['bbox'], s['bbox'])
            if u > best:
                best, who = u, s['id']
        if best >= iou:
            out[i] = (who, round(best, 3))
    return out


def _run_box(run):
    (ax, ay), (bx, by) = run['a'], run['b']
    pts = run.get('points')
    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return [min(xs), min(ys), max(xs), max(ys)]
    return [min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)]


# ---------------------------------------------------------------- 5 리더 추적

def _dist(p, q):
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** .5


def _d2box(p, b):
    gx = max(b[0] - p[0], 0, p[0] - b[2])
    gy = max(b[1] - p[1], 0, p[1] - b[3])
    return (gx * gx + gy * gy) ** .5


def _leader_pass(texts, symbols, runs, edges, tb, used, dropped,
                 run_hit, sym_hit, where, tol, snap, hops=LEADER_HOPS):
    """리더 선을 잉크로 따라가 텍스트를 배관 run 또는 심볼에 붙인다.

    길이로 리더를 가르지 않는다. 리더는 몇 번을 꺾든 리더이고, 배관을 만나는
    자리는 거기서 길이 갈라진다 — `leader.py` 가 골격 교차수로 하던 판정
    ("꺾임은 통과하고 분기만 잡는다")을 run 그래프 위에서 그대로 한다.

        출발  글자 쪽 끝이 **아무것도 안 달린 자유 끝**인 run. 리더는 라벨에서
              홀로 출발한다 — 그 끝에 뭔가 이어져 있으면 분기 근처를 지나가는
              남의 배관이고, 옆에 놓인 글자를 붙이면 매핑이 엉뚱해진다
        진행  반대쪽 끝에 이어지는 run 이 하나뿐이면 방향이 바뀌어도 계속 간다
        정지  (a) 둘 이상이 모이는 분기  (b) 채택 심볼에 닿음
              (c) 이미 글자가 붙은 run 을 만남
    """
    R = {r['id']: r for r in runs}
    adj, at_of = {}, {}
    for e in edges or ():
        adj.setdefault(e['a'], set()).add(e['b'])
        adj.setdefault(e['b'], set()).add(e['a'])
        at_of[(e['a'], e['b'])] = at_of[(e['b'], e['a'])] = e['at']
    sym_box = [(s['id'], s['bbox']) for s in symbols if s['status'] == 'ACCEPTED']

    def ends(r):
        p = _polyline(r)
        return p[0], p[-1]

    def at_end(rid, nb, pt):
        return _dist(at_of[(rid, nb)], pt) <= snap

    def in_symbol(p, pad=3):
        for sid, (x0, y0, x1, y1) in sym_box:
            if x0-pad <= p[0] <= x1+pad and y0-pad <= p[1] <= y1+pad:
                return sid
        return None

    def trace(i):
        box = tb[i]
        seeds = []
        for r in runs:
            if r['id'] in run_hit:
                continue                    # 글자가 붙은 run = 배관이지 리더가 아니다
            e0, e1 = ends(r)
            d0, d1 = _d2box(e0, box), _d2box(e1, box)
            if min(d0, d1) > tol:
                continue
            entry = e0 if d0 <= d1 else e1
            if any(at_end(r['id'], nb, entry) for nb in adj.get(r['id'], ())):
                continue                    # 자유 끝이 아니다
            seeds.append((min(d0, d1), r['id'], entry))
        seeds.sort(key=lambda s: (s[0], -R[s[1]]['length']))
        for _, sid, entry in seeds:
            cur, came, path, seen = sid, entry, [sid], {sid}
            for _ in range(hops):
                e0, e1 = ends(R[cur])
                far = e1 if _dist(e0, came) <= _dist(e1, came) else e0
                hit = in_symbol(far)
                if hit:
                    return ('symbol', hit, path, far)
                nbs = [nb for nb in adj.get(cur, ())
                       if nb not in seen and at_end(cur, nb, far)]
                if not nbs:
                    break
                if len(nbs) == 1:
                    nxt = nbs[0]
                    if nxt in run_hit:
                        return ('run', nxt, path, at_of[(cur, nxt)])
                    path.append(nxt); seen.add(nxt)
                    came = at_of[(cur, nxt)]; cur = nxt
                    continue
                tgt = max(nbs, key=lambda n: R[n]['length'])
                return ('run', tgt, path, at_of[(cur, tgt)])
        return None

    found = {}
    for i in range(len(texts)):
        if i in used or i in dropped:
            continue
        got = trace(i)
        if not got:
            continue
        kind, tgt, path, at = got
        (sym_hit if kind == 'symbol' else run_hit).setdefault(tgt, []).append(i)
        used[i] = 'leader'
        where[i] = {'side': 'leader',
                    'gap': round(sum(R[p]['length'] for p in path), 1),
                    'at': [round(float(at[0]), 1), round(float(at[1]), 1)],
                    'leader_runs': path, 'segments': len(path)}
        found[i] = kind
    return found


# ---------------------------------------------------------------- 매핑

def assign(texts, symbols, runs, edges=(),
           conn_gap=CONN_GAP, line_gap=LINE_GAP,
           leader_tol=LEADER_TOL, leader_snap=LEADER_SNAP, accepted_only=True):
    """(links, stats). texts/symbols/runs 는 읽기만 하고 고치지 않는다.

    links = {'symbols': {sym_id: [text_idx…]}, 'runs': {run_id: [text_idx…]},
             'rule': {text_idx: 'inside'|'connector'|'line_no'}}
    """
    tb = [t['bbox'] for t in texts]
    dropped = symbol_misreads(texts, symbols)
    used = {}                                    # text_idx → 규칙 이름
    sym_hit = {}                                 # sym_id → [text_idx]
    run_hit = {}                                 # run_id → [text_idx]
    where = {}                                   # text_idx → {side, gap, at}

    S_BOX = {s['id']: s['bbox'] for s in symbols}
    geom = [s for s in symbols
            if s['source'] in GEOM_SOURCES
            and (s['status'] == 'ACCEPTED' or not accepted_only)]

    # --- 2 심볼 내부 -----------------------------------------------------
    # 겹쳐 있는 심볼(프레임 안의 프레임 등)이면 **가장 작은** 것에 붙인다.
    # 큰 쪽에 붙이면 바깥 상자가 안쪽 코드까지 다 먹는다.
    for i, box in enumerate(tb):
        if i in dropped:
            continue
        owner, area = None, None
        for s in geom:
            if not _inside(box, s['bbox']):
                continue
            x0, y0, x1, y1 = s['bbox']
            a = (x1 - x0) * (y1 - y0)
            if area is None or a < area:
                owner, area = s['id'], a
        if owner:
            sym_hit.setdefault(owner, []).append(i)
            used[i] = 'inside'
            at, d = contact_box(box, S_BOX[owner])
            where[i] = {'side': 'inside', 'gap': d, 'at': at}

    # --- 3 커넥터 이웃 ---------------------------------------------------
    # 오프페이지 커넥터는 장축 양옆에 두 줄(위=스트림 번호, 아래=라인 스펙)을 달고
    # 있다. 어느 쪽이 '양옆' 인가는 heading 이 말해 준다 — 세로 커넥터(UP)는 좌우다.
    for s in geom:
        if s['source'] != 'connector':
            continue
        sides = CONN_SIDES.get(s['geometry'].get('heading'), ('above', 'below'))
        for side in sides:
            for i, box in enumerate(tb):
                if i in used or i in dropped:
                    continue
                g = _gap(s['bbox'], side, box)
                if g is None or not (-1 <= g <= conn_gap):
                    continue
                sym_hit.setdefault(s['id'], []).append(i)
                used[i] = 'connector'
                at, _ = contact_box(box, s['bbox'])
                where[i] = {'side': side, 'gap': round(float(g), 1), 'at': at}

    # --- 3.5 밸브 태그 → 가장 가까운 심볼 --------------------------------
    acc = [s for s in symbols if s['status'] == 'ACCEPTED']
    for i, box in enumerate(tb):
        if i in used or i in dropped or not VALVE_TAG.match(texts[i]['text'].strip()):
            continue
        best = None
        for s in acc:
            at, d = contact_box(box, s['bbox'])
            if best is None or d < best[0]:
                best = (d, s['id'], at)
        if best is None:
            continue
        d, sid, at = best
        sym_hit.setdefault(sid, []).append(i)
        used[i] = 'valve_tag'
        where[i] = {'side': 'nearest', 'gap': d, 'at': at}

    # --- 4 라인 번호 → run ----------------------------------------------
    # 1차는 아래(가로 글자) / 오른쪽(세로 글자) 을 전부 돌고, **그 다음에** 남은
    # 것만 반대쪽을 본다. 한 번에 양쪽을 보면 아래에 제 짝이 있는 글자가 위쪽
    # 남의 run 을 먼저 집어간다.
    cand = [i for i, t in enumerate(texts)
            if i not in used and i not in dropped and LINE_NO.search(t['text'])]
    boxes = {r['id']: _run_box(r) for r in runs}
    RUN = {r['id']: r for r in runs}
    taken = set()                                # 이미 글자가 붙은 run

    def try_side(idx, side):
        best = None
        for r in runs:
            rid = r['id']
            if rid in taken or rid in run_hit:
                continue
            g = _gap(tb[idx], side, boxes[rid])
            if g is None or not (-1 <= g <= line_gap):
                continue
            rb = boxes[rid]
            b = tb[idx]
            share = (_ov(b[0], b[2], rb[0], rb[2]) if side in ('above', 'below')
                     else _ov(b[1], b[3], rb[1], rb[3]))
            key = (g, -share, rid)               # 가까운 것 → 많이 겹치는 것
            if best is None or key < best[0]:
                best = (key, rid)
        if best is None:
            return False
        rid = best[1]
        run_hit[rid] = [idx]
        taken.add(rid)
        used[idx] = 'line_no'
        at, d = contact(tb[idx], RUN[rid])
        where[idx] = {'side': side, 'gap': d, 'at': at}
        return True

    first, second = {}, {}
    for idx in cand:
        s1, s2 = (('below', 'above') if _orient(tb[idx]) == 'H'
                  else ('right', 'left'))
        first[idx], second[idx] = s1, s2
    pass1 = [i for i in cand if try_side(i, first[i])]
    pass2 = [i for i in cand if i not in used and try_side(i, second[i])]

    # --- 5 리더 추적 -----------------------------------------------------
    # 남은 텍스트 중 **리더 선을 달고 있는 것**을 잉크로 따라간다. 앞 네 규칙이
    # 근접으로 붙였다면 이쪽은 실제로 이어진 선을 밟는다.
    lead = _leader_pass(texts, symbols, runs, edges, tb, used, dropped,
                        run_hit, sym_hit, where, leader_tol, leader_snap)

    stats = {
        'texts': len(texts),
        'dropped_symbol_misread': len(dropped),
        'inside': sum(1 for v in used.values() if v == 'inside'),
        'connector': sum(1 for v in used.values() if v == 'connector'),
        'valve_tag': sum(1 for v in used.values() if v == 'valve_tag'),
        'line_no_candidates': len(cand),
        'line_no_pass1': len(pass1),
        'line_no_pass2': len(pass2),
        'line_no_unmatched': len(cand) - len(pass1) - len(pass2),
        'leader': len(lead),
        'leader_to_run': sum(1 for v in lead.values() if v == 'run'),
        'leader_to_symbol': sum(1 for v in lead.values() if v == 'symbol'),
        'symbols_mapped': len(sym_hit),
        'runs_mapped': len(run_hit),
        'unmapped_texts': len(texts) - len(used) - len(dropped),
    }
    return {'symbols': sym_hit, 'runs': run_hit, 'rule': used,
            'where': where, 'dropped': dropped}, stats


# ---------------------------------------------------------------- 출력

def text_ids(texts):
    """T0001… — 좌상단부터 읽기 순서. OCR 출력 순서는 타일 순서라 재실행에
    흔들리므로 좌표로 못 박는다."""
    order = sorted(range(len(texts)),
                   key=lambda i: (texts[i]['bbox'][1], texts[i]['bbox'][0]))
    ids = [None] * len(texts)
    for n, i in enumerate(order, 1):
        ids[i] = 'T%04d' % n
    return ids


def apply_to(texts, symbols, runs, links, ids):
    """세 파일의 행을 제자리에서 고친다. 여러 개면 쉼표로 잇는다."""
    # 재실행에 대비해 지난 매핑을 먼저 지운다 — 남겨 두면 이번에 짝을 잃은 심볼이
    # 옛 tag_text 를 그대로 달고 있어 문턱을 좁힌 일이 결과에 안 보인다.
    for s in symbols:
        s['geometry'].pop('tag_text_ids', None)
        s.pop('text_links', None)
        if 'tag_text' in s['geometry']:
            s['geometry']['tag_text'] = None
    for r in runs:
        r.pop('text', None)
        r.pop('text_ids', None)
        r.pop('text_links', None)

    for i, t in enumerate(texts):
        w = links['where'].get(i) or {}
        t['id'] = ids[i]
        t['link'] = None
        t['link_rule'] = links['rule'].get(i)
        d = links['dropped'].get(i)
        t['dropped'] = 'SYMBOL_MISREAD' if d else None
        t['dropped_by'] = d[0] if d else None
        t['dropped_iou'] = d[1] if d else None
        t['link_side'] = w.get('side')
        t['link_gap'] = w.get('gap')
        t['link_at'] = w.get('at')
        t['link_leader_runs'] = w.get('leader_runs')
    for s in symbols:
        idxs = sorted(links['symbols'].get(s['id'], []),
                      key=lambda i: (texts[i]['bbox'][1], texts[i]['bbox'][0]))
        if not idxs:
            continue
        s['geometry']['tag_text'] = ', '.join(texts[i]['text'].strip() for i in idxs)
        s['geometry']['tag_text_ids'] = ', '.join(ids[i] for i in idxs)
        s['text_links'] = [{'text': ids[i], **links['where'][i]} for i in idxs]
        for i in idxs:
            texts[i]['link'] = s['id']
    for r in runs:
        idxs = links['runs'].get(r['id'], [])
        if not idxs:
            continue
        r['text'] = ', '.join(texts[i]['text'].strip() for i in idxs)
        r['text_ids'] = ', '.join(ids[i] for i in idxs)
        # 접점은 symbol_links 와 같은 모양으로 남긴다 — 어느 쪽에서 몇 px 떨어진
        # 어디에 붙었는가. 나중에 되짚을 때 필요한 것은 그것뿐이다.
        r['text_links'] = [{'text': ids[i], **links['where'][i]} for i in idxs]
        for i in idxs:
            texts[i]['link'] = r['id']


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('out_dir', help='pipeline.py 의 출력 디렉터리')
    ap.add_argument('--conn-gap', type=int, default=CONN_GAP)
    ap.add_argument('--line-gap', type=int, default=LINE_GAP)
    ap.add_argument('--leader-tol', type=int, default=LEADER_TOL)
    ap.add_argument('--leader-snap', type=int, default=LEADER_SNAP)
    ap.add_argument('--dry-run', action='store_true', help='파일을 고치지 않는다')
    a = ap.parse_args()

    def load(name):
        with open(os.path.join(a.out_dir, name), encoding='utf-8') as fp:
            return json.load(fp)

    def save(name, obj):
        if a.dry_run:
            return
        with open(os.path.join(a.out_dir, name), 'w', encoding='utf-8') as fp:
            json.dump(obj, fp, ensure_ascii=False, indent=1)

    tj, sj, lj = load('texts.json'), load('symbols.json'), load('lines.json')
    texts, symbols, runs = tj['texts'], sj['symbols'], lj['runs']

    links, stats = assign(texts, symbols, runs, lj.get('edges', ()),
                          a.conn_gap, a.line_gap, a.leader_tol, a.leader_snap)
    ids = text_ids(texts)
    apply_to(texts, symbols, runs, links, ids)

    tj['text_map'] = {'params': {'conn_gap': a.conn_gap, 'line_gap': a.line_gap,
                                 'leader_tol': a.leader_tol, 'leader_snap': a.leader_snap},
                      'stats': stats}
    save('texts.json', tj)
    save('symbols.json', sj)
    save('lines.json', lj)
    kind = {}
    for k, v in links['symbols'].items():
        for i in v:
            kind[i] = ('symbol', k)
    for k, v in links['runs'].items():
        for i in v:
            kind[i] = ('run', k)
    save('text_map.json', {
        'params': tj['text_map']['params'], 'stats': stats,
        'links': [{'text': ids[i], 'raw': texts[i]['text'].strip(),
                   'bbox': texts[i]['bbox'], 'rule': links['rule'][i],
                   'target_kind': kind[i][0], 'target': kind[i][1],
                   **links['where'][i]}
                  for i in sorted(links['rule'],
                                  key=lambda i: (texts[i]['bbox'][1], texts[i]['bbox'][0]))],
        'dropped': [{'text': ids[i], 'raw': texts[i]['text'].strip(),
                     'bbox': texts[i]['bbox'], 'symbol': sid, 'iou': u}
                    for i, (sid, u) in sorted(links['dropped'].items(),
                                              key=lambda kv: -kv[1][1])],
        'symbols': {k: [ids[i] for i in v] for k, v in sorted(links['symbols'].items())},
        'runs': {str(k): [ids[i] for i in v] for k, v in sorted(links['runs'].items())},
    })

    w = max(len(k) for k in stats)
    for k, v in stats.items():
        print(f'  {k:<{w}}  {v}')


if __name__ == '__main__':
    main()

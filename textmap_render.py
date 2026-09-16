"""textmap_render.py — textmap.py 의 결과를 그림으로 낸다.

    python textmap_render.py <out_dir> [texts|map|ids|all]

    texts  01_texts.png          심볼 오독으로 걷어낸 텍스트
    map    10_text_map.png       규칙별 텍스트 매핑
    ids    09_component_id.png   심볼 id · run id 지도
"""
import json
import os
import sys

import cv2
import numpy as np


def _texts(OUT):
    tj = json.load(open(os.path.join(OUT, 'texts.json'), encoding='utf-8'))
    S = json.load(open(os.path.join(OUT, 'symbols.json'), encoding='utf-8'))['symbols']
    page = cv2.imread(os.path.join(OUT, '00_page.png'), 0)
    im = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)

    RAW, KEEP, DROP, SYM = (190, 190, 190), (0, 150, 0), (0, 0, 230), (255, 0, 255)
    for t in tj.get('texts_raw', []):
        x0, y0, x1, y1 = t['bbox']
        cv2.rectangle(im, (x0, y0), (x1, y1), RAW, 1)
    drop = [t for t in tj['texts'] if t.get('dropped')]
    for t in tj['texts']:
        x0, y0, x1, y1 = t['bbox']
        if t.get('dropped'):
            continue
        cv2.rectangle(im, (x0, y0), (x1, y1), KEEP, 2)
    # 걷어낸 것은 맨 위에 — 짝이 된 심볼 bbox 도 함께 그린다
    for t in drop:
        x0, y0, x1, y1 = t['bbox']
        sym = next((s for s in S if s['id'] == t['dropped_by']), None)
        if sym:
            a, b, c, d = sym['bbox']
            cv2.rectangle(im, (a, b), (c, d), SYM, 2)
        cv2.rectangle(im, (x0, y0), (x1, y1), DROP, 3)
        cv2.line(im, (x0, y0), (x1, y1), DROP, 2)
        cv2.line(im, (x0, y1), (x1, y0), DROP, 2)
        cv2.putText(im, f"{t['id']} {t['text'].strip()[:12]} x{t['dropped_by']} IoU{t['dropped_iou']:.2f}",
                    (x0 - 4, y0 - 8), cv2.FONT_HERSHEY_PLAIN, 1.4, DROP, 2, cv2.LINE_AA)

    leg = [(f'texts_raw {len(tj.get("texts_raw", []))}', RAW),
           (f'남긴 텍스트 {len(tj["texts"])-len(drop)}', KEEP),
           (f'심볼 오독으로 제거 {len(drop)}', DROP),
           ('짝이 된 심볼 bbox', SYM)]
    cv2.rectangle(im, (0,0), (400, 26*len(leg)+18), (255,255,255), -1)
    cv2.rectangle(im, (0,0), (400, 26*len(leg)+18), (0,0,0), 1)
    for i,(t_,c) in enumerate(leg):
        y=26*i+26; cv2.rectangle(im,(10,y-10),(30,y+4),c,-1)
        cv2.putText(im,t_,(40,y+2),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1,cv2.LINE_AA)
    p = os.path.join(OUT, '01_texts.png')
    cv2.imwrite(p, im, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    print(p, '| 제거', len(drop))
    for t in drop:
        print(f"   {t['id']} {t['text'].strip()!r:<12} {t['bbox']}  ↔ {t['dropped_by']} IoU {t['dropped_iou']}")


def _map(OUT):
    FADE = 0.16
    C = {'inside': (0, 160, 0), 'connector': (0, 140, 255), 'valve_tag': (200, 0, 200),
         'line_no': (220, 60, 0), 'leader': (0, 200, 120), None: (200, 200, 200)}
    SYMC = (255, 0, 255)
    AT = (0, 0, 220)                 # 연결점(at) 은 규칙 색과 따로 둔다
    FONT = cv2.FONT_HERSHEY_PLAIN

    page = cv2.imread(os.path.join(OUT, '00_page.png'), 0)
    H, W = page.shape
    im = cv2.cvtColor((255 - (255 - page).astype(np.float32) * FADE).astype(np.uint8),
                      cv2.COLOR_GRAY2BGR)

    texts = json.load(open(os.path.join(OUT, 'texts.json'), encoding='utf-8'))['texts']
    L = json.load(open(os.path.join(OUT, 'lines.json'), encoding='utf-8'))
    syms = json.load(open(os.path.join(OUT, 'symbols.json'), encoding='utf-8'))['symbols']
    S = {s['id']: s for s in syms}
    R = {r['id']: r for r in L['runs']}

    def ctr(b): return (int((b[0]+b[2])/2), int((b[1]+b[3])/2))
    def runmid(r):
        p = r.get('points')
        if p: return tuple(map(int, p[len(p)//2]))
        return (int((r['a'][0]+r['b'][0])/2), int((r['a'][1]+r['b'][1])/2))

    # 매핑 안 된 run 은 옅게, 글자가 붙은 run 은 굵게
    for r in L['runs']:
        a = (int(r['a'][0]), int(r['a'][1])); b = (int(r['b'][0]), int(r['b'][1]))
        p = r.get('points')
        if r.get('text'):
            col, th = C['line_no'], 3
        else:
            col, th = (205, 205, 205), 1
        if p: cv2.polylines(im, [np.array(p, np.int32)], False, col, th, cv2.LINE_AA)
        else: cv2.line(im, a, b, col, th, cv2.LINE_AA)

    # 심볼: 글자가 붙었으면 마젠타, 아니면 옅은 회색
    for s in syms:
        x0, y0, x1, y1 = s['bbox']
        got = bool(s['geometry'].get('tag_text'))
        cv2.rectangle(im, (x0, y0), (x1-1, y1-1), SYMC if got else (205, 205, 205), 2 if got else 1)

    # 텍스트 상자 + 연결선
    for t in texts:
        rule = t.get('link_rule')
        col = C[rule]
        x0, y0, x1, y1 = t['bbox']
        cv2.rectangle(im, (x0, y0), (x1, y1), col, 1)
        # 리더는 밟은 조각을 굵게 덧그린다 — 어느 잉크를 따라갔는지가 곧 근거다
        if rule == 'leader':
            for rid in (t.get('link_leader_runs') or []):
                rr = R.get(rid)
                if rr:
                    cv2.line(im, (int(rr['a'][0]), int(rr['a'][1])),
                             (int(rr['b'][0]), int(rr['b'][1])), col, 4, cv2.LINE_AA)
        at = t.get('link_at')
        # 접점(at) 표식은 **리더에만** 그린다. 나머지 규칙의 at 은 "가장 가까운 점" 을
        # 계산한 것일 뿐 잉크가 만나는 자리가 아니라, 찍어 두면 근거가 있는 것처럼 보인다.
        if at and rule == 'leader':
            dst = (int(at[0]), int(at[1]))
            cv2.line(im, ctr(t['bbox']), dst, col, 2, cv2.LINE_AA)
            cv2.circle(im, dst, 13, (255, 255, 255), -1)
            cv2.circle(im, dst, 13, AT, 3)
            cv2.circle(im, dst, 4, AT, -1)
        elif rule in ('connector', 'line_no', 'valve_tag'):
            cv2.line(im, ctr(t['bbox']), (int(at[0]), int(at[1])), col, 1, cv2.LINE_AA)
        if rule:
            cv2.putText(im, t['id'], (x0, y0 - 2), FONT, 0.55, col, 1, cv2.LINE_AA)

    rows = [('inside  → 심볼 내부', C['inside']), ('connector → 커넥터 이웃', C['connector']),
            ('valve_tag → 가장 가까운 심볼', C['valve_tag']),
            ('line_no → 나란한 run', C['line_no']),
            ('leader → 리더 따라간 run/심볼', C['leader']),
            ('매핑 없음', C[None]),
            ('리더 접점 at (리더만 표시)', AT),
            ('심볼(글자 붙음)', SYMC), ('run(글자 없음)', (205, 205, 205))]
    bh = 26*len(rows)+18
    cv2.rectangle(im, (0,0), (330,bh), (255,255,255), -1)
    cv2.rectangle(im, (0,0), (330,bh), (0,0,0), 1)
    for i,(t,c) in enumerate(rows):
        y=26*i+26
        cv2.rectangle(im,(10,y-10),(30,y+4),c,-1)
        cv2.putText(im,t,(40,y+2),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1,cv2.LINE_AA)

    p = os.path.join(OUT, '10_text_map.png')
    cv2.imwrite(p, im, [cv2.IMWRITE_PNG_COMPRESSION, 3]); print(p)


def _ids(OUT):
    FADE = 0.18                     # 배경 도면 농도
    CELL = 16
    SYM = {'carrier': (255, 0, 255), 'frame': (0, 170, 0), 'ring': (220, 0, 0),
           'connector': (0, 140, 255), 'boxed': (200, 160, 0)}
    REJ = (185, 185, 185)
    RUN_SOLID, RUN_DASH = (210, 110, 0), (150, 60, 200)
    FONT = cv2.FONT_HERSHEY_PLAIN   # PLAIN 이 작은 배율에서 SIMPLEX 보다 덜 뭉갠다
    FS, FT = 0.55, 1                # 글자 높이 약 6px

    page = cv2.imread(os.path.join(OUT, '00_page.png'), 0)
    H, W = page.shape
    im = (255 - (255 - page).astype(np.float32) * FADE).astype(np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    lines = json.load(open(os.path.join(OUT, 'lines.json'), encoding='utf-8'))
    syms = json.load(open(os.path.join(OUT, 'symbols.json'), encoding='utf-8'))['symbols']

    # 글자를 놓을 자리를 성긴 격자로 예약한다 — 6600×4859 에 1300개를 찍으면
    # 라벨끼리 겹쳐 둘 다 못 읽는다. 자리를 못 잡으면 글자는 접고 도형만 남긴다.
    CELL = 16
    taken = np.zeros((H // CELL + 2, W // CELL + 2), bool)


    def put(x, y, text, color):
        (tw, th), _ = cv2.getTextSize(text, FONT, FS, FT)
        cand = [(4, -3), (4, th + 4), (-tw - 4, -3), (-tw - 4, th + 4),
                (4, th // 2), (-tw - 4, th // 2),
                (-tw // 2, -6), (-tw // 2, th + 7),          # 위·아래 중앙
                (10, -3 - th - 3), (10, th + 4 + th + 3)]    # 한 줄 더 밖으로
        for dx, dy in cand:
            x0, y0 = x + dx, y + dy - th
            if not (0 <= x0 and x0 + tw < W and 0 <= y0 and y0 + th < H):
                continue
            c0, c1 = x0 // CELL, (x0 + tw) // CELL + 1
            r0, r1 = y0 // CELL, (y0 + th) // CELL + 1
            if taken[r0:r1, c0:c1].any():
                continue
            taken[r0:r1, c0:c1] = True
            cv2.putText(im, text, (x0, y0 + th), FONT, FS, color, FT, cv2.LINE_AA)
            return True
        return False

    # --- 심볼 : bbox + id ----------------------------------------------------
    drawn_sym = 0
    for s in syms:
        x0, y0, x1, y1 = s['bbox']
        col = SYM[s['source']] if s['status'] == 'ACCEPTED' else REJ
        cv2.rectangle(im, (x0, y0), (x1 - 1, y1 - 1), col, 2)
        drawn_sym += put(x1, y0, s['id'], col)

    # --- 배관 run : 경로 + id ------------------------------------------------
    drawn_run = 0
    for r in lines['runs']:
        col = RUN_DASH if r.get('line_type') == 'dashed' else RUN_SOLID
        pts = r.get('points')
        if pts:
            cv2.polylines(im, [np.array(pts, np.int32)], False, col, 1, cv2.LINE_AA)
            mx, my = pts[len(pts) // 2]
        else:
            ax, ay = r['a']; bx, by = r['b']
            cv2.line(im, (int(ax), int(ay)), (int(bx), int(by)), col, 1, cv2.LINE_AA)
            mx, my = (ax + bx) / 2, (ay + by) / 2
        drawn_run += put(int(mx), int(my), str(r['id']), col)

    # --- 엣지 접점 -----------------------------------------------------------
    for e in lines['edges']:
        x, y = e['at']
        cv2.circle(im, (int(x), int(y)), 2, (0, 0, 255) if e['kind'] == 'tee' else (120, 120, 120), -1)

    # --- 심볼-run 링크 -------------------------------------------------------
    for sl in lines['symbol_links']:
        for rr in sl['runs']:
            x, y = rr['at']
            cv2.circle(im, (int(x), int(y)), 3, (0, 200, 255), 1, cv2.LINE_AA)

    # --- 범례 ---------------------------------------------------------------
    rows = ([(f'symbol {k}', SYM[k]) for k in SYM] +
            [('symbol REJECTED', REJ), ('run solid', RUN_SOLID), ('run dashed', RUN_DASH),
             ('tee', (0, 0, 255)), ('symbol-run link', (0, 200, 255))])
    bw, bh = 300, 26 * len(rows) + 18
    cv2.rectangle(im, (0, 0), (bw, bh), (255, 255, 255), -1)
    cv2.rectangle(im, (0, 0), (bw, bh), (0, 0, 0), 1)
    for i, (t, c) in enumerate(rows):
        y = 26 * i + 26
        cv2.rectangle(im, (10, y - 10), (30, y + 4), c, -1)
        cv2.putText(im, t, (40, y + 2), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1, cv2.LINE_AA)

    p = os.path.join(OUT, '09_component_id.png')
    cv2.imwrite(p, im, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    print(f'{p}')
    print(f'  run  {len(lines["runs"])}개 중 id 표기 {drawn_run}  (자리 없어 접음 {len(lines["runs"])-drawn_run})')
    print(f'  심볼 {len(syms)}개 중 id 표기 {drawn_sym}  (접음 {len(syms)-drawn_sym})')
    print(f'  엣지 {len(lines["edges"])}  심볼-run 링크 {sum(len(x["runs"]) for x in lines["symbol_links"])}')


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'out/full'
    what = sys.argv[2] if len(sys.argv) > 2 else 'all'
    for name, fn in (('texts', _texts), ('map', _map), ('ids', _ids)):
        if what in ('all', name):
            fn(out)

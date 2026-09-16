"""심볼 검출 → 선 검출까지 4단계로 잇는다.

  python pipeline.py <img|pdf> <out_dir> [--ocr-python ...] [--texts ...]

  1 텍스트   PaddleOCR 로 글자를 읽는다 (ocr_paddle.py, 다른 인터프리터)
  2 타원     라인 클래스 경계 태그와 거기 매달린 리더를 지운다 (corner.py · leader.py)
  3 심볼     지운 도면에서 심볼을 검출한다 (detect.py)
  4 선       글자와 심볼을 지운 뒤 배관 run 을 뽑는다 (textink.py · lines.py)

순서가 곧 설계다
  각 단계는 다음 단계의 입력을 치우려고 존재한다. 2 를 건너뛰면 리더가 배관에 붙은
  가지로 남고, 3 을 건너뛰면 심볼 몸통의 획이 전부 배관 run 이 되며, 4 의 글자
  지우기를 건너뛰면 글자 획이 run 으로 샌다. 옮겨 온 쪽 실측으로는 심볼을 안 지우면
  2458 run 중 929 개가 심볼 몸통이었다.

글자는 bbox 가 아니라 획을 지운다
  라벨은 배관 **위에** 앉으므로 OCR 상자를 칠하면 그 아래 배관이 같이 사라진다.
  textink 가 연결 성분 단위로 글자만 골라낸다.

  그 전에 상자를 글줄 방향으로 늘린다 (`--box-grow`). 타일보다 긴 글줄은 어느
  타일에도 온전히 들어가지 못해 상자가 타일 경계에서 잘리고, 잘려 나간 앞글자가
  지워지지 않고 배관 run 으로 샌다 — NOTES 문단이 그렇다.

심볼은 bbox 를 채운다
  우리 검출기의 bbox 는 심볼에 딱 맞으므로 (템플릿·DETR 상자와 달리) 안쪽으로
  물릴 이유가 없다. 기본 `--symbol-inset 0` 이다.

  테두리만 지우는 것은 **속에 무언가 있을 때만** 이다. 옮겨 온 쪽은 "짧은 변이
  big_shape 보다 크면 장비" 라는 크기 규칙을 쓰는데, 우리 도면에서는 그 규칙이
  오프페이지 커넥터 31개를 전부 잡아먹었다 — 커넥터는 331×61 로 길지만 속에는
  글자뿐이라 채워야 할 것이다. 그래서 크기가 아니라 **실제로 품고 있는 것**으로
  가른다: 안쪽에 다른 채택 심볼이 있거나 글자를 지우고도 잉크가 남으면 테두리만,
  아니면 채운다. page_001 의 TANK 4 · VESSEL 1 은 내부 잉크가 0.2~0.7% 라
  전부 채우는 쪽으로 떨어진다.

두 스케일
  ref_h  OCR 글자 높이. 텍스트 관련 임계가 여기 붙는다.
  draw_h 도면 자체의 스케일. 기하 임계가 여기 붙는다. 계기 버블의 지름에서
         뽑는다 (버블은 어느 P&ID 에서나 같은 크기로 그려지므로 배율을 그대로
         말해 준다). 버블이 없으면 stroke width × 5.6 으로 물러선다.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import carrier  # noqa: E402
import dashes as dash_stage  # noqa: E402
import detect  # noqa: E402
import leader  # noqa: E402
import metrics  # noqa: E402
import ocrcache  # noqa: E402
import lines as line_stage  # noqa: E402
import render  # noqa: E402
import residual as residual_stage  # noqa: E402
import textink  # noqa: E402
from lineparams import Params as LineParams  # noqa: E402

#: paddle 이 있는 인터프리터. 기계마다 다르므로 환경변수로 덮을 수 있다.
#: 우선순위는 --ocr-python > $SYMBOL_DETECTION_OCR_PYTHON > 아래 기본값.
DETECT_VENV = os.environ.get('SYMBOL_DETECTION_OCR_PYTHON',
                             '/home/rx/project/STA-main/.venv-detect/bin/python')

#: 계기 버블의 지름 대비 draw_h. 옮겨 온 쪽이 네 시트에서 맞춘 값이다.
BUBBLE_TO_DRAW_H = 0.215
#: 버블이 모자랄 때 물러설 곳. stroke width 배수다.
DRAW_H_PER_STROKE = 5.6
MIN_BUBBLES = 5


def step(n, msg):
    print(f'[{n}] {msg}', flush=True)


# ---------------------------------------------------------------- 1 텍스트

def stage_text(page_png, out_dir, ocr_python, texts_path, force, ocr_args):
    """texts.json 을 얻는다. 캐시가 있으면 OCR 을 아예 띄우지 않는다.

    캐시 키는 렌더된 페이지의 내용 해시 + OCR 설정이다 (ocrcache 참조). 여기서
    맞춰 보고 넘어가면 paddle 프로세스를 띄우지 않으므로 모델 적재 20초도 든다.
    """
    path = texts_path or os.path.join(out_dir, 'texts.json')
    if texts_path:
        d = json.load(open(path, encoding='utf-8'))
        step(1, f"텍스트: {len(d['texts'])}개 (--texts {path})")
        return d
    ck = ocrcache.key(page_png, ocr_args['tile'], ocr_args['overlap'],
                      f"{ocr_args['scale']}|rot={not ocr_args['no_rotate']}"
                      f"|det={ocr_args['det_thresh']},{ocr_args['det_box_thresh']},"
                      f"{ocr_args['det_unclip']}")
    if not force:
        hit = ocrcache.load(ck)
        if hit is not None:
            with open(path, 'w', encoding='utf-8') as fp:
                json.dump(hit, fp, ensure_ascii=False, indent=1)
            step(1, f"텍스트: {len(hit['texts'])}개 "
                    f"(캐시 {ck[:12]}… 적중, ref_h={hit['ref_h']:.0f})")
            return hit
    if not os.path.exists(ocr_python):
        raise SystemExit(f'ERROR: paddle 인터프리터가 없다: {ocr_python}\n'
                         f'       --texts 로 만들어 둔 texts.json 을 주거나 '
                         f'--ocr-python 으로 경로를 지정한다.')
    cmd = [ocr_python, str(Path(__file__).resolve().parent / 'ocr_paddle.py'),
           str(page_png), out_dir,
           '--tile', str(ocr_args['tile']), '--overlap', str(ocr_args['overlap']),
           '--scale', str(ocr_args['scale'])]
    if force:
        cmd.append('--no-cache')
    if ocr_args['no_rotate']:
        cmd.append('--no-rotate')
    for flag, key in (('--det-thresh', 'det_thresh'),
                      ('--det-box-thresh', 'det_box_thresh'),
                      ('--det-unclip', 'det_unclip')):
        if ocr_args[key] is not None:
            cmd += [flag, str(ocr_args[key])]
    step(1, f'텍스트: OCR 실행 — {" ".join(cmd)}')
    subprocess.run(cmd, check=True)
    d = json.load(open(path, encoding='utf-8'))
    step(1, f"텍스트: {len(d['texts'])}개  ref_h={d['ref_h']:.0f}px")
    return d


# ---------------------------------------------------------------- 2 타원 태그

def stage_tags(gray, out_dir, params):
    """타원 경계 태그와 리더를 지운 그림. (지운 그림, 태그 행 목록)."""
    rows, stats, _ = detect.run(gray, params, boxed_corner='stadium')
    erased = gray.copy()
    tags = [r for r in rows
            if r['status'] == 'ACCEPTED' and r.get('_leaders') is not None]
    for r in tags:
        leader.erase(erased, r['bbox'], r['_leaders'])
    lens = [r['geometry'].get('leader_length', 0) for r in tags]
    step(2, f'타원 태그: {len(tags)}개 + 리더 지움 '
            f'(리더 길이 {min(lens) if lens else 0}~{max(lens) if lens else 0}px)')
    return erased, [{k: v for k, v in r.items() if not k.startswith('_')} for r in tags]


# ---------------------------------------------------------------- 3 심볼

def stage_symbols(gray, out_dir, params, use_rejected=True):
    """심볼 행. 4단계에서 지울 대상은 기각된 것까지 포함한다.

    기각은 "심볼이 아니다" 가 아니라 "이 임계를 못 넘었다" 는 뜻이다. page_001 의
    기각 54개는 하나도 채택 심볼과 겹치지 않는다 — 전부 설명되지 않은 잉크이고,
    지우지 않으면 그대로 배관 run 으로 샌다.
    """
    rows, stats, _ = detect.run(gray, params)
    acc = [r for r in rows if r['status'] == 'ACCEPTED']
    erase = rows if use_rejected else acc
    step(3, f"심볼: 채택 {stats['accepted_total']}  {stats['accepted_by_source']}")
    if use_rejected:
        step(3, f"  지울 대상에 기각 {len(rows)-len(acc)}개 포함 "
                f"({stats['rejected_by_reason']}) → {len(erase)}개")
    return rows, stats, acc, erase


def drawing_scale(accepted, stroke):
    """draw_h — 계기 버블 지름에서, 없으면 stroke width 에서."""
    bubbles = [max(r['bbox'][2] - r['bbox'][0], r['bbox'][3] - r['bbox'][1])
               for r in accepted if r['shape'] == 'closed_circle']
    if len(bubbles) >= MIN_BUBBLES:
        h = BUBBLE_TO_DRAW_H * statistics.median(bubbles)
        return h, f'버블 {len(bubbles)}개 중앙값 {statistics.median(bubbles):.0f}px × {BUBBLE_TO_DRAW_H}'
    return DRAW_H_PER_STROKE * stroke, f'stroke {stroke:.1f}px × {DRAW_H_PER_STROKE}'


# ---------------------------------------------------------------- 4 선

def grow_to_pipe(ink, bbox, stroke, max_grow, pipe_mult=2.5):
    """심볼 bbox 를 **잉크가 배관 굵기로 줄어들 때까지** 바깥으로 넓힌다.

    왜 필요한가
      검출기의 bbox 가 심볼을 딱 맞게 감싸지 못하는 일이 있다. page_001 의 V045 는
      bbox 가 90x55 인데 밸브의 위·아래 플랜지 바(폭 30px, 두께 3줄)가 그 밖으로
      삐져나온다. bbox 만 채우면 그 바가 남고, 폭 30px 짜리 잉크 덩어리라 다음
      단계가 그것을 **배관으로 읽는다**.

    왜 고정 여백이 아닌가
      몇 px 이 필요한지는 심볼마다 다르고 도면 배율마다 다르다. 대신 **무엇이 남아
      있는지** 를 본다: bbox 경계 바로 바깥의 잉크가 배관 한 가닥(stroke)보다 훨씬
      넓으면 아직 심볼 몸통이고, 배관 굵기로 줄면 거기서부터가 배관이다. 그 지점까지만
      넓힌다. 배관은 어느 도면에서나 한 가닥이므로 이 기준은 배율을 따라간다.

    max_grow 는 안전장치다 — 심볼이 배관과 나란히 붙어 있으면 멈출 자리를 못 찾는다.
    """
    x0, y0, x1, y1 = bbox
    H, W = ink.shape
    lim = max(1, int(round(max_grow)))
    wide = pipe_mult * stroke
    out = [x0, y0, x1, y1]
    for side in ('top', 'bottom', 'left', 'right'):
        n = 0
        while n < lim:
            if side == 'top':
                p = y0 - 1 - n
                if p < 0:
                    break
                w = int(ink[p, x0:x1].sum())
            elif side == 'bottom':
                p = y1 + n
                if p >= H:
                    break
                w = int(ink[p, x0:x1].sum())
            elif side == 'left':
                p = x0 - 1 - n
                if p < 0:
                    break
                w = int(ink[y0:y1, p].sum())
            else:
                p = x1 + n
                if p >= W:
                    break
                w = int(ink[y0:y1, p].sum())
            if w <= wide:
                break                  # 배관 굵기로 줄었다 — 여기서부터는 배관이다
            n += 1
        if side == 'top':
            out[1] = max(0, y0 - n)
        elif side == 'bottom':
            out[3] = min(H, y1 + n)
        elif side == 'left':
            out[0] = max(0, x0 - n)
        else:
            out[2] = min(W, x1 + n)
    return out


def erase_shape(img, bbox, sym, stroke=3.0, value=0, fallback_rect=True):
    """심볼을 **그 형태대로** 지운다. 사각형으로 지우면 모서리의 남의 잉크가 같이 날아간다.

    검출기가 형태를 이미 말해 주고 있다 — 원인지, 양끝이 둥근 캡슐인지, 테이퍼진
    오각형인지. 윤곽 점은 저장하지 않지만 형태 이름과 bbox 만으로 충분히 그릴 수 있다.

      closed_circle · dashed_circle   bbox 에 내접하는 타원
      STADIUM 인 elongated_box        양끝이 둥근 캡슐
      taper_flag                      잉크에서 윤곽을 다시 떠서 채운다
      그 외                           사각형 (carrier 는 형태가 없다 — 배관이 끊긴 구간이다)

    그리는 도형은 stroke 두께의 절반만큼 키운다. bbox 에 딱 내접시키면 윤곽선의
    **바깥쪽 획**이 도형 밖으로 남는다 — 계기 버블 하나당 70px 씩, page_001 전체로
    3,089px 이 그렇게 남았다. 얇은 호 고리라 눈에는 안 띄지만 run 으로는 잡힌다.

    돌려주는 것은 실제로 지운 픽셀 수다.
    """
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return 0
    before = int((img[y0:y1, x0:x1] != value).sum())
    shape = sym.get('shape')
    g = sym.get('geometry', {})
    pad = int(math.ceil(stroke / 2)) + 1
    if shape in ('closed_circle', 'dashed_circle'):
        cv2.ellipse(img, ((x0 + x1) // 2, (y0 + y1) // 2),
                    (w // 2 + pad, h // 2 + pad), 0, 0, 360, value, -1)
    elif shape == 'elongated_box' and g.get('corner_style') == 'STADIUM':
        r = min(w, h) // 2 + pad
        if w >= h:
            cv2.rectangle(img, (x0 + r, y0), (x1 - r - 1, y1 - 1), value, -1)
            cv2.circle(img, (x0 + r, y0 + r), r, value, -1)
            cv2.circle(img, (x1 - r - 1, y0 + r), r, value, -1)
        else:
            cv2.rectangle(img, (x0, y0 + r), (x1 - 1, y1 - r - 1), value, -1)
            cv2.circle(img, (x0 + r, y0 + r), r, value, -1)
            cv2.circle(img, (x0 + r, y1 - r - 1), r, value, -1)
    elif shape == 'taper_flag':
        sub = (img[y0:y1, x0:x1] != value).astype(np.uint8)
        cs = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if cs:
            c = max(cs, key=cv2.contourArea)
            view = img[y0:y1, x0:x1]
            cv2.drawContours(view, [c], -1, value, -1)
            cv2.drawContours(view, [c], -1, value, 2 * pad)   # 윤곽 획도 덮는다
        elif fallback_rect:
            cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), value, -1)
    else:
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), value, -1)
    return before - int((img[y0:y1, x0:x1] != value).sum())


def absorb_residual(rec, runs, labels, radius):
    """2패스 조각 중 기존 run 끝에 붙은 것은 그 run 에 흡수시킨다.

    화살촉은 리더의 **일부**지 별개 선이 아니다. 조각으로 두면 run 목록이 지저분해질
    뿐 아니라, 그 조각이 독자적으로 접촉을 만들어 리더가 엉뚱한 배관에 이어지기도 한다.
    한 물체는 한 run 이어야 한다.

    흡수는 조각의 골격 픽셀을 그 run 의 라벨로 칠하는 것이다. 라벨을 통해 픽셀 인접
    접촉이 잡히므로 연결은 그대로 유지되면서 run 수만 줄어든다.

    **끝점(a/b)은 옮기지 않는다.** 직선 run 의 끝을 꺾인 조각의 먼 끝으로 당기면 선
    전체가 기울어 원래 잉크를 벗어난다 — 그렇게 했더니 미설명 잉크가 630px 늘었다.
    조각은 꼬리(tail)로 따로 들고, 덮임 계산과 그림에만 쓴다.

    (흡수된 조각, 남은 조각) 을 돌려준다.
    """
    ends = [(k, e, r[e]) for k, r in runs.items() for e in ('a', 'b')]
    taken, kept = [], []
    for frag in rec:
        best = None
        for side in ('a', 'b'):
            px, py = frag[side]
            for k, e, q in ends:
                d = math.hypot(px - q[0], py - q[1])
                if d <= radius and (best is None or d < best[0]):
                    best = (d, k, e, side)
        if best is None:
            kept.append(frag)
            continue
        _, k, e, side = best
        host = runs[k]
        host.setdefault('tail', []).extend(frag['points'])
        for x, y in frag['points']:
            if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1] and labels[y, x] == 0:
                labels[y, x] = k
        taken.append(frag)
    return taken, kept


def _reaches_text(r, boxes, touch):
    """run 의 끝이 글자 상자에 닿는가 — 라벨을 가리키는 리더의 꼬리인가.

    상자 **안에 통째로 들어앉은** run 은 세지 않는다. 그것은 지우고 남은 글자 획이지
    라벨에 도착한 선이 아니다. 밖에서 뻗어와 닿은 것만 글자와 이어졌다고 본다.
    """
    x1, x2 = sorted((r['a'][0], r['b'][0]))
    y1, y2 = sorted((r['a'][1], r['b'][1]))
    for b in boxes:
        if b[0] <= x1 and x2 <= b[2] and b[1] <= y1 and y2 <= b[3]:
            continue                      # 상자 안에 통째로 들어앉았다
        for px, py in (r['a'], r['b']):
            if (b[0] - touch <= px <= b[2] + touch
                    and b[1] - touch <= py <= b[3] + touch):
                return True
    return False


def _hugs_one_symbol(r, symbols, touch):
    """양 끝이 **같은** 심볼 둘레에 갇혀 있는가.

    심볼 bbox 를 채우고 나면 경계 바깥으로 삐져나온 조각이 남는다 — 화살촉 꼬리,
    윤곽이 튀어나온 부분 같은 것들이다. 그 조각은 짧고, 그 심볼 둘레를 벗어나지
    않는다. 진짜 배관 스텁은 한쪽 끝만 심볼에 있고 다른 끝은 바깥으로 간다.
    """
    for s in symbols:
        b = s['bbox']
        inside = all(b[0] - touch <= px <= b[2] + touch
                     and b[1] - touch <= py <= b[3] + touch
                     for px, py in (r['a'], r['b']))
        if inside:
            return True
    return False


def _drop_crumbs(runs, labels, edges, symbols, text_boxes, P, args):
    """아무것에도 닿지 않은 run 을 버린다. 길이는 보지 않는다.

    심볼에 물렸다는 것만으로는 살리지 않는다. 심볼 둘레에 갇힌 채 아무것과도 이어지지
    않은 조각은 배관이 아니라 잘리고 남은 심볼 잉크다 (page_001 에서 47개).
    """
    links = line_stage.symbol_links(runs, symbols, P.touch)
    linked = {h['run'] for l in links for h in l['runs']}
    touched = {k for e in edges for k in e}
    keep = {k: r for k, r in runs.items()
            if k in touched or k in linked
            or _reaches_text(r, text_boxes, P.touch)}
    hugs = 0
    dropped = len(runs) - len(keep)
    if hugs:
        step(4, f'  심볼 둘레에 갇힌 잔해 {hugs}개 버림')
    if not dropped:
        via = collections.Counter(i.get('via') for i in edges.values())
        return runs, edges, collections.Counter(
            i['kind'] for i in edges.values()), None, via, links, 0
    labels2 = np.where(np.isin(labels, np.array(sorted(keep), np.int32)), labels, 0)
    via = collections.Counter()
    contacts, edges2, kinds2 = _contacts(keep, labels2, P, args, via)
    links = line_stage.symbol_links(keep, symbols, P.touch)
    return keep, edges2, kinds2, contacts, via, links, dropped


def _contacts(runs, labels, P, args, note):
    """픽셀 접촉 → 끝-대-끝 스냅 → 끝-대-중간 스냅 순서로 모은다.

    순서가 곧 우선순위다. 같은 쌍을 두 규칙이 잡으면 더 직접적인 증거가 이긴다:
    픽셀이 맞닿은 것 > 끝끼리 가까운 것 > 내 끝이 남의 중간에 가까운 것.
    """
    con = line_stage.contacts_from_labels(labels, runs)
    via = {k: 'pixels' for k in con}
    snap_r = 0.0 if args.no_snap_ends else P.endpoint_radius
    if snap_r:
        e2e = line_stage.contacts_from_endpoints(runs, snap_r, set(con))
        con.update(e2e)
        via.update({k: 'snap-end' for k in e2e})
        if not args.no_snap_mid:
            mid_r = args.snap_mid_radius or P.endpoint_radius
            e2m = line_stage.contacts_end_to_middle(runs, mid_r, set(con))
            con.update(e2m)
            via.update({k: 'snap-mid' for k in e2m})
    edges, kinds = line_stage.classify_contacts(
        runs, con, P.endpoint_radius, keep_crossovers=not args.drop_crossovers)
    for k, info in edges.items():
        info['via'] = via.get(k, 'pixels')
    note.update(collections.Counter(i['via'] for i in edges.values()))
    return con, edges, kinds


def holds_something(r, symbols, cleaned, P, args):
    """이 심볼을 테두리만 지워야 하는가 — 속에 무언가 있는가.

    크기로 가르지 않는다. 옮겨 온 쪽의 big_shape 규칙은 이 도면에서 오프페이지
    커넥터(331×61)를 전부 장비로 잘못 보았다. 길다는 것과 속을 품었다는 것은
    다른 말이다.
    """
    x0, y0, x1, y1 = r['bbox']
    if max(x1 - x0, y1 - y0) < args.hold_min:
        return False                      # 작은 것은 품을 것이 없다
    for o in symbols:
        if o is r:
            continue
        b = o['bbox']
        if b[0] >= x0 and b[2] <= x1 and b[1] >= y0 and b[3] <= y1:
            return True                   # 다른 심볼을 품었다
    m = max(int(round(args.erase_outline * P.draw_h)), 3) + 2
    inner = cleaned[y0 + m:y1 - m, x0 + m:x1 - m]
    if inner.size == 0:
        return False
    return float((inner > 0).mean()) > args.hold_ink


def stage_lines(gray, texts, symbols, P, out_dir, args, M, stroke=3.0):
    """글자와 심볼을 지운 뒤 run 을 뽑는다. (runs, edges, kinds, pipe_ink)."""
    ink = (carrier.binarize(gray).astype(np.uint8)) * 255

    # 마스크는 중복 제거 **전** 목록을 쓴다. 겹친 상자는 지우개에 해가 없고,
    # 중복 제거는 한 글줄의 두 절반을 하나로 묶어 절반을 버린다 (ocr_paddle 참조).
    mask_src = texts.get('texts_raw') or texts['texts']
    boxes = [t['bbox'] for t in mask_src]
    grown = textink.grow_boxes(boxes, args.box_grow) if args.box_grow else boxes
    # 검출된 심볼 자리의 잉크는 글자가 아니다 — OCR 상자가 아무리 크게 잡혀도.
    # 심볼이 **실제로 차지한 자리**를 칠한다. bbox 가 아니다 — 커넥터처럼 사각형이
    # 아닌 도형은 bbox 안에 빈 귀퉁이가 생기고, 거기 놓인 글자가 심볼로 오인된다.
    sym_mask = None
    if not args.no_protect_symbols:
        sym_mask = np.zeros(ink.shape, np.uint8)
        raw01 = (ink > 0).astype(np.uint8)
        for s in symbols:
            x0, y0, x1, y1 = s['bbox']
            if s['source'] == 'carrier':
                sym_mask[y0:y1, x0:x1] = 1      # 배관이 끊긴 구간 — 도형이 없다
                continue
            cs = cv2.findContours(raw01[y0:y1, x0:x1], cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)[0]
            if cs:
                c = max(cs, key=cv2.contourArea)
                cv2.drawContours(sym_mask[y0:y1, x0:x1], [c], -1, 1, -1)
            else:
                sym_mask[y0:y1, x0:x1] = 1
    cleaned, removed = textink.erase_text_ink(
        ink, grown, protect=sym_mask,
        symbol_overlap=args.symbol_overlap, margin=args.text_margin)
    M['ink_raw'] = int((ink > 0).sum())
    M['ink_after_text'] = int((cleaned > 0).sum())
    M['text_components_removed'] = removed
    step(4, f'  글자 지움: 상자 {len(boxes)}개 → 성분 {removed}개 제거, '
            f'잉크 {M["ink_raw"]:,} → {M["ink_after_text"]:,}')
    cv2.imwrite(os.path.join(out_dir, '04_text_erased.png'),
                np.where(cleaned > 0, gray, 255).astype(np.uint8),
                [cv2.IMWRITE_PNG_COMPRESSION, 1])

    pipe_ink = cleaned.copy()
    accepted = accepted_all = symbols
    inset = int(round(args.symbol_inset))
    filled = outlined = 0
    grown_px = 0
    for r in accepted:
        x0, y0, x1, y1 = r['bbox']
        if args.symbol_grow:
            gx0, gy0, gx1, gy1 = grow_to_pipe(
                cleaned > 0, r['bbox'], stroke, args.symbol_grow * stroke)
            grown_px += (gx1 - gx0) * (gy1 - gy0) - (x1 - x0) * (y1 - y0)
            x0, y0, x1, y1 = gx0, gy0, gx1, gy1
        if holds_something(r, accepted_all, cleaned, P, args):
            # 속에 진짜 배관이나 다른 심볼이 있다. 테두리만 가져간다.
            cv2.rectangle(pipe_ink, (x0, y0), (x1 - 1, y1 - 1), 0,
                          max(int(round(args.erase_outline * P.draw_h)), 3))
            outlined += 1
        elif args.no_shape_erase:
            cv2.rectangle(pipe_ink, (x0 + inset, y0 + inset),
                          (x1 - 1 - inset, y1 - 1 - inset), 0, -1)
            filled += 1
        else:
            erase_shape(pipe_ink, (x0 + inset, y0 + inset,
                                   x1 - inset, y1 - inset), r, stroke)
            filled += 1
    M['ink_pipe'] = int((pipe_ink > 0).sum())
    M['symbol_grow_px'] = grown_px
    step(4, f'  심볼 지움: {filled}개 채움 · {outlined}개 테두리만 (inset {inset}px, '
            f'배관 굵기까지 넓힘 +{grown_px:,}px²), 잉크 {M["ink_pipe"]:,}')
    cv2.imwrite(os.path.join(out_dir, '05_pipe_ink.png'),
                np.where(pipe_ink > 0, 0, 255).astype(np.uint8),
                [cv2.IMWRITE_PNG_COMPRESSION, 1])

    text_boxes = [t['bbox'] for t in texts['texts']]
    field = line_stage.extract_runs(pipe_ink, P, directions=args.directions)
    pipe_min = args.pipe_min if args.pipe_min is not None else 0.0
    runs = {k: r for k, r in field['runs'].items() if r['length'] >= pipe_min}
    labels = np.where(np.isin(field['labels'], np.array(sorted(runs), np.int32)),
                      field['labels'], 0)
    via = collections.Counter()
    contacts, edges, kinds = _contacts(runs, labels, P, args, via)

    # 글자 상자 안에만 있는 run 은 글자 획이다 — 단, 무언가에 이어져 있으면
    # 라벨까지 뻗은 리더 꼬리이므로 살린다. 그래서 접촉을 안 뒤에 거른다.
    boxed = 0
    if not args.keep_boxed_runs:
        slack = P.ref_h * args.box_slack
        touched = {k for e in edges for k in e}
        keep = {}
        for k, r in runs.items():
            x1, x2 = sorted((r['a'][0], r['b'][0]))
            y1, y2 = sorted((r['a'][1], r['b'][1]))
            in_box = any(b[0] - slack <= x1 and x2 <= b[2] + slack
                         and b[1] - slack <= y1 and y2 <= b[3] + slack
                         for b in text_boxes)
            if in_box and k not in touched:
                boxed += 1
                continue
            keep[k] = r
        runs = keep
        labels = np.where(np.isin(labels, np.array(sorted(runs), np.int32)), labels, 0)
        via = collections.Counter()
        contacts, edges, kinds = _contacts(runs, labels, P, args, via)

    added = 0
    if not args.no_split_branches:
        labels, runs, added = line_stage.split_at_branches(labels, runs, edges, P)
        runs = {k: r for k, r in runs.items() if r['length'] >= pipe_min}
        labels = np.where(np.isin(labels, np.array(sorted(runs), np.int32)), labels, 0)
        via = collections.Counter()
        contacts, edges, kinds = _contacts(runs, labels, P, args, via)

    # 길이로 자르지 않는다. 짧다는 것은 부스러기라는 뜻이 아니다 — 밸브와 밸브
    # 사이의 배관 토막은 짧고, 그 토막이 없으면 그래프가 끊긴다. 대신 **아무것에도
    # 닿지 않은 것**만 버린다: 다른 run 과도, 심볼과도, 글자와도 이어지지 않은 run.
    crumbs = 0
    if not args.keep_crumbs:
        runs, edges, kinds, contacts, via, links, crumbs = _drop_crumbs(
            runs, labels, edges, symbols, text_boxes, P, args)

    # 2패스 — 방향장이 못 덮은 잉크를 골격 경로로 줍는다. 프로브(25px 자)보다 짧은
    # 조각은 1패스에서 support 가 모자라 통째로 버려지는데, 골격 걷기는 자를 쓰지
    # 않으므로 그 조각들이 살아난다. 꺾인 선은 폴리라인 하나로 나온다.
    rec = []
    if not args.no_residual:
        rec, rstat = residual_stage.recover(
            pipe_ink > 0, runs, start_id=(max(runs) if runs else 0) + 1,
            min_path=args.residual_min)
        absorbed, rec = absorb_residual(
            rec, runs, labels,
            args.absorb_radius if args.absorb_radius is not None else P.endpoint_radius)
        M['residual_absorbed'] = len(absorbed)
        for r in rec:
            runs[r['id']] = r
            # 골격 픽셀을 라벨 그림에 얹어야 픽셀 인접 접촉이 잡힌다. 이미 남이
            # 차지한 자리는 건드리지 않는다 — 1패스 결과를 덮어쓰면 안 된다.
            for x, y in r['points']:
                if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1] \
                        and labels[y, x] == 0:
                    labels[y, x] = r['id']
        M['residual_px'] = rstat['residual_px']
        M['residual_paths'] = rstat['paths']
        M['residual_arcs_dropped'] = rstat['arcs_dropped']
        step(4, f"  2패스: 잔여 {rstat['residual_px']}px → 폴리라인 {rstat['paths']}개 "
                f"(호 {rstat['arcs_dropped']}개 제외) → 기존 run 에 흡수 {len(absorbed)}개, "
                f"독립 {len(rec)}개")
        via = collections.Counter()
        contacts, edges, kinds = _contacts(runs, labels, P, args, via)

    # 심볼 둘레 잔해 정리 — **접촉을 다 센 뒤**에 한 번만 건다.
    #
    # 판정은 둘을 함께 본다: 양 끝이 같은 심볼 둘레에 갇혀 있고, **아무 run 과도
    # 이어지지 않았다**. 전자만 보면 정상적인 경계 스텁까지 걸린다 — 스텁은 13px
    # 남짓이라 양 끝이 다 bbox 20px 안에 들어오기 때문이다. 실제로 그렇게 걸었더니
    # 2패스가 88 → 18 로 무너지고 미설명 잉크가 원상복귀했다.
    #
    # 진짜 잔해는 이어진 곳이 없다. 배관을 심볼에 붙여 주는 스텁은 배관 쪽에 엣지가
    # 있으므로 남는다.
    hug = 0
    if not args.keep_symbol_scraps:
        touched = {k for e in edges for k in e}
        drop = [k for k, r in runs.items()
                if k not in touched and _hugs_one_symbol(r, symbols, P.touch)]
        if drop:
            for k in drop:
                runs.pop(k)
            labels = np.where(np.isin(labels, np.array(sorted(drop), np.int32)),
                              0, labels)
            via = collections.Counter()
            contacts, edges, kinds = _contacts(runs, labels, P, args, via)
            hug = len(drop)
            step(4, f'  심볼 둘레 잔해 {hug}개 버림 (이어진 곳 없음)')
    M['symbol_scraps_dropped'] = hug

    # 점선 판정은 1패스 run 에만 건다. 2패스는 폴리라인이라 a-b 직선을 따라 재면
    # 꺾인 바깥의 잉크를 읽는다. 짧기도 해서 리듬을 말할 만큼의 길이가 없다.
    straight_runs = {k: r for k, r in runs.items() if r.get('from') != 'residual'}
    dash_stage.classify(pipe_ink, straight_runs, P, min_gaps=args.min_gaps)
    for r in rec:
        if r['id'] in runs:
            r['line_type'] = 'solid'
    lt = collections.Counter(r['line_type'] for r in runs.values())
    step(4, f'  선: run {len(runs)}개 (이어붙임 {field["rejoined"]}, '
            f'글자상자 안이라 버림 {boxed}, 분기 분할 {added}, '
            f'방향 불명 {field["no_direction"]}px)')
    step(4, '  접촉: ' + ', '.join(f'{k} {v}' for k, v in kinds.most_common())
            + f' → {len(edges)}개 채택')
    step(4, '  근거별: ' + ', '.join(f'{k} {v}' for k, v in via.most_common()))
    step(4, '  ' + ', '.join(f'{v} {k}' for k, v in lt.items()))
    links = line_stage.symbol_links(runs, symbols, P.touch)
    nlink = sum(len(l['runs']) for l in links)
    if crumbs:
        step(4, f'  부스러기 {crumbs}개 버림 (선·심볼·글자 어디에도 안 닿음)')
    step(4, f'  심볼-배관: 심볼 {len(links)}/{len(symbols)}개에 run {nlink}개가 물림 '
            f'(touch {P.touch:.0f}px)')
    return runs, edges, kinds, pipe_ink, labels, via, links


def _place_labels(items, W, H, cell=220):
    """겹치지 않게 라벨 자리를 잡는다. (놓은 [(x, y, w, h, i)], 못 놓은 수).

    상자마다 위·아래·오른쪽·왼쪽 순으로 시도하고, 이미 놓인 라벨과 겹치면 다음
    자리로 넘어간다. 네 자리가 다 막히면 포기한다 — 억지로 끼워 넣으면 겹쳐서
    둘 다 못 읽게 되고, 읽을 수 있는 것 몇 개가 읽을 수 없는 전부보다 낫다.

    격자에 나눠 담아 근처 것만 비교한다. 상자가 900개라 전수 비교는 느리다.
    """
    grid = collections.defaultdict(list)

    def hits(x, y, w, h):
        for gx in range(x // cell, (x + w) // cell + 1):
            for gy in range(y // cell, (y + h) // cell + 1):
                for (ax, ay, aw, ah) in grid[(gx, gy)]:
                    if x < ax + aw and ax < x + w and y < ay + ah and ay < y + h:
                        return True
        return False

    def put(x, y, w, h):
        for gx in range(x // cell, (x + w) // cell + 1):
            for gy in range(y // cell, (y + h) // cell + 1):
                grid[(gx, gy)].append((x, y, w, h))

    placed, skipped = [], 0
    for i, (box, tw, th) in enumerate(items):
        x0, y0, x1, y1 = box
        for x, y in ((x0, y0 - th - 3), (x0, y1 + 3),
                     (x1 + 4, y0), (x0 - tw - 4, y0)):
            x = max(0, min(x, W - tw)); y = max(0, min(y, H - th))
            if not hits(x, y, tw, th):
                put(x, y, tw, th)
                placed.append((x, y, tw, th, i))
                break
        else:
            skipped += 1
    return placed, skipped


def draw_texts(gray, texts, scale=0.55, thick=1):
    """OCR 이 글자로 본 자리와, 무엇으로 읽었는지를 함께 표시한다.

    두 겹을 같이 그린다. 중복 제거 **전** 상자(연회색)가 지우개가 실제로 쓰는
    마스크이고, 중복 제거 **후** 상자(초록)가 글자 목록이다. 둘을 겹쳐 두면
    한 글줄의 절반이 중복 제거에서 사라지는 일이 보인다 — 초록이 없는 자리에
    회색만 있으면 그 자리다.

    읽은 글자는 **중복 제거된 쪽만** 적는다. raw 까지 적으면 같은 말이 두 번씩
    겹쳐 어느 쪽도 못 읽는다.
    """
    im = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    im = (im * 0.30 + 175).astype(np.uint8)
    H, W = im.shape[:2]
    raw = texts.get('texts_raw') or texts['texts']
    for t in raw:
        x0, y0, x1, y1 = t['bbox']
        cv2.rectangle(im, (x0, y0), (x1 - 1, y1 - 1), (190, 190, 190), 2)
    for t in texts['texts']:
        x0, y0, x1, y1 = t['bbox']
        cv2.rectangle(im, (x0, y0), (x1 - 1, y1 - 1), (0, 155, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    items = []
    for t in texts['texts']:
        label = t['text'].strip()
        if not label:
            continue
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        items.append((t['bbox'], tw + 4, th + 4))
    placed, skipped = _place_labels(items, W, H)
    keep = [t for t in texts['texts'] if t['text'].strip()]
    for (x, y, tw, th, i) in placed:
        label = keep[i]['text'].strip()
        cv2.rectangle(im, (x, y), (x + tw, y + th), (255, 255, 255), -1)
        cv2.putText(im, label, (x + 2, y + th - 4), font, scale, (170, 0, 0), thick,
                    cv2.LINE_AA)

    cv2.rectangle(im, (12, 12), (760, 152), (255, 255, 255), -1)
    cv2.rectangle(im, (12, 12), (760, 152), (0, 0, 0), 2)
    cv2.putText(im, f'raw {len(raw)} (erase mask)', (28, 56), font, 1.0,
                (150, 150, 150), 3)
    cv2.putText(im, f'deduped {len(texts["texts"])} (text list)', (28, 104), font,
                1.0, (0, 155, 0), 3)
    cv2.putText(im, f'{len(placed)} labels drawn, {skipped} skipped (no free space)',
                (28, 140), font, 0.65, (170, 0, 0), 2)
    return im


#: 발표용 팔레트 (BGR). 원색을 피하고 채도를 낮췄다 — 원색은 선이 얇아지면 눈이
#: 아프고, 인쇄·프로젝터에서 서로 뭉친다. 배관 두 방향은 같은 계열의 명도 차이로,
#: 신호선은 따뜻한 색으로 갈라 한눈에 계통이 보이게 했다.
PALETTE = {
    'pipe_h':   (150, 108,  62),   # 청회색  — 수평 배관
    'pipe_v':   (118, 138,  70),   # 청록     — 수직 배관
    'dashed':   ( 90,  92, 190),   # 벽돌색   — 점선 신호선
    'residual': ( 70, 145, 205),   # 호박색   — 2패스 폴리라인
    'symbol':   (185, 178, 168),   # 따뜻한 회색 — 심볼 테두리
    'symbol_x': (205, 205, 205),   # 옅은 회색   — 기각된 심볼
    'link':     (120, 165, 120),   # 연녹      — 심볼-배관 접점
    'pixels':   ( 70,  70,  70),   # 짙은 회색  — 픽셀 접촉
    'snap_end': ( 70, 145, 205),   # 호박색    — 끝-대-끝 스냅
    'snap_mid': (160,  95, 150),   # 자주      — 끝-대-중간 스냅
}

VIA_COLOR = {'pixels': PALETTE['pixels'], 'snap-end': PALETTE['snap_end'],
             'snap-mid': PALETTE['snap_mid']}


def draw_lines(gray, runs, edges, symbols=None, links=None, legend=False,
               lw=2, fade=0.16, ground=214):
    """run · 접촉 · 심볼을 한 장에 얹는다. 발표에 쓰는 그림이라 얇고 차분하게 그린다.

    점 색은 **kind 가 아니라 근거(via)** 로 준다. 무엇이 T 접합이냐보다 그 접합을
    무엇으로 잡았느냐가 되짚을 때 필요한 정보이기 때문이다.

    lw 는 선 굵기, fade/ground 는 배경 도면을 얼마나 옅게 깔지다. 점 지름은 선
    굵기에서 따라간다 — 굵기를 바꿔도 비율이 유지된다.
    """
    im = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    im = (im * fade + ground).clip(0, 255).astype(np.uint8)
    dot = max(2, int(round(lw * 1.6)))
    ring = max(4, int(round(lw * 3.2)))
    if symbols:
        for sy in symbols:
            x0, y0, x1, y1 = sy['bbox']
            c = PALETTE['symbol_x'] if sy['status'] != 'ACCEPTED' else PALETTE['symbol']
            cv2.rectangle(im, (x0, y0), (x1 - 1, y1 - 1), c, max(1, lw - 1))
    for r in runs.values():
        if r.get('from') == 'residual':
            cv2.polylines(im, [np.array(r['points'], np.int32)], False,
                          PALETTE['residual'], lw, cv2.LINE_AA)
            continue
        c = (PALETTE['dashed'] if r['line_type'] == 'dashed'
             else (PALETTE['pipe_h'] if r['orient'] == 'h' else PALETTE['pipe_v']))
        cv2.line(im, (int(r['a'][0]), int(r['a'][1])),
                 (int(r['b'][0]), int(r['b'][1])), c, lw, cv2.LINE_AA)
        if r.get('tail'):
            cv2.polylines(im, [np.array(r['tail'], np.int32)], False,
                          PALETTE['residual'], max(1, lw - 1), cv2.LINE_AA)
    if links:
        for l in links:
            for h in l['runs']:
                cv2.circle(im, (int(h['at'][0]), int(h['at'][1])), ring,
                           PALETTE['link'], max(1, lw - 1), cv2.LINE_AA)
    for info in edges.values():
        x, y = info['at']
        col = VIA_COLOR.get(info.get('via'), PALETTE['pixels'])
        cv2.circle(im, (int(x), int(y)), dot, col, -1, cv2.LINE_AA)
        if info['kind'] != 'end-to-end':
            cv2.circle(im, (int(x), int(y)), ring + dot, col,
                       max(1, lw - 1), cv2.LINE_AA)
    if legend:
        _lines_legend(im, runs, edges, links)
    return im


def _lines_legend(im, runs, edges, links):
    rows = [('residual polyline', PALETTE['residual']),
            ('pixel contact', PALETTE['pixels']),
            ('snap end-to-end', PALETTE['snap_end']),
            ('snap end-to-middle', PALETTE['snap_mid']),
            ('symbol link', PALETTE['link'])]
    n = collections.Counter(i.get('via') for i in edges.values())
    cnt = [sum(1 for r in runs.values() if r.get('from') == 'residual'),
           n['pixels'], n['snap-end'], n['snap-mid'],
           sum(len(l['runs']) for l in (links or []))]
    h = 52 * len(rows) + 90
    cv2.rectangle(im, (12, 12), (700, h), (255, 255, 255), -1)
    cv2.rectangle(im, (12, 12), (700, h), (120, 120, 120), 2)
    cv2.putText(im, f'{len(runs)} runs, {len(edges)} edges', (30, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 60), 2)
    for i, ((name, col), v) in enumerate(zip(rows, cnt)):
        y = 110 + 52 * i
        cv2.circle(im, (44, y - 8), 12, col, -1, cv2.LINE_AA)
        cv2.putText(im, f'{name}  {v}', (72, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (60, 60, 60), 2)


# ---------------------------------------------------------------- main

def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input')
    ap.add_argument('out_dir')
    ap.add_argument('--ocr-python', default=DETECT_VENV, help='paddle 이 있는 인터프리터')
    ap.add_argument('--texts', default=None, help='만들어 둔 texts.json')
    ap.add_argument('--redo-ocr', action='store_true',
                    help='캐시를 무시하고 OCR 을 다시 돌린다 (기본은 캐시 사용)')
    ap.add_argument('--ocr-tile', type=int, default=1280)
    ap.add_argument('--ocr-overlap', type=int, default=384)
    ap.add_argument('--ocr-scale', type=float, default=1.0)
    ap.add_argument('--ocr-det-thresh', type=float, default=None)
    ap.add_argument('--ocr-det-box-thresh', type=float, default=None,
                    help='OCR 상자 채택 문턱 (기본 0.3 — 한 글자짜리도 잡게 푼 값)')
    ap.add_argument('--ocr-det-unclip', type=float, default=None,
                    help='OCR 상자 확장 비율 (기본 2.2 — 글자를 넉넉히 감싸게)')
    ap.add_argument('--no-ocr-rotate', action='store_true',
                    help='OCR 의 90도 회전 패스를 건너뛴다')
    ap.add_argument('--no-tag-erase', action='store_true', help='2단계를 건너뛴다')
    ap.add_argument('--symbols-after-text', action='store_true',
                    help='글자를 먼저 지운 그림에서 심볼을 찾는다 (순서 실험용)')
    ap.add_argument('--skip-rejected-symbols', action='store_true',
                    help='기각된 심볼은 지우지 않는다 (기본은 함께 지운다)')
    # 선
    ap.add_argument('--directions', type=int, default=36,
                    help='방향장 버킷 수. 36 = 5도 간격. 18(10도)은 얕은 대각선이 '
                         '두 버킷 사이에 떨어져 방향 불명으로 버려지고, 72(2.5도)는 '
                         '같은 직선이 버킷을 넘나들어 잘게 쪼개진다')
    ap.add_argument('--no-shape-erase', action='store_true',
                    help='심볼을 형태 대신 사각형으로 지운다')
    ap.add_argument('--symbol-grow', type=float, default=4.0,
                    help='심볼 bbox 를 배관 굵기가 될 때까지 넓혀서 지운다. '
                         'stroke 배수 상한 (0 이면 끄기)')
    ap.add_argument('--symbol-inset', type=float, default=0.0,
                    help='심볼 bbox 를 안쪽으로 물릴 픽셀 (기본 0 — 우리 bbox 는 딱 맞다)')
    ap.add_argument('--erase-outline', type=float, default=0.35,
                    help='큰 도형 테두리 두께, draw_h 배수')
    ap.add_argument('--pipe-min', type=float, default=None,
                    help='run 길이 하한 (px). 기본 0 — 길이로 자르지 않고 '
                         '아무것에도 안 닿은 것만 버린다')
    ap.add_argument('--keep-symbol-scraps', action='store_true',
                    help='심볼 둘레에 갇힌 채 아무것과도 안 이어진 조각을 남긴다')
    ap.add_argument('--absorb-radius', type=float, default=None,
                    help='2패스 조각을 기존 run 에 흡수시킬 반경 (기본 endpoint_radius). '
                         '0 이면 흡수하지 않는다')
    ap.add_argument('--residual-min', type=int, default=8,
                    help='2패스가 회수할 최소 경로 길이 (골격 픽셀). 작을수록 화살촉·'
                         '원호 조각이 섞인다')
    ap.add_argument('--line-width', type=int, default=2,
                    help='06_lines.png 의 선 굵기 (기본 2)')
    ap.add_argument('--line-fade', type=float, default=0.16,
                    help='배경 도면을 얼마나 진하게 깔지 (0~1, 낮을수록 옅다)')
    ap.add_argument('--lines-legend', action='store_true',
                    help='06_lines.png 좌상단에 범례를 그린다 (기본 없음)')
    ap.add_argument('--no-residual', action='store_true',
                    help='2패스(잔여 잉크 골격 회수)를 건너뛴다')
    ap.add_argument('--keep-crumbs', action='store_true',
                    help='아무것에도 안 닿은 run 도 남긴다')
    ap.add_argument('--box-slack', type=float, default=0.15)
    ap.add_argument('--text-margin', type=float, default=0.45,
                    help='글자 판정 창을 OCR 상자보다 키울 양 (짧은 변 배수)')
    ap.add_argument('--symbol-overlap', type=float, default=0.35,
                    help='상자 안 덩어리가 심볼과 이 비율 넘게 겹치면 글자로 보지 않는다')
    ap.add_argument('--protect-overlap', type=float, default=0.2,
                    help='글자 상자가 심볼에 이 비율 이하로 겹치면 심볼을 침범해서라도 '
                         '지운다 (기본 0.2). 0 이면 보호를 항상 지킨다')
    ap.add_argument('--partial-max-glyph', type=float, default=1.5,
                    help='부분 지우기가 한 번에 지울 수 있는 최대 크기 (ref_h 배수). '
                         '0 이면 무제한')
    ap.add_argument('--partial-thick', type=float, default=1.3,
                    help='부분 지우기에서 글자로 볼 최소 획 두께 (stroke 배수). '
                         '배관의 두께가 곧 stroke 이므로 1.0 이하면 배관을 먹는다')
    ap.add_argument('--partial-grow', type=float, default=0.6,
                    help='부분 지우기 갈래에서만 OCR 상자를 늘릴 양 (글자 높이 배수)')
    ap.add_argument('--no-partial-text', action='store_true',
                    help='심볼에 붙은 글자를 상자 안 부분만 지우는 갈래를 끈다')
    ap.add_argument('--no-protect-symbols', action='store_true',
                    help='글자 지우개가 심볼 자리도 지우게 둔다 (기본은 보호)')
    ap.add_argument('--box-grow', type=float, default=0.0,
                    help='OCR 상자를 글줄 방향으로 늘릴 양, 글자 높이 배수. '
                         '기본 0 — raw 마스크를 쓰면 대개 필요 없고, 늘리면 라벨 옆 '
                         '리더 선과 화살촉을 글자로 오인해 함께 지운다')
    ap.add_argument('--hold-min', type=float, default=150.0,
                    help='이보다 작으면 무엇도 품지 않은 것으로 본다 (px)')
    ap.add_argument('--hold-ink', type=float, default=0.03,
                    help='테두리 안쪽 잉크 비율이 이보다 크면 속을 품은 것으로 본다')
    ap.add_argument('--min-gaps', type=int, default=1,
                    help='점선으로 보려면 필요한 간격 수. 1 이면 간격 하나짜리 짧은 '
                         '신호선도 잡는다 (dashes.py 의 잘린끝 규칙과 짝이다)')
    ap.add_argument('--inside', type=float, default=0.8)
    ap.add_argument('--reach', type=int, default=None)
    ap.add_argument('--drop-crossovers', action='store_true',
                    help='양쪽 다 중간에서 만난 접촉(crossover)을 버린다. 기본은 '
                         '버리지 않는다 — 이 도면에는 교차가 없다고 보고 있다')
    ap.add_argument('--no-snap-ends', action='store_true')
    ap.add_argument('--no-snap-mid', action='store_true',
                    help='내 끝이 남의 중간에 닿는 T 접합 스냅을 끈다')
    ap.add_argument('--snap-mid-radius', type=float, default=None,
                    help='끝-대-중간 스냅 반경 (기본 endpoint_radius)')
    ap.add_argument('--no-split-branches', action='store_true')
    ap.add_argument('--keep-boxed-runs', action='store_true')
    ap.add_argument('--draw-h', type=float, default=None, help='도면 스케일을 직접 준다')
    ap.add_argument('--bridge-gap', type=float, default=None,
                    help='교차가 훔쳐간 픽셀 너머로 run 을 다시 잇는 최대 간격 (px). '
                         '기본 0.6 x draw_h. 대각선 점선은 같은 간격이라도 조각 끝에서 '
                         '끝까지가 멀어 기본값으로는 안 이어진다')
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.perf_counter()
    gray = carrier.load(a.input)
    H, W = gray.shape[:2]
    print(f'{a.input}  {W}x{H}')

    # 페이지를 한 번만 렌더해 둔다. OCR 은 다른 venv 에서 도는데 그쪽에는 PDF 를
    # 여는 v34(→skimage)가 없다. 렌더 코드를 두 벌로 만들지 않으려는 것이다.
    page_png = os.path.join(a.out_dir, '00_page.png')
    cv2.imwrite(page_png, gray, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    texts = stage_text(page_png, a.out_dir, a.ocr_python, a.texts, a.redo_ocr,
                       {'tile': a.ocr_tile, 'overlap': a.ocr_overlap,
                        'scale': a.ocr_scale, 'no_rotate': a.no_ocr_rotate,
                        'det_thresh': a.ocr_det_thresh,
                        'det_box_thresh': a.ocr_det_box_thresh,
                        'det_unclip': a.ocr_det_unclip})

    cv2.imwrite(os.path.join(a.out_dir, '01_texts.png'), draw_texts(gray, texts),
                [cv2.IMWRITE_PNG_COMPRESSION, 1])

    params = carrier.Params()
    if a.no_tag_erase:
        work, tags = gray, []
        step(2, '타원 태그: 건너뜀 (--no-tag-erase)')
    else:
        work, tags = stage_tags(gray, a.out_dir, params)
        cv2.imwrite(os.path.join(a.out_dir, '02_tags_erased.png'), work,
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])

    # 심볼을 글자 지운 그림에서 찾을 것인가. 옮겨 온 쪽은 그렇게 한다 — 글자가
    # 남아 있으면 검출기가 글자 뭉치를 도형으로 읽는다. 다만 우리 기하 검출기는
    # **내부 잉크를 게이트로 쓴다** (계기 버블 안 코드, 태그 안 글자). 글자를 먼저
    # 지우면 그 증거가 사라져 버블과 태그를 놓칠 수 있다. 그래서 기본은 끄고,
    # --symbols-after-text 로 재 볼 수 있게만 둔다.
    sym_input = work
    if a.symbols_after_text:
        ink0 = (carrier.binarize(work).astype(np.uint8)) * 255
        pre, npre = textink.erase_text_ink(
            ink0, [t['bbox'] for t in (texts.get('texts_raw') or texts['texts'])],
            a.inside, a.reach)
        sym_input = np.where(pre > 0, work, 255).astype(np.uint8)
        step(3, f'  글자를 먼저 지우고 심볼을 찾는다 (성분 {npre}개 제거)')
    rows, sym_stats, accepted, to_erase = stage_symbols(
        sym_input, a.out_dir, params, use_rejected=not a.skip_rejected_symbols)
    detect.write_json(a.out_dir, rows, sym_stats, params, a.input)
    detect.write_csv(a.out_dir, detect._public(rows))
    cv2.imwrite(os.path.join(a.out_dir, '03_symbols.png'),
                render.overlay(work, rows), [cv2.IMWRITE_PNG_COMPRESSION, 1])

    stroke = sym_stats['carrier'].get('stroke_width_px') or 3.0
    draw_h = a.draw_h if a.draw_h else None
    if draw_h is None:
        draw_h, how = drawing_scale(accepted, stroke)
    else:
        how = '직접 지정'
    ref_h = texts['ref_h'] or draw_h
    P = LineParams.from_text_height(ref_h, draw_h)
    if a.bridge_gap is not None:
        P.bridge_gap = a.bridge_gap
    step(4, f'스케일: ref_h={ref_h:.1f}px  draw_h={draw_h:.1f}px ({how})')
    step(4, f'  line_min_len={P.line_min_len:.0f} probe={P.probe:.0f} '
            f'bridge_gap={P.bridge_gap:.0f} endpoint_radius={P.endpoint_radius:.0f} '
            f'big_shape={P.big_shape:.0f}')

    M = {}
    runs, edges, kinds, pipe_ink, labels, via, links = stage_lines(
        work, texts, to_erase, P, a.out_dir, a, M, stroke)

    cv2.imwrite(os.path.join(a.out_dir, '06_lines.png'),
                draw_lines(work, runs, edges, to_erase, links, a.lines_legend,
                           lw=a.line_width, fade=a.line_fade),
                [cv2.IMWRITE_PNG_COMPRESSION, 1])
    with open(os.path.join(a.out_dir, 'lines.json'), 'w', encoding='utf-8') as fp:
        json.dump({'input': str(a.input), 'size': [W, H],
                   'params': {'ref_h': ref_h, 'draw_h': draw_h,
                              'draw_h_from': how, 'directions': a.directions,
                              **{k: round(float(v), 2) for k, v in
                                 (('line_min_len', P.line_min_len), ('probe', P.probe),
                                  ('bridge_gap', P.bridge_gap),
                                  ('endpoint_radius', P.endpoint_radius))}},
                   'stats': {'runs': len(runs), 'edges': len(edges),
                             'contact_kinds': dict(kinds),
                             'contact_via': dict(via),
                             'symbol_links': len(links),
                             'symbol_link_runs': sum(len(l['runs']) for l in links),
                             'line_types': dict(collections.Counter(
                                 r['line_type'] for r in runs.values()))},
                   'runs': list(runs.values()),
                   'edges': [{'a': k[0], 'b': k[1], **v} for k, v in edges.items()],
                   'symbol_links': links,
                   'erased_tags': tags},
                  fp, ensure_ascii=False, indent=1)

    # 덮임은 run 이 **실제로 지나간 자리**로 센다. 폴리라인과 흡수한 꼬리를 직선
    # a-b 로 근사하면 꺾인 부분이 안 덮인 것으로 잡혀, 흡수할수록 수치가 나빠진다.
    cov = np.zeros(pipe_ink.shape, np.uint8)
    for r in runs.values():
        if r.get('points'):
            cv2.polylines(cov, [np.array(r['points'], np.int32)], False, 1, 7)
        else:
            cv2.line(cov, (int(r['a'][0]), int(r['a'][1])),
                     (int(r['b'][0]), int(r['b'][1])), 1, 7)
        if r.get('tail'):
            cv2.polylines(cov, [np.array(r['tail'], np.int32)], False, 1, 7)
    unexplained = int(((pipe_ink > 0) & (cov == 0)).sum())
    prev = metrics.load(a.out_dir)
    cur = metrics.collect(
        texts=len(texts['texts']), texts_raw=len(texts.get('texts_raw') or []),
        tags_erased=len(tags),
        symbols_accepted=sym_stats['accepted_total'],
        symbols_rejected=sym_stats['rejected_total'],
        **{f'sym_{k}': v for k, v in sym_stats['accepted_by_source'].items()},
        **M,
        ink_unexplained=unexplained,
        ink_explained_pct=round(100 * (1 - unexplained / max(M.get('ink_pipe', 1), 1)), 2),
        runs=len(runs), edges=len(edges),
        **{f'contact_{k.replace("-", "_")}': v for k, v in kinds.items()},
        **{f'via_{k.replace("-", "_")}': v for k, v in via.items()},
        solid=sum(1 for r in runs.values() if r['line_type'] == 'solid'),
        dashed=sum(1 for r in runs.values() if r['line_type'] == 'dashed'),
        symbol_links=len(links),
        symbol_link_runs=sum(len(l['runs']) for l in links),
        seconds=round(time.perf_counter() - t0))
    print('\n[비교] 직전 실행 대비')
    for line in metrics.report(prev, cur):
        print(line)
    metrics.save(a.out_dir, cur)

    print(f'\n소요 {time.perf_counter()-t0:.0f}s  →  {a.out_dir}')
    print('  texts.json  01_texts.png  02_tags_erased.png  symbols.json 03_symbols.png')
    print('  04_text_erased.png  05_pipe_ink.png  ← 선 검출 직전 그림')
    print('  lines.json  06_lines.png')


if __name__ == '__main__':
    main()

"""symbol_detection 진입점 — 배관 기하(v48) + 닫힌 윤곽 기하를 합쳐 심볼을 낸다.

  python symbol_detection/detect.py <img|pdf> <out_dir> [--stages] [--no-carrier-filter]

두 갈래
  carrier   배관이 끊긴 자리 (carrier.py → ap1000/ v48). 인라인 밸브류.
  기하      닫힌 사각/원 윤곽 · 점선 큰 원 · 테이퍼 커넥터 · 세로긴 박스
            (outline.py · boxed.py, pkg9 에서 이식)

클래스를 내지 않는다
  family/kind 필드를 두지 않는다. 검출기가 기하로 가른 갈래 이름(INSTRUMENT FRAME,
  TANK, VESSEL, BOXED CODE TAG …)은 추적용으로 geometry 안에만 남긴다. 템플릿
  정합(native_bank)과 버블 내부 크롭 OCR(label_frames)은 이 폴더에 없다.

잉크맵을 한 번만 만든다
  v34.binarize 와 pkg9 binarize_otsu 가 같은 맵을 만드는 것을 확인했으므로
  (page_001 에서 불일치 픽셀 0) v34 쪽으로 한 번 이진화해 두 갈래가 나눠 쓴다.
  v34 를 기준으로 삼은 것은 이미 1bit 로 저장된 도면을 따로 처리하는 갈래가
  그쪽에만 있기 때문이다.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import carrier  # noqa: E402
import corner  # noqa: E402
import leader  # noqa: E402
import merge  # noqa: E402
import render  # noqa: E402
from boxed import boxed_tags, connector_flags  # noqa: E402
from outline import dashed_circles, instrument_frames  # noqa: E402


def run(gray, params=None, boxed_corner='all'):
    """(rows, stats, ctx). rows 의 bbox 는 전부 배타 구간 [x0,y0,x1,y1) 이다.

    boxed_corner 는 박스 태그를 끝단 모양으로 거른다 ('all'|'stadium'|'rect').
    'all' 이 아니면 어긋나는 것을 CORNER_STYLE 로 기각한다 — 버리지는 않는다.
    """
    params = params or carrier.Params()
    timing = {}

    t = time.perf_counter()
    ink = carrier.binarize(gray)
    raw = ink.astype(np.uint8)
    timing['binarize'] = round(time.perf_counter() - t, 2)

    # 순서는 pkg9 symbol_detect.run 과 같다 — boxed_tags 가 앞선 두 갈래를
    # exclude 로 받아야 커넥터·프레임을 태그로 두 번 세지 않는다.
    t = time.perf_counter()
    frames = instrument_frames(raw)
    connectors = connector_flags(raw)
    tags = boxed_tags(raw, exclude=connectors + frames)
    corner.annotate(raw, tags)
    rings = dashed_circles(raw)
    timing['geometry'] = round(time.perf_counter() - t, 2)

    t = time.perf_counter()
    carrier_rows, carrier_stats, ctx = carrier.run(gray, ink, params)
    timing['carrier'] = round(time.perf_counter() - t, 2)

    geom_rows = (merge.normalize(frames, 'frame')
                 + merge.normalize(rings, 'ring')
                 + merge.normalize(connectors, 'connector')
                 + merge.normalize(tags, 'boxed'))
    merge.number_carrier(carrier_rows)
    covered = merge.suppress_covered_carrier(carrier_rows, geom_rows)
    # 끝단 필터는 carrier 억제 뒤에 건다 — 앞에 걸면 기각된 박스가 덮고 있던
    # carrier 가 되살아나서, 박스를 거르는 일이 carrier 개수를 조용히 바꾼다.
    styled = _filter_corner(geom_rows, boxed_corner)
    _trace_leaders(raw, geom_rows)

    rows = carrier_rows + geom_rows
    accepted = [r for r in rows if r['status'] == 'ACCEPTED']
    stats = {
        'image_size': [int(gray.shape[1]), int(gray.shape[0])],
        'accepted_total': len(accepted),
        'accepted_by_source': dict(sorted(Counter(r['source'] for r in accepted).items())),
        'accepted_by_shape': dict(sorted(Counter(r['shape'] for r in accepted).items())),
        'rejected_total': len(rows) - len(accepted),
        'rejected_by_reason': dict(sorted(Counter(
            r['reject_reason'] for r in rows if r['status'] != 'ACCEPTED').items())),
        'carrier_covered_by_outline': covered,
        'boxed_by_corner_style': styled,
        'carrier': carrier_stats,
        'seconds': timing,
    }
    return rows, stats, ctx


def _trace_leaders(raw, geom_rows):
    """채택된 타원 태그에 매달린 리더 선을 따라가 geometry 에 적어 둔다.

    픽셀 목록은 geometry 에 넣지 않는다 — 수십만 개가 symbols.json 을 채우고,
    되짚는 데 필요한 것은 어느 변에서 어디까지 어떤 이유로 멈췄는가 뿐이다.
    지우개(--erase)는 행에 따로 실어 둔 _leaders 를 쓴다.
    """
    for r in geom_rows:
        g = r['geometry']
        if r['source'] != 'boxed' or r['status'] != 'ACCEPTED':
            continue
        if g.get('corner_style') != 'STADIUM' or (g.get('cells') or 0) < 2:
            continue
        found = leader.trace(raw, r['bbox'])
        r['_leaders'] = found
        g['leaders'] = len(found)
        if found:
            g['leader_side'] = ','.join(L['side'] for L in found)
            g['leader_length'] = max(L['length'] for L in found)
            g['leader_stop'] = ','.join(L['stop'] for L in found)
    return geom_rows


def _filter_corner(geom_rows, mode):
    """박스 태그를 끝단 모양으로 거르고, 모양별 개수를 돌려준다.

    'stadium' 은 칸막이가 있는 것만 남긴다. 캡슐형 용기(TANK)도 끝이 둥글어서
    끝단 모양만으로는 라인 클래스 태그와 갈리지 않는데, 태그는 코드를 칸으로
    나눠 담고 용기는 속이 비어 있다. boxed_tags 가 이미 세어 둔 cells 가 그
    차이를 그대로 들고 있다 — page_001 에서 cells 1 은 용기 2개와 단일 코드
    태그 1개뿐이고, 경계 태그 18개는 전부 2 다. 크기로 가르지 않는 편이 낫다:
    짧은 변 임계는 배율을 따라가야 하지만 칸막이는 어느 배율에서도 칸막이다.
    """
    boxed = [r for r in geom_rows if r['source'] == 'boxed']
    counts = dict(sorted(Counter(
        f"{r['geometry'].get('corner_style')}/{r['geometry'].get('family')}"
        for r in boxed).items()))
    if mode == 'all':
        return counts
    want = 'STADIUM' if mode == 'stadium' else 'RECT'
    for r in boxed:
        g = r['geometry']
        ok = g.get('corner_style') == want
        if want == 'STADIUM' and (g.get('cells') or 0) < 2:
            ok = False
        if not ok:
            r['status'] = 'REJECTED'
            r['reject_reason'] = 'CORNER_STYLE'
    return counts


# ---------------------------------------------------------------- 출력

def _public(rows):
    """출력용 사본 — 밑줄로 시작하는 내부 키(_leaders)를 뺀다."""
    return [{k: v for k, v in r.items() if not k.startswith('_')} for r in rows]


def write_json(out_dir, rows, stats, params, src):
    rows = _public(rows)
    payload = {'version': 'symbol_detection-v1',
               'input': {'path': str(src), 'size': stats['image_size']},
               'params': params.as_dict(), 'stats': stats, 'symbols': rows}
    with open(os.path.join(out_dir, 'symbols.json'), 'w', encoding='utf-8') as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=1)


def write_csv(out_dir, rows):
    """geometry 의 키를 전부 열로 편다 — 엑셀에서 지표별로 정렬해 보기 위함이다."""
    head = ['id', 'source', 'shape', 'status', 'reject_reason', 'score', 'score_kind',
            'x0', 'y0', 'x1', 'y1', 'width', 'height']
    extra = sorted({k for r in rows for k in r['geometry']})
    with open(os.path.join(out_dir, 'symbols.csv'), 'w', encoding='utf-8-sig', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=head + extra, restval='', extrasaction='ignore')
        w.writeheader()
        for r in rows:
            x0, y0, x1, y1 = r['bbox']
            w.writerow({**{k: r.get(k) for k in head},
                        'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
                        'width': x1 - x0, 'height': y1 - y0,
                        **{k: _flat(v) for k, v in r['geometry'].items()}})


def _flat(v):
    return json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v


def write_stats(out_dir, stats):
    with open(os.path.join(out_dir, 'stats.json'), 'w', encoding='utf-8') as fp:
        json.dump(stats, fp, ensure_ascii=False, indent=1)


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(
        description='배관 기하(v48) + 닫힌 윤곽 기하로 심볼을 검출한다.')
    ap.add_argument('input', help='도면 이미지 또는 PDF')
    ap.add_argument('out_dir', help='산출물 디렉터리 (예: out/symbol_detection)')
    ap.add_argument('--stages', action='store_true', help='v48 내부 단계 그림 4장도 쓴다')
    ap.add_argument('--no-rejected', action='store_true', help='오버레이에서 기각 상자를 뺀다')
    ap.add_argument('--no-carrier-filter', action='store_true',
                    help='배관 기하 필터를 끈다 (v48 원본 개수 확인용)')
    ap.add_argument('--min-span-strokes', type=float, default=13.0,
                    help='배관이 끊긴 길이의 하한, stroke width 배수 (기본 13.0)')
    ap.add_argument('--min-ink-strokes2', type=float, default=8.0,
                    help='끊긴 구간 안 잉크 픽셀 수의 하한, stroke width^2 배수 (기본 8.0)')
    ap.add_argument('--boxed-corner', choices=('all', 'stadium', 'rect'), default='all',
                    help='박스 태그를 끝단 모양으로 거른다. stadium 은 둥근 끝 TAG '
                         '(라인 클래스 경계), rect 는 각진 끝 (작동 로직 태그). '
                         '기본 all')
    ap.add_argument('--erase', action='store_true',
                    help='채택된 타원 태그와 거기 매달린 리더 선을 지운 도면을 '
                         'erased.png 로 쓴다')
    a = ap.parse_args()

    params = carrier.Params(min_span_strokes=a.min_span_strokes,
                            min_ink_strokes2=a.min_ink_strokes2,
                            filter_enabled=not a.no_carrier_filter)

    gray = carrier.load(a.input)
    rows, stats, ctx = run(gray, params, boxed_corner=a.boxed_corner)

    os.makedirs(a.out_dir, exist_ok=True)
    write_json(a.out_dir, rows, stats, params, a.input)
    write_csv(a.out_dir, rows)
    write_stats(a.out_dir, stats)
    cv2.imwrite(os.path.join(a.out_dir, 'overlay.png'),
                render.overlay(gray, rows, show_rejected=not a.no_rejected),
                [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if a.erase:
        erased = gray.copy()
        n = 0
        for r in rows:
            if r['status'] == 'ACCEPTED' and r.get('_leaders') is not None:
                leader.erase(erased, r['bbox'], r['_leaders'])
                n += 1
        cv2.imwrite(os.path.join(a.out_dir, 'erased.png'), erased,
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])
        print(f'지움 {n}개 태그 + 리더 → {a.out_dir}/erased.png')
    if a.stages:
        for name, im in render.carrier_stages(gray, ctx).items():
            cv2.imwrite(os.path.join(a.out_dir, name), im, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    print(f"채택 {stats['accepted_total']}  {stats['accepted_by_source']}")
    print(f"기각 {stats['rejected_total']}  {stats['rejected_by_reason']}")
    print(f"소요 {stats['seconds']}  →  {a.out_dir}")


if __name__ == '__main__':
    main()

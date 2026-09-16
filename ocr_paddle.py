"""1단계 — PaddleOCR 로 도면의 글자를 읽는다. texts.json 을 쓴다.

**이 파일만 다른 인터프리터로 돈다.** paddle 은 이 저장소 `.venv` 에 없고
`/home/rx/project/STA-main/.venv-detect` 에 있다. pipeline.py 가 그 파이썬으로
이 스크립트를 따로 띄운다.

  <detect-venv>/bin/python ocr_paddle.py <page.png> <out_dir>

입력은 **이미 렌더된 PNG** 다
  PDF 를 여는 것은 v34 인데 그 모듈은 skimage 를 물고 있고, paddle 쪽 venv 에는
  skimage 가 없다. 그래서 pipeline.py 가 페이지를 한 번 렌더해 00_page.png 로
  써 두고 그것을 넘긴다. 렌더 코드를 두 벌로 만들지 않으려는 것이다.

타일로 나눠 읽는다
  전지 한 장(6600×4859)을 통째로 넣으면 PP-OCRv5 의 검출 해상도에 맞춰 축소되어
  10px 짜리 계기 태그가 뭉개진다. 1280px 타일로 자르고 절반을 겹쳐 읽은 뒤
  겹침이 만든 중복을 IoU·포함률로 묶어 하나만 남긴다
  (digitization-of-piping-and-instrument-diagrams 의 sta_bridge/PID_pipeline_ 에서
  가져온 구조다).

세로 글자는 90도 돌려 한 번 더 읽는다
  PaddleOCR 은 `use_textline_orientation` 으로 뒤집힌 줄까지는 다루지만, 90도 누운
  글자는 검출 상자만 잡고 내용을 흘리는 일이 있다. page_001 에서 세로 상자 59개 중
  `APP-PGS-M6-001`(1.00) 처럼 잘 읽힌 것도 많지만 `GS  PLY` · `PPM001` · ` B 23A`
  처럼 뭉개진 것도 섞였다. 그래서 페이지를 시계방향 90도 돌려 같은 타일 과정을 한 번
  더 돌리고, 상자를 원래 좌표로 되돌려 합친다. 시간은 두 배가 된다.

겹침을 절반으로 두는 이유
  타일 경계에 걸친 글자는 양쪽에서 조각으로 읽힌다. 겹침이 글자 하나보다 넉넉해야
  어느 한 타일에는 온전히 들어간다. 그 대신 타일 수가 늘어 시간이 든다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ocrcache  # noqa: E402

TILE_SIZE = 1280
TILE_OVERLAP = 640
IOU_THRESHOLD = 0.20
CONTAINMENT_THRESHOLD = 0.60


def load_gray(path):
    """pipeline.py 가 써 둔 회색조 PNG 를 그대로 읽는다."""
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise SystemExit(f'ERROR: 못 읽었다: {path}')
    return g


def generate_tiles(image, tile_size, overlap):
    h, w = image.shape[:2]
    step = max(1, tile_size - overlap)

    def anchors(dim):
        pos = list(range(0, max(1, dim - tile_size + 1), step))
        last = max(0, dim - tile_size)
        if not pos or pos[-1] != last:
            pos.append(last)
        return pos

    return [{'image': image[y:y + tile_size, x:x + tile_size], 'x': x, 'y': y}
            for y in anchors(h) for x in anchors(w)]


def _iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    if xb <= xa or yb <= ya:
        return 0.0
    inter = (xb - xa) * (yb - ya)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(aa + ab - inter, 1e-9)


def _containment(small, large):
    xa, ya = max(small[0], large[0]), max(small[1], large[1])
    xb, yb = min(small[2], large[2]), min(small[3], large[3])
    if xb <= xa or yb <= ya:
        return 0.0
    area = max(0, small[2] - small[0]) * max(0, small[3] - small[1])
    return (xb - xa) * (yb - ya) / area if area else 0.0


def deduplicate(dets):
    """겹친 검출을 한 덩어리로 묶고 `점수 × 글자수` 가 가장 큰 것만 남긴다.

    점수만 보면 한 글자짜리 조각이 온전한 태그를 이긴다 — 짧을수록 확신이
    높게 나오기 때문이다. 글자수를 곱해 그것을 상쇄한다.
    """
    n = len(dets)
    nb = [[] for _ in range(n)]
    # 좌표로 먼저 추려 O(n^2) 비교를 줄인다.
    order = sorted(range(n), key=lambda i: dets[i]['bbox'][0])
    for ii, i in enumerate(order):
        bi = dets[i]['bbox']
        for j in order[ii + 1:]:
            bj = dets[j]['bbox']
            if bj[0] > bi[2]:
                break
            if (_iou(bi, bj) > IOU_THRESHOLD
                    or _containment(bi, bj) > CONTAINMENT_THRESHOLD
                    or _containment(bj, bi) > CONTAINMENT_THRESHOLD):
                nb[i].append(j)
                nb[j].append(i)
    seen, out = [False] * n, []
    for s in range(n):
        if seen[s]:
            continue
        stack, comp = [s], []
        while stack:
            k = stack.pop()
            if seen[k]:
                continue
            seen[k] = True
            comp.append(k)
            stack.extend(nb[k])
        out.append(max((dets[i] for i in comp),
                       key=lambda d: len(d['text']) * d.get('score', 1.0)))
    return out


def main():
    ap = argparse.ArgumentParser(description='PaddleOCR 로 도면 글자를 읽어 texts.json 을 쓴다.')
    ap.add_argument('input', help='pipeline.py 가 쓴 00_page.png')
    ap.add_argument('out_dir', help='texts.json 을 쓸 디렉터리')
    ap.add_argument('--tile', type=int, default=TILE_SIZE)
    ap.add_argument('--overlap', type=int, default=TILE_OVERLAP)
    ap.add_argument('--scale', type=float, default=1.0,
                    help='OCR 전에 확대할 배율. 글자가 작을 때만 올린다 (타일 수가 제곱으로 는다)')
    ap.add_argument('--min-score', type=float, default=0.0)
    ap.add_argument('--no-cache', action='store_true', help='캐시를 무시하고 다시 읽는다')
    ap.add_argument('--det-thresh', type=float, default=None,
                    help='검출 픽셀 문턱 (기본 0.3). 낮추면 흐린 글자도 잡는다')
    ap.add_argument('--det-box-thresh', type=float, default=None,
                    help='상자 채택 문턱 (기본 0.6). 낮추면 한 글자짜리도 상자가 된다')
    ap.add_argument('--det-unclip', type=float, default=None,
                    help='상자 확장 비율 (기본 1.5~2.0). 키우면 글자를 더 넉넉히 감싼다')
    ap.add_argument('--no-rotate', action='store_true',
                    help='90도 회전 패스를 건너뛴다 (시간이 절반)')
    a = ap.parse_args()

    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    ck = ocrcache.key(a.input, a.tile, a.overlap,
                      f'{a.scale}|rot={not a.no_rotate}'
                      f'|det={a.det_thresh},{a.det_box_thresh},{a.det_unclip}')
    if not a.no_cache:
        hit = ocrcache.load(ck)
        if hit is not None:
            os.makedirs(a.out_dir, exist_ok=True)
            with open(os.path.join(a.out_dir, 'texts.json'), 'w', encoding='utf-8') as fp:
                json.dump(hit, fp, ensure_ascii=False, indent=1)
            print(f"캐시 적중 {ck} — OCR 건너뜀 ({len(hit['texts'])}개)")
            return

    from paddleocr import PaddleOCR

    gray = load_gray(a.input)
    H, W = gray.shape[:2]
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if a.scale != 1.0:
        rgb = cv2.resize(rgb, None, fx=a.scale, fy=a.scale,
                         interpolation=cv2.INTER_LANCZOS4)

    # 검출 문턱을 풀 수 있게 열어 둔다. 기본값은 '글줄' 을 찾도록 맞춰져 있어
    # 주변에 아무것도 없는 **한 글자**(`D` · `H` · `V` 같은 유량 방향 표시)를 흘린다.
    # box_thresh 를 낮추면 약한 후보도 상자가 되고, unclip 을 키우면 상자가 글자를
    # 더 넉넉히 감싼다 (page_001 의 `V044` 는 상자가 `044` 만 감쌌다).
    opt = {}
    if a.det_thresh is not None:
        opt['text_det_thresh'] = a.det_thresh
    if a.det_box_thresh is not None:
        opt['text_det_box_thresh'] = a.det_box_thresh
    if a.det_unclip is not None:
        opt['text_det_unclip_ratio'] = a.det_unclip
    if opt:
        print('  검출 설정:', opt, flush=True)
    ocr = PaddleOCR(use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=True, **opt)

    def read(img, label):
        tiles = generate_tiles(img, a.tile, a.overlap)
        print(f'  [{label}] {img.shape[1]}x{img.shape[0]}  타일 {len(tiles)}장', flush=True)
        out = []
        for i, tile in enumerate(tiles, 1):
            res = ocr.predict(tile['image'])
            if not res:
                continue
            r = res[0]
            for txt, score, poly in zip(r['rec_texts'], r['rec_scores'], r['rec_polys']):
                if score < a.min_score or not str(txt).strip():
                    continue
                poly = np.asarray(poly)
                x1, y1 = int(poly[:, 0].min()), int(poly[:, 1].min())
                x2, y2 = int(poly[:, 0].max()), int(poly[:, 1].max())
                out.append({'text': str(txt), 'score': float(score), 'rot': label,
                            'bbox': [x1 + tile['x'], y1 + tile['y'],
                                     x2 + tile['x'], y2 + tile['y']]})
            if i % 10 == 0 or i == len(tiles):
                print(f'    {i}/{len(tiles)}  누적 {len(out)}', flush=True)
        return out

    t0 = time.perf_counter()
    dets = read(rgb, '0도')
    if not a.no_rotate:
        # 시계방향 90도: 돌린 그림의 (x', y') 는 원본의 (y', Hr-1-x') 다.
        # Hr 은 **돌리기 전** 높이다.
        Hr = rgb.shape[0]
        rot = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
        got = read(rot, '90도')
        for d in got:
            x1, y1, x2, y2 = d['bbox']
            d['bbox'] = [y1, Hr - 1 - x2, y2, Hr - 1 - x1]
        print(f'  90도 패스에서 {len(got)}개', flush=True)
        dets += got

    if a.scale != 1.0:
        for d in dets:
            d['bbox'] = [int(round(v / a.scale)) for v in d['bbox']]
    for d in dets:
        x1, y1, x2, y2 = d['bbox']
        d['bbox'] = [max(0, x1), max(0, y1), min(W, x2), min(H, y2)]

    raw = len(dets)
    raw_dets = [dict(d) for d in dets]
    dets = deduplicate(dets)
    heights = sorted(d['bbox'][3] - d['bbox'][1] for d in dets)
    ref_h = float(heights[len(heights) // 2]) if heights else 0.0

    os.makedirs(a.out_dir, exist_ok=True)
    # 중복 제거 **전** 목록도 남긴다. 중복 제거는 글자 목록을 정리하려는 것이지
    # 지우개 마스크를 줄이려는 것이 아니다 — 한 글줄의 왼쪽 절반과 오른쪽 절반이
    # 서로 다른 타일에서 잡히면 둘이 겹쳐 '같은 영역'으로 묶이고 한쪽이 버려진다.
    # page_001 의 3번 주석이 그래서 절반만 지워졌다. 마스크는 raw 를 쓴다.
    payload = {'input': str(a.input), 'size': [W, H],
               'params': {'tile': a.tile, 'overlap': a.overlap, 'scale': a.scale},
               'ref_h': ref_h, 'raw': raw, 'texts': dets, 'texts_raw': raw_dets}
    with open(os.path.join(a.out_dir, 'texts.json'), 'w', encoding='utf-8') as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=1)
    ocrcache.save(ck, payload)
    print(f'OCR {raw} → 중복 제거 {len(dets)}  ref_h={ref_h:.0f}px  '
          f'{time.perf_counter()-t0:.0f}s  → {a.out_dir}/texts.json')


if __name__ == '__main__':
    main()

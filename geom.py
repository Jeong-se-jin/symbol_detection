"""이진화와 겹침 판정 — pkg9/shape_geom.py 에서 쓰는 것만 발췌했다.

가져온 것
  binarize   shape_geom.binarize_otsu (이름만 줄였다)
  iom        shape_geom.iom
  iom_nms    shape_geom.iom_nms

가져오지 않은 것
  convex_hull_mask · transform · dedup_key · roi_iou_cv2 는 전부 템플릿 정합
  전용이다. 이 폴더는 템플릿을 쓰지 않으므로 따라올 이유가 없다.

binarize 와 ap1000 의 v34.binarize 는 같은 잉크맵을 만든다
  둘 다 THRESH_BINARY_INV+OTSU 다. page_001(6600x4859)에서 불일치 픽셀 0 으로
  확인했다 (v34 는 bool, 이쪽은 0/1 uint8). 그래서 detect.py 가 잉크맵을 한 번만
  만들어 두 갈래에 함께 내려보낸다 — 두 번 이진화하면 같은 값을 두 번 계산하는
  낭비이면서, 언젠가 한쪽만 바뀌면 좌표가 조용히 어긋난다.

  단 v34.binarize 는 고유값이 4개 이하일 때 중간값으로 자르는 갈래가 따로 있다
  (이미 1bit 로 저장된 도면). 그 경우까지 같으려면 detect.py 가 v34 쪽을 기준으로
  삼아야 하므로, 실제 이진화는 v34 가 하고 이 함수는 단독 실행 · 시험용으로 남긴다.
"""
from __future__ import annotations

import cv2
import numpy as np


def binarize(gray: np.ndarray) -> np.ndarray:
    """잉크=1, 배경=0 인 uint8 맵."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (b > 0).astype(np.uint8)


def iom(a, b):
    """교집합 / 작은 쪽 넓이. bbox 는 배타 구간 [x0,y0,x1,y1) 이다."""
    ax0, ay0, ax1, ay1 = a[:4]; bx0, by0, bx1, by1 = b[:4]
    iw = max(0, min(ax1, bx1) - max(ax0, bx0)); ih = max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    aa = max(1, (ax1 - ax0) * (ay1 - ay0)); bb = max(1, (bx1 - bx0) * (by1 - by0))
    return inter / max(1, min(aa, bb))


def iom_nms(rows: list[dict], thr: float) -> list[dict]:
    """점수 내림차순으로 훑으며 IoM 이 thr 을 넘는 뒤 후보를 지운다.

    격자는 O(N^2) 를 줄이기만 하고 판정 자체는 바꾸지 않는다.
    """
    rows = sorted(rows, key=lambda r: r['score'], reverse=True)
    keep = []
    cell = 64; grid = {}
    for r in rows:
        x0, y0, x1, y1 = r['bbox']; gx0 = x0 // cell; gx1 = x1 // cell; gy0 = y0 // cell; gy1 = y1 // cell
        candidates = []; seen = set()
        for gy in range(gy0 - 1, gy1 + 2):
            for gx in range(gx0 - 1, gx1 + 2):
                for j in grid.get((gx, gy), []):
                    if j not in seen:
                        seen.add(j); candidates.append(j)
        dup = False
        for j in candidates:
            if iom(r['bbox'], keep[j]['bbox']) > thr:
                dup = True; break
        if dup:
            continue
        j = len(keep); keep.append(r)
        for gy in range(gy0, gy1 + 1):
            for gx in range(gx0, gx1 + 1):
                grid.setdefault((gx, gy), []).append(j)
    return keep

"""타원 태그에 매달린 리더 선 — 태그에서 출발해 분기 직전까지 따라간다.

무엇을 위한 것인가
  라인 클래스 경계 태그(ECE|ECB …)는 배관 옆에 떠 있고 1px 리더 선 하나로 배관에
  묶인다. 태그만 지우면 이 선이 배관에 붙은 짧은 가지로 남아 배관 추적이 그것을
  스텁으로 읽는다. 태그와 선은 한 몸이라 함께 지워야 한다.

어디서 끝내는가 — 꺾임이 아니라 분기다
  "다른 방향으로 확장되는 부분 이전까지" 를 **골격 경로의 갈래 수**로 잰다.
  선을 1px 골격으로 얇게 만든 뒤 한 픽셀씩 걸으며 다음 갈래를 센다:

    갈래 1개  선이 이어진다 (곧든 꺾이든) → 계속 간다
    갈래 0개  선이 끝났다                 → END
    갈래 2개+ 여기서 다른 방향이 갈라진다 → JUNCTION, **그 자리를 포함하지 않고** 멈춘다

  축을 따라가며 단면 폭만 보는 방법을 먼저 썼다가 버렸다. 그 방법은 page_001 에서
  두 가지를 못 가린다:

    B0006  태그에서 수평으로 14px 나간 뒤 45° 로 꺾여 내려가는 진짜 리더다.
           출발점에서 벗어난 거리로 자르면 꺾이는 자리에서 잘려 대각선이 남는다.
    B0022  수직 리더가 체크밸브에 닿은 뒤 삼각형 빗변으로 갈아탄다. 빗변도 1~2px 라
           폭으로는 안 잡히고, 지우개가 밸브를 같이 지운다.

  둘 다 "방향이 갑자기 바뀐다" 는 점에서는 같다. 다른 것은 **바뀌는 자리에 갈래가
  몇 개냐** 다 — B0006 의 꺾임은 들어온 길과 나갈 길뿐(차수 2)이고, B0022 가 밸브를
  만나는 자리는 직진하는 배관까지 셋이다(차수 3). 그래서 차수로 가른다.

갈래 수는 교차수로 센다
  이웃을 뭉쳐 세는 방법은 T 접합에서 깨진다. 수평선에 아래로 스템이 붙은 자리의
  이웃은 좌·우·아래인데, 아래(y+1,x)와 좌(y,x-1)가 서로 8-이웃이라 한 덩어리로
  뭉쳐 버린다. 그래서 8-이웃 고리를 한 바퀴 돌며 0→1 로 바뀌는 횟수(교차수)를 센다:

    끝점 1 · 지나가는 자리 2 · 분기 3 이상

  90° 꺾임도 2 로 나온다 (고리에서 두 이웃이 떨어져 있다). 꺾임은 통과하고 분기만
  잡는다는 요구가 이 한 수로 그대로 성립한다.

가운데서 나온 선만 리더로 본다
  리더는 태그 칸막이의 연장이라 변의 한가운데에서 나간다. 변 어디에 붙은 잉크든
  따라가면 옆을 지나던 남의 선을 리더로 오인한다. 그래서 씨앗을 변의 중점 둘레로
  한정한다.

가짜 스텁을 거른다
  태그 안 칸막이 벽 끝이 bbox 경계에서 1~2px 튀어나와 보이는 일이 있다
  (page_001 의 B0008 아래쪽). 최소 길이를 걸어 걸러낸다 — 진짜 리더는 수십 px 다.
"""
from __future__ import annotations

import numpy as np
from skimage.morphology import skeletonize

#: 태그 둘레로 잘라낼 작업창 반경 (px). 관측된 가장 긴 리더가 225px 다.
MARGIN = 520

#: 진짜 리더로 볼 최소 길이 (px). 칸막이 벽 끝이 만드는 1~2px 스텁을 거른다.
MIN_LEN = 8

#: 한 리더를 따라갈 최대 골격 픽셀 수. 도면 전체로 새는 것을 막는 안전장치다.
MAX_STEPS = 1500

#: 지울 때 골격 경로를 부풀릴 반경 (px). 선이 1~2px 라 이만큼이면 덮는다.
DILATE = 2

#: 씨앗이 변의 중점에서 벗어나도 되는 거리 — 변 길이 대비 비율과 절대 하한.
CENTER_FRAC = 0.15
CENTER_MIN = 4

_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

#: 교차수를 셀 8-이웃 고리. 시계 방향으로 이어져야 한다.
_RING = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def _crossing(sk, y, x):
    """8-이웃 고리를 돌며 0→1 로 바뀐 횟수. 끝점 1 · 지나가는 자리 2 · 분기 3+."""
    h, w = sk.shape
    ring = [1 if (0 <= y + dy < h and 0 <= x + dx < w and sk[y + dy, x + dx]) else 0
            for dy, dx in _RING]
    return sum(1 for i in range(8) if ring[i - 1] == 0 and ring[i] == 1)


def _walk(sk, start, max_steps):
    """골격 위를 한 픽셀씩 간다. 분기 픽셀은 밟지 않는다. (경로, 멈춘 이유)."""
    h, w = sk.shape
    path, visited = [start], {start}
    cur, stop = start, 'MAX_STEPS'
    for _ in range(max_steps):
        y, x = cur
        nbrs = [(y + dy, x + dx) for dy, dx in _N8
                if 0 <= y + dy < h and 0 <= x + dx < w
                and sk[y + dy, x + dx] and (y + dy, x + dx) not in visited]
        if not nbrs:
            stop = 'END_OF_LINE'
            break
        # 직교 이웃을 먼저 본다 — 골격이 대각으로 한 칸 새는 자리에서 경로가
        # 톱니처럼 길어지는 것을 막는다.
        nxt = min(nbrs, key=lambda q: abs(q[0] - y) + abs(q[1] - x))
        if _crossing(sk, *nxt) > 2:
            stop = 'JUNCTION'
            break
        path.append(nxt)
        visited.add(nxt)
        cur = nxt
    return path, stop


def trace(raw, bbox, margin=MARGIN, min_len=MIN_LEN, max_steps=MAX_STEPS,
          center_frac=CENTER_FRAC, center_min=CENTER_MIN):
    """bbox 네 변에서 나가는 리더를 모두 따라간다.

    돌려주는 것은 [{side, length, steps, stop, pixels}] 다. pixels 는 원본 좌표의
    골격 경로이고, 지울 때 erase 가 부풀린다. 길이가 min_len 에 못 미치는 것은
    스텁으로 보고 버린다.
    """
    x0, y0, x1, y1 = bbox
    h, w = raw.shape
    wx0, wy0 = max(0, x0 - margin), max(0, y0 - margin)
    wx1, wy1 = min(w, x1 + margin), min(h, y1 + margin)
    sub = raw[wy0:wy1, wx0:wx1].copy()
    bx0, by0, bx1, by1 = x0 - wx0, y0 - wy0, x1 - wx0, y1 - wy0
    # 태그 자체를 지우고 골격을 뜬다. 남겨 두면 윤곽이 골격에 들어와 출발점
    # 바로 옆이 분기가 되고, 리더가 한 픽셀도 못 나간다.
    sub[by0:by1, bx0:bx1] = 0
    sk = skeletonize(sub.astype(bool))

    out = []
    for side in ('top', 'bottom', 'left', 'right'):
        if side in ('top', 'bottom'):
            p = by1 if side == 'bottom' else by0 - 1
            if not 0 <= p < sk.shape[0]:
                continue
            lo, hi = bx0, bx1
            seeds = [(p, lo + int(q)) for q in np.nonzero(sk[p, lo:hi])[0]]
            pos = lambda q: q[1]
        else:
            p = bx1 if side == 'right' else bx0 - 1
            if not 0 <= p < sk.shape[1]:
                continue
            lo, hi = by0, by1
            seeds = [(lo + int(q), p) for q in np.nonzero(sk[lo:hi, p])[0]]
            pos = lambda q: q[0]
        if not seeds:
            continue
        # 가운데서 나온 선만 리더다. 변의 중점에서 벗어난 씨앗은 옆을 지나던
        # 남의 선이거나 윤곽이 스친 자국이다.
        mid = (lo + hi) / 2.0
        tol = max(center_min, round((hi - lo) * center_frac))
        seeds = [q for q in seeds if abs(pos(q) - mid) <= tol]
        if not seeds:
            continue
        path, stop = _walk(sk, min(seeds, key=lambda q: abs(pos(q) - mid)), max_steps)
        ys = [q[0] for q in path]
        xs = [q[1] for q in path]
        length = max(max(ys) - min(ys), max(xs) - min(xs)) + 1
        if length < min_len:
            continue
        out.append({'side': side, 'length': int(length), 'steps': len(path),
                    'stop': stop, 'offset': int(round(pos(path[0]) - mid)),
                    'pixels': [(int(y + wy0), int(x + wx0)) for y, x in path]})
    return out


def erase(img, bbox, leaders, pad=1, dilate=DILATE, value=255):
    """bbox 와 리더 경로를 제자리에서 지운다. 배경값은 value 다.

    bbox 는 통째로 비운다 — 태그 윤곽과 속 글자와 칸막이를 따로 셀 이유가 없고,
    이 태그들은 배관에서 떨어져 떠 있어서 사각형 안에 남길 잉크가 없다.
    리더는 골격이라 dilate 만큼 부풀려 원래 선 두께를 덮는다.
    """
    x0, y0, x1, y1 = bbox
    h, w = img.shape
    img[max(0, y0 - pad):min(h, y1 + pad),
        max(0, x0 - pad):min(w, x1 + pad)] = value
    for L in leaders:
        for (y, x) in L['pixels']:
            img[max(0, y - dilate):min(h, y + dilate + 1),
                max(0, x - dilate):min(w, x + dilate + 1)] = value
    return img

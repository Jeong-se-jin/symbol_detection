"""2패스 — 방향장이 못 덮은 잉크를 골격 경로로 줍는다.

왜 2패스인가
  1패스(lines.py 방향장)는 긴 배관과 축 정렬 선을 잘 잡지만, **프로브(25px 자)보다
  짧은 조각**에서 무너진다. 자를 어디에 대도 절반이 조각 밖으로 나가 support 가
  0.45 아래로 떨어지고, 픽셀이 '방향 불명' 으로 버려진다. page_001 의 14px 점선 한
  칸이 그랬다 — 잉크 16px 중 12px 이 그 자리에서 탈락했다.

  그렇다고 프로브를 줄이면 긴 선에서 방향이 잡음에 흔들려 run 이 잘게 쪼개진다.
  전면 교체가 아니라 **1패스가 못 덮은 자리에만** 다른 자를 대는 것이 값싸다.

왜 골격인가
  골격 걷기는 자를 쓰지 않는다. 한 픽셀씩 이웃을 따라가므로 조각이 짧아도 상관없고,
  꺾임은 교차수 2 라 그냥 지나가고 분기(3 이상)에서만 끊는다. ㄱ자든 곡선이든 한
  경로로 나온다 — 한 선이 구부러진 것을 직선 둘로 나눌 이유가 없다.

  1패스에 골격을 쓰지 않는 이유는 lines.py 가 적어 둔 그대로다: 세선화는 긴 대각선을
  계단으로 부수고 교차마다 잔가지를 만든다. 잔여(전체의 2% 남짓)에만 쓰면 그 대가를
  치르지 않는다.

선과 호를 가른다
  골격 걷기는 "분기에서 멈춘다" 만 알지 이것이 선인지 고리인지는 모른다. 배관 위에
  얹힌 작은 원(지름 20px)이 어느 검출기에도 안 걸려 잔여로 내려오면, 그 원의 호가
  경로로 주워진다 — page_001 에서 가장 긴 잔여 경로 넷 중 둘이 그것이었다.

  두 가지로 가른다.

    straightness  끝점 사이 직선거리 / 경로 길이. 직선 1.0, 얕은 ㄱ자 .92, 호 .73~.77
    한 자리 꺾임   최대 꺾임각 / 총 꺾임각. 호는 여러 자리에서 조금씩 돌아 .1~.3,
                  직각으로 꺾인 선은 한 자리가 전부라 .5 를 넘는다

  straightness 만 쓰면 진짜 90도 꺾인 리더(.71)까지 버리므로, **둘 중 하나만 만족해도
  남긴다**. 곧은 것이거나, 굽었더라도 꺾인 자리가 하나면 선이다.

무엇을 돌려주는가
  폴리라인이다. 꺾인 선을 두 점으로 줄이면 그 꺾임이 사라지므로 점 목록을 그대로
  들고, 접촉 판정에는 양 끝만 쓴다.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from skimage.morphology import skeletonize

import leader

#: 회수할 최소 경로 길이 (골격 픽셀 수). 이보다 짧으면 얼룩이다.
MIN_PATH = 8

#: 이보다 곧으면 굽은 정도를 더 따지지 않는다.
MIN_STRAIGHT = 0.85

#: 총 꺾임 중 한 자리가 이 비율을 넘으면 '한 번 꺾인 선' 으로 본다.
CORNER_SHARE = 0.5

#: 끝점이 이만큼 가까우면 닫힌 고리다 (경로 길이 대비).
LOOP_RATIO = 0.25

#: 1패스 run 을 덮은 것으로 칠 때 쓰는 선 두께. 실제 획보다 넉넉해야
#: '덮었는데 못 덮었다' 고 오인하지 않는다.
COVER_WIDTH = 7


def residual_mask(ink, runs, cover_width=COVER_WIDTH):
    """1패스 run 이 설명하지 못한 잉크."""
    cov = np.zeros(ink.shape, np.uint8)
    for r in runs.values() if isinstance(runs, dict) else runs:
        cv2.line(cov, (int(r['a'][0]), int(r['a'][1])),
                 (int(r['b'][0]), int(r['b'][1])), 1, cover_width)
    return (ink > 0) & (cov == 0)


def _paths(sk, min_path):
    """골격을 끝점·분기에서 끊어 경로 목록으로. 각 경로는 (y, x) 목록이다."""
    h, w = sk.shape
    ys, xs = np.nonzero(sk)
    deg = {}
    for y, x in zip(ys.tolist(), xs.tolist()):
        deg[(y, x)] = leader._crossing(sk, y, x)
    seeds = [p for p, d in deg.items() if d != 2]
    seen = set()
    out = []
    for sy, sx in seeds:
        for dy, dx in leader._N8:
            ny, nx = sy + dy, sx + dx
            if not (0 <= ny < h and 0 <= nx < w) or not sk[ny, nx]:
                continue
            if (ny, nx) in seen:
                continue
            path = [(sy, sx)]
            cur, prev = (ny, nx), (sy, sx)
            while True:
                path.append(cur)
                seen.add(cur)
                if deg.get(cur, 0) != 2:
                    break
                nxt = None
                for ddy, ddx in leader._N8:
                    q = (cur[0] + ddy, cur[1] + ddx)
                    if (0 <= q[0] < h and 0 <= q[1] < w and sk[q[0], q[1]]
                            and q != prev and q not in seen):
                        nxt = q
                        break
                if nxt is None:
                    break
                prev, cur = cur, nxt
            if len(path) >= min_path:
                out.append(path)
    return out


def _shape(points):
    """(straightness, 한 자리 꺾임 비율). 점이 모자라면 (1.0, 1.0) 로 통과시킨다."""
    if len(points) < 5:
        return 1.0, 1.0
    ax, ay = points[0]
    bx, by = points[-1]
    span = math.hypot(bx - ax, by - ay)
    L = sum(math.hypot(points[k][0] - points[k - 1][0],
                       points[k][1] - points[k - 1][1])
            for k in range(1, len(points)))
    straight = span / L if L else 1.0
    step = max(2, len(points) // 12)
    S = points[::step]
    if len(S) < 3:
        return straight, 1.0
    turns = []
    for i in range(1, len(S) - 1):
        a1 = math.atan2(S[i][1] - S[i - 1][1], S[i][0] - S[i - 1][0])
        a2 = math.atan2(S[i + 1][1] - S[i][1], S[i + 1][0] - S[i][0])
        t = abs(math.degrees(a2 - a1))
        turns.append(min(t, 360 - t))
    total = sum(turns)
    return straight, (max(turns) / total if total > 1e-6 else 1.0)


def is_line(points, min_straight=MIN_STRAIGHT, corner_share=CORNER_SHARE,
            loop_ratio=LOOP_RATIO):
    """선인가 호인가. 곧거나, 굽었어도 꺾인 자리가 하나면 선이다."""
    straight, share = _shape(points)
    if straight < loop_ratio:
        return False, straight, share          # 끝이 제자리로 돌아왔다 — 고리
    return (straight >= min_straight or share >= corner_share), straight, share


def recover(ink, runs, start_id=1, min_path=MIN_PATH, cover_width=COVER_WIDTH,
            min_straight=MIN_STRAIGHT, corner_share=CORNER_SHARE,
            loop_ratio=LOOP_RATIO):
    """잔여 잉크에서 주운 폴리라인들. 1패스 run 과 같은 스키마에 points 를 더한다."""
    res = residual_mask(ink, runs, cover_width)
    if not res.any():
        return [], {'residual_px': 0, 'paths': 0}
    sk = skeletonize(res)
    paths = _paths(sk, min_path)
    out, arcs = [], 0
    i = start_id
    for p in paths:
        pts = [(q[1], q[0]) for q in p]        # (x, y)
        ok, straight, share = is_line(pts, min_straight, corner_share, loop_ratio)
        if not ok:
            arcs += 1
            continue
        ys = [q[0] for q in p]
        xs = [q[1] for q in p]
        ax, ay, bx, by = float(xs[0]), float(ys[0]), float(xs[-1]), float(ys[-1])
        # 길이는 경로를 따라간 거리다. 끝점 사이 직선 거리로 재면 꺾인 선이
        # 실제보다 짧게 나오고, 그것이 곧 이 갈래가 잡으려는 대상이다.
        L = sum(math.hypot(xs[k] - xs[k - 1], ys[k] - ys[k - 1])
                for k in range(1, len(p)))
        ang = math.degrees(math.atan2(by - ay, bx - ax)) % 180
        out.append({'id': i, 'length': round(L, 1), 'pixels': len(p),
                    'angle': round(ang, 1),
                    'a': [round(ax, 1), round(ay, 1)], 'b': [round(bx, 1), round(by, 1)],
                    'orient': 'h' if min(ang, 180 - ang) <= 45 else 'v',
                    'line_type': 'solid', 'from': 'residual',
                    'straightness': round(straight, 2),
                    'corner_share': round(share, 2),
                    'points': [[int(x), int(y)] for x, y in zip(xs, ys)]})
        i += 1
    return out, {'residual_px': int(res.sum()), 'skeleton_px': int(sk.sum()),
                 'paths': len(out), 'arcs_dropped': arcs}

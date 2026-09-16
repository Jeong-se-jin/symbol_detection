"""Solid pipe or dashed signal line, measured along each run.

A dashed line on this drawing is an instrument signal, not piping, and the
orientation field cannot tell the difference -- it groups the dashes into one
run exactly as it groups a solid stroke, which is what we want for finding the
line but leaves the question of what kind it is.

The method is the one in src/app/services/line_detection/line_type_classifier,
applied per run instead of per collinear group of segments, since a run already
is the group. Two things make it work, and both are borrowed intact:

* read the ink before thinning. A thinned stroke wobbles one pixel wide and a
  solid line then measures as broken -- 5px gaps on a line that has none.
* judge on regularity, not on the presence of gaps. A solid pipe punched
  through by symbol masking has gaps too; on this sheet one had gaps of
  15/18/16px, a convincing rhythm, but its dashes ran 339/139/86/430px. The
  coefficient of variation of the dash lengths is what separates them, and both
  coefficients have to be checked.
"""
import math
import statistics

import numpy as np

SOLID = 'solid'
DASHED = 'dashed'


def _cv(values):
    if len(values) < 2:
        return 0.0
    mean = float(np.mean(values))
    if mean == 0:
        return float('inf')
    return float(np.std(values)) / mean


def _encode(profile):
    """Run-length encode a boolean profile into (is_ink, start, length)."""
    out, start = [], 0
    for i in range(1, len(profile) + 1):
        if i == len(profile) or profile[i] != profile[start]:
            out.append((bool(profile[start]), start, i - start))
            start = i
    return out


def sample(ink, run, band):
    """run 을 따라간 잉크 프로파일. 선 둘레 좁은 띠의 잉크를 축에 투영해 만든다.

    **여기만 우리 쪽에서 바꿨다. 나머지는 이식분 그대로다.**

    원래는 축 위를 한 칸씩 걸으며 그 점의 잉크를 읽었다. 축 정렬 선에서는 걷는 경로가
    픽셀 열과 정확히 겹쳐 문제가 없지만, **대각선에서는 이산화된 선과 경로가 어긋나
    중간중간 잉크를 놓친다.** page_001 의 대각선 점선이 그랬다:

        수평 점선  잉크4 · 간격8 · 잉크20 · 간격8 · 잉크20        → gaps CV 0.00  dashed
        대각 점선  잉크3·간격1·잉크1·간격1·잉크10·간격1·잉크3·간격9·…
                   진짜 간격 8,9 가 목록에 있는데 1px 가짜가 개수로 압도해
                   중앙값이 1 이 되고 CV 가 1.23 으로 튄다              → solid

    좌우 오프셋을 몇 칸 시도해 가장 잉크가 많은 쪽을 쓰는 원래 보정은 선 전체가 반
    픽셀 밀린 경우를 위한 것이라, 계단 모양은 메우지 못한다.

    그래서 축을 걷는 대신 **선 둘레 ±band 안의 잉크 픽셀을 전부 모아 축에 투영**한다.
    계단의 어느 칸에 있든 같은 축 좌표로 떨어지므로 어긋남이 사라진다.

    투영할 때 픽셀 하나를 눈금 하나에 찍으면 안 된다. 대각선은 **픽셀 수가 축 길이보다
    적기** 때문이다 — 45° 면 픽셀이 축 위에서 √2 간격으로 놓여 눈금의 29% 가 빈 채로
    남고, 그 빈칸이 다시 1px 가짜 간격이 된다. 픽셀 하나가 실제로 덮는 축 구간은
    `1 / max(|ux|, |uy|)` (축 정렬 1.0, 45° 1.41) 이므로 그 폭만큼 칠한다.
    """
    (ax, ay), (bx, by) = run['a'], run['b']
    length = math.hypot(bx - ax, by - ay)
    n = int(round(length))
    if n < 4:
        return None
    ux, uy = (bx - ax) / length, (by - ay) / length
    H, W = ink.shape[:2]
    x0 = max(0, int(min(ax, bx)) - band - 1)
    x1 = min(W, int(max(ax, bx)) + band + 2)
    y0 = max(0, int(min(ay, by)) - band - 1)
    y1 = min(H, int(max(ay, by)) + band + 2)
    if x1 <= x0 or y1 <= y0:
        return None
    ys, xs = np.nonzero(ink[y0:y1, x0:x1])
    if ys.size == 0:
        return None
    px = xs + x0 - ax
    py = ys + y0 - ay
    t = px * ux + py * uy                 # 축 위 위치
    d = np.abs(-px * uy + py * ux)        # 축에서 떨어진 거리
    keep = (d <= band) & (t >= 0) & (t <= n)
    prof = np.zeros(n + 1, bool)
    tk = t[keep]
    if tk.size:
        w = 1.0 / max(abs(ux), abs(uy))      # 픽셀 하나가 덮는 축 구간
        lo = np.floor(tk - w / 2).astype(int)
        hi = np.floor(tk + w / 2).astype(int)
        prof[np.clip(lo, 0, n)] = True
        prof[np.clip(hi, 0, n)] = True
    # 1px 구멍은 메운다. 가장 짧은 진짜 간격(gap_min)도 2px 을 넘으므로 한 칸짜리
    # 구멍은 정의상 간격이 아니라 투영이 남긴 자국이다. 남겨 두면 개수로 진짜
    # 간격을 압도해 중앙값과 CV 를 망친다.
    hole = ~prof[1:-1] & prof[:-2] & prof[2:]
    prof[1:-1] |= hole
    return prof


def classify(ink, runs, params, band=3, min_gaps=2, clip_ratio=1.0):
    """Tag every run solid or dashed, in place."""
    h = getattr(params, 'draw_h', 0) or params.ref_h
    limits = {
        'gap_min': max(0.14 * h, 2.0), 'gap_max': 1.8 * h,
        'dash_min': max(0.28 * h, 3.0), 'dash_max': 4.3 * h,
        'gap_cv_max': 0.45, 'dash_cv_max': 0.45,
        'min_gap_count': min_gaps, 'break_on_blank': 1.8 * h,
        # 간격 하나짜리에서 양끝 잉크가 서로 얼마나 달라도 되는가 (긴쪽/짧은쪽 - 1)
        'clip_ratio': clip_ratio,
    }
    counts = {SOLID: 0, DASHED: 0}
    for run in runs.values():
        run['line_type'] = SOLID
        profile = sample(ink, run, band)
        if profile is None:
            counts[SOLID] += 1
            continue
        pieces = _encode(profile)
        # Cut at any blank longer than a dash gap could be: that is a symbol
        # mask hole or the end of the line, not part of the rhythm.
        chunks, current = [], []
        for is_ink, start, length in pieces:
            if not is_ink and length > limits['break_on_blank']:
                if current:
                    chunks.append(current)
                current = []
            else:
                current.append((is_ink, start, length))
        if current:
            chunks.append(current)

        for chunk in chunks:
            # The first and last pieces are clipped by where the run ends, so
            # their lengths say nothing.
            inner = chunk[1:-1] if len(chunk) > 2 else []
            gaps = [ln for is_ink, _, ln in inner if not is_ink]
            dashes = [ln for is_ink, _, ln in inner if is_ink]
            # 간격이 하나뿐인 짧은 점선은 위 규칙으로는 영영 판정되지 않는다.
            # 조각이 [잉크, 간격, 잉크] 셋뿐이라 첫·끝을 버리고 나면 잉크가 하나도
            # 남지 않고, `not dashes` 에서 탈락한다. page_001 의 H/L 신호선이 그렇다
            # (잉크 21 · 간격 9 · 잉크 21 인데 solid 로 나온다). 그래서 이 경우에만
            # 잘린 양끝 잉크를 되살린다 — 길이 범위 확인에는 쓰되, 규칙성(CV)
            # 계산에서는 뺀다. 잘린 길이는 리듬을 말해 주지 못하기 때문이다.
            clipped = False
            if (len(gaps) == 1 and not dashes and len(chunk) == 3
                    and chunk[0][0] and chunk[2][0]):
                lo, hi = sorted((chunk[0][2], chunk[2][2]))
                # 양끝이 서로 비슷해야 한다. 잘렸어도 같은 획에서 잘린 것이므로
                # 한쪽만 1px 인 것은 점선이 아니라 실선 끝의 잉크 파편이다
                # (page_001 의 run 3205: 잉크 78 · 간격 4 · 잉크 1 인 실선 배관).
                if lo >= limits['dash_min'] and hi <= lo * (1 + limits['clip_ratio']):
                    dashes = [chunk[0][2], chunk[2][2]]
                    clipped = True
            if len(gaps) < limits['min_gap_count'] or not dashes:
                continue
            g, d = statistics.median(gaps), statistics.median(dashes)
            if not (limits['gap_min'] <= g <= limits['gap_max']):
                continue
            if not (limits['dash_min'] <= d <= limits['dash_max']):
                continue
            if _cv(gaps) >= limits['gap_cv_max']:
                continue
            if not clipped and _cv(dashes) >= limits['dash_cv_max']:
                continue
            run['line_type'] = DASHED
            run['dash'] = {'dash_px': round(float(d), 1),
                           'gap_px': round(float(g), 1),
                           'period_px': round(float(d + g), 1),
                           'from_clipped_ends': clipped}
            break
        counts[run['line_type']] += 1
    return counts, limits

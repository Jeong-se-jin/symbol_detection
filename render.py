"""확인용 그림 — 결과를 눈으로 검수하기 위한 것이고 판정에는 관여하지 않는다.

기각된 것도 그린다
  옅은 회색으로 함께 그린다. 필터가 무엇을 지웠는지 보이지 않으면 임계가 맞는지
  판단할 수 없고, "안 보이니까 없다" 로 읽히는 것이 가장 나쁘다.

단계 그림은 v48 것을 그대로 부른다
  draw_raw / draw_witnesses / draw_proposals / draw_final 은 ap1000 원본에 있다.
  다시 그리면 원본과 어긋날 수 있으므로 호출만 한다.
"""
from __future__ import annotations

import cv2
import numpy as np

#: 출처별 색 (BGR). 기각은 출처와 무관하게 회색이다.
COLORS = {
    'carrier':   (255, 0, 255),
    'frame':     (0, 170, 0),
    'ring':      (220, 0, 0),
    'connector': (0, 140, 255),
    'boxed':     (200, 160, 0),
}
REJECTED = (185, 185, 185)


def overlay(gray, rows, show_rejected=True):
    """도면 위에 상자를 얹은 BGR 이미지."""
    im = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # 기각을 먼저 깔아서 채택된 상자가 위로 오게 한다.
    if show_rejected:
        for r in rows:
            if r['status'] == 'ACCEPTED':
                continue
            x0, y0, x1, y1 = r['bbox']
            cv2.rectangle(im, (x0, y0), (x1 - 1, y1 - 1), REJECTED, 1)
    for r in rows:
        if r['status'] != 'ACCEPTED':
            continue
        x0, y0, x1, y1 = r['bbox']
        cv2.rectangle(im, (x0, y0), (x1 - 1, y1 - 1), COLORS[r['source']], 2)
    _legend(im, rows, show_rejected)
    return im


def _legend(im, rows, show_rejected):
    """왼쪽 위에 출처별 색과 개수. 도면 잉크를 가리지 않게 흰 바탕을 깐다."""
    counts = {s: sum(1 for r in rows if r['source'] == s and r['status'] == 'ACCEPTED')
              for s in COLORS}
    lines = [(s, f'{s:<10s} {counts[s]:4d}', COLORS[s]) for s in COLORS]
    if show_rejected:
        n = sum(1 for r in rows if r['status'] != 'ACCEPTED')
        lines.append((None, f'{"rejected":<10s} {n:4d}', REJECTED))
    pad, lh, bw = 14, 34, 330
    h = pad * 2 + lh * len(lines)
    cv2.rectangle(im, (0, 0), (bw, h), (255, 255, 255), -1)
    cv2.rectangle(im, (0, 0), (bw, h), (0, 0, 0), 1)
    for i, (_, text, color) in enumerate(lines):
        y = pad + lh * i + 8
        cv2.rectangle(im, (pad, y), (pad + 22, y + 18), color, -1)
        cv2.putText(im, text, (pad + 32, y + 16), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1,
                    cv2.LINE_AA)
    return im


def carrier_stages(gray, ctx):
    """v48 내부 단계 4장. {파일이름: 이미지} 로 돌려준다."""
    import carrier as _c
    v48 = _c.v48
    rs, active = ctx['rs'], ctx['active']
    return {
        'carrier_01_raw_hv_runs.png': v48.draw_raw(gray, rs, active),
        'carrier_02_lane_witnesses.png': v48.draw_witnesses(gray, ctx['groups'], ctx['eligible'], rs),
        'carrier_03_displacements.png': v48.draw_proposals(gray, ctx['proposals'], rs),
        'carrier_04_accepted_bbox.png': v48.draw_final(gray, ctx['cands'], ctx['accepted'], rs),
    }

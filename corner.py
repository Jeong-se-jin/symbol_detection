"""박스 태그의 끝단 모양 — 둥근 캡(스타디움)인가 각진 벽인가.

왜 따로 재는가
  boxed_tags 는 닫힌 윤곽 · 내부 잉크 · 장단비 세 가지만 잰다. 끝이 둥근지
  각진지는 측정 항목에 없어서 두 갈래가 같은 elongated_box 로 나온다.
  page_001 에서 둘은 용도가 다르다:

    둥근 끝  ECE|ECB · BTA|BBC   배관 위에 얹혀 라인 클래스가 갈리는 지점을 찍는다.
                                 그래프에서는 배관을 자르는 분할점이다.
    각진 끝  PMS|OP|CMT · PLS|OP|S  점선 신호선으로 밸브 액추에이터에 묶인다.
                                 그래프에서는 밸브에 붙는 제어 태그다.

  boxed.py 는 pkg9 이식본이라 본문을 고치지 않는다. 그래서 검출을 끝낸 bbox 를
  다시 재는 별도 갈래로 둔다.

무엇을 재는가
  장축을 따라간 단면 높이 프로파일에서 끝단이 평지(90%)를 깨고 좁아지기 시작하는
  길이를 짧은 변으로 나눈다. connector_flags 의 _end_taper 와 같은 착상이되
  **직선 테이퍼를 요구하지 않는다** — 둥근 캡은 호이지 직선이 아니고, 커넥터의
  오각형 팁과 갈라야 하는 것도 아니기 때문이다.

  반지름 r 인 반원 캡의 프로파일은 hs(d) = sqrt(1 - (1 - d/r)^2) 이므로 hs 가
  .90 을 깨는 지점이 d = .564 r = .282 * short 다. 각진 벽은 끝까지 평평해 0 이다.

임계는 실측으로 정했다 (page_001, boxed 30개)
  각진 10개  cap_ratio = 0.000  (전부 정확히 0)
  둥근 20개  cap_ratio = 0.258 ~ 0.320
  사이가 통째로 비어 있다. 기본 .12 는 양쪽 어디에서도 멀다.

한계
  이것은 형태이지 클래스가 아니다. 둥근 끝 20개 중 2개(B0003 · B0004)는 라인
  클래스 태그가 아니라 CORE MAKEUP TANK 동체다 — 캡슐형 용기도 둥근 끝이기
  때문이다. 태그만 원하면 boxed_tags 가 짧은 변으로 이미 가른 family == 'TAG' 를
  함께 걸어야 한다 (detect.py 의 --boxed-corner 가 그렇게 한다).
"""
from __future__ import annotations

import numpy as np

#: 둥근 끝으로 볼 최소 cap 비율. 위 실측의 빈 구간 한가운데다.
STADIUM_MIN_RATIO = 0.12

#: 평지로 볼 프로파일 높이. connector_flags._end_taper 와 같은 값이다.
PLATEAU = 0.90


def _cap(hs, plateau):
    """프로파일 끝에서 안쪽으로 걸으며 평지를 깬 구간의 길이 (샘플 수)."""
    i = len(hs) - 1
    while i >= 0 and hs[i] < plateau:
        i -= 1
    if i < 0:            # 평지가 아예 없다 — 캡이 아니라 측정 실패다.
        return 0
    return len(hs) - 1 - i


def cap_ratio(raw, bbox, plateau=PLATEAU):
    """끝단이 좁아지는 길이 / 짧은 변. 각진 끝은 0.0, 반원 캡은 ~0.28 이다.

    양 끝 중 큰 쪽을 쓴다 — 한쪽에 배관 스텁이 붙어 캡이 메워진 경우에도
    반대쪽 캡이 남아 있으면 둥근 것으로 읽어야 하기 때문이다.
    """
    x0, y0, x1, y1 = bbox
    sub = raw[y0:y1, x0:x1]
    if sub.size == 0:
        return 0.0
    h, w = sub.shape
    short = min(h, w)
    if short < 1:
        return 0.0
    ys, xs = np.nonzero(sub)
    if ys.size == 0:
        return 0.0
    # 장축 위 각 자리의 윤곽 높이 = (최하 잉크 - 최상 잉크). 잉크 합이 아니다 —
    # 속의 글자가 세는 데 끼면 캡이 아니라 글자 밀도를 재게 된다.
    u, v, n = (xs, ys, w) if w >= h else (ys, xs, h)
    top = np.full(n, -1, np.int64)
    bot = np.full(n, 1 << 30, np.int64)
    np.maximum.at(top, u, v)
    np.minimum.at(bot, u, v)
    prof = np.where(top >= 0, top - bot + 1, 0).astype(float)
    body = float(np.percentile(prof, 90))
    if body <= 0:
        return 0.0
    hs = prof / body
    return round(max(_cap(hs, plateau), _cap(hs[::-1], plateau)) / short, 4)


def corner_style(raw, bbox, min_ratio=STADIUM_MIN_RATIO, plateau=PLATEAU):
    """('STADIUM'|'RECT', cap_ratio)."""
    r = cap_ratio(raw, bbox, plateau)
    return ('STADIUM' if r >= min_ratio else 'RECT'), r


def annotate(raw, dets, min_ratio=STADIUM_MIN_RATIO):
    """boxed_tags 결과에 corner_style · cap_ratio 를 제자리에서 달아 준다.

    게이트하지 않고 기록만 한다 — connector_flags 가 interior_ink 를 다루는 것과
    같은 규약이다. 거르는 것은 detect.py 의 --boxed-corner 가 한다.
    """
    for d in dets:
        d['corner_style'], d['cap_ratio'] = corner_style(raw, d['bbox'], min_ratio)
    return dets

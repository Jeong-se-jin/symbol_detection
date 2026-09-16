"""두 갈래를 한 목록으로 합친다 — 기하(닫힌 윤곽)가 배관(carrier)을 이긴다.

왜 기하가 이기는가
  닫힌 윤곽과 테이퍼 프로파일은 "이 잉크가 하나의 도형이다" 를 직접 재는 증거다.
  carrier 는 "배관이 여기서 끊겼다" 는 간접 증거여서, 계기 버블이나 박스 태그가
  배관 위에 얹혀 있으면 그 버블을 배관 끊김으로도 읽는다. 같은 잉크를 두 번 세지
  않으려면 더 직접적인 쪽을 남겨야 한다. page_001 실측으로 이 충돌은
  INSTRUMENT 8 · TAG 2 로 열 건 남짓이다.

  반대 방향(carrier 가 기하를 지우는 것)은 하지 않는다. 기하가 잡은 것은 배관과
  무관하게 그 자리에 도형이 있다는 뜻이고, 배관이 그 옆을 지나간다는 사실이
  그것을 무르지 못한다.

기하끼리는 손대지 않는다
  boxed_tags 가 exclude=connectors+frames 로 이미 배제를 끝냈고 (pkg9 규칙 그대로),
  각 검출기는 자기 안에서 iom_nms 를 돌린다. 여기서 또 겹침을 정리하면 그 규칙을
  두 벌로 만들게 된다.

id 는 위치 순번이 아니다
  출처별 접두사 + 그 검출기 안의 순번으로 준다 (C/F/R/N/B). 한 갈래의 파라미터를
  바꿔도 다른 갈래의 id 가 밀리지 않는다 — 좌표순 일련번호였다면 필터 임계 하나에
  전체 id 가 조용히 어긋나고, 그 어긋남은 실패하지 않으므로 알아채기 어렵다.
"""
from __future__ import annotations

from geom import iom

#: 출처 → (id 접두사, 기본 shape). shape 은 검출기가 판정한 기하 형태이지
#: 심볼의 클래스가 아니다.
SOURCES = {
    'carrier':   ('C', 'carrier_gap'),
    'frame':     ('F', 'closed_outline'),
    'ring':      ('R', 'dashed_circle'),
    'connector': ('N', 'taper_flag'),
    'boxed':     ('B', 'elongated_box'),
}

#: 공통 스키마가 자기 자리에서 쓰는 키. 나머지는 전부 geometry 로 내린다.
_LIFTED = {'bbox', 'score', 'score_kind', 'label'}

#: 한 잉크를 두 번 세지 않기 위한 겹침 기준. 작은 쪽 넓이 대비 절반을 넘으면
#: 같은 것으로 본다 (buble 이 배관 폭보다 크므로 IoU 가 아니라 IoM 이다).
COVER_THRESHOLD = 0.50


def normalize(dets, source):
    """검출기 dict → 공통 행 스키마. bbox 는 이미 배타 구간이다."""
    prefix, default_shape = SOURCES[source]
    rows = []
    for i, d in enumerate(dets, 1):
        shape = default_shape
        form = d.get('frame_form')
        if source == 'frame':
            shape = 'closed_rect' if form == 'RECTANGULAR' else 'closed_circle'
        rows.append({
            'id': f'{prefix}{i:04d}',
            'bbox': [int(v) for v in d['bbox']],
            'source': source,
            'shape': shape,
            'status': 'ACCEPTED',
            'reject_reason': None,
            'score': float(d.get('score') or 0.0),
            'score_kind': d.get('score_kind'),
            'geometry': {k: v for k, v in d.items() if k not in _LIFTED},
        })
    return rows


def number_carrier(rows):
    """carrier 행에 id 를 준다. 기각된 것도 번호를 받아야 되짚을 수 있다."""
    for i, r in enumerate(rows, 1):
        r['id'] = f'C{i:04d}'
    return rows


def suppress_covered_carrier(carrier_rows, geom_rows, threshold=COVER_THRESHOLD):
    """기하가 이미 차지한 자리의 carrier 행을 기각으로 돌린다. 기각 수를 돌려준다."""
    boxes = [r['bbox'] for r in geom_rows if r['status'] == 'ACCEPTED']
    n = 0
    for r in carrier_rows:
        if r['status'] != 'ACCEPTED':
            continue
        hit = max(((iom(r['bbox'], b), b) for b in boxes), default=(0.0, None))
        if hit[0] > threshold:
            r['status'] = 'REJECTED'
            r['reject_reason'] = 'COVERED_BY_OUTLINE'
            r['geometry']['covered_by_iom'] = round(float(hit[0]), 4)
            n += 1
    return n

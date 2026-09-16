"""실행마다 핵심 수치를 남기고 직전 실행과 비교한다.

왜 필요한가
  파라미터 하나를 바꾸면 여러 수치가 같이 움직인다. 로그를 눈으로 비교하면 의도한
  것만 보고 따라 움직인 것을 놓친다. 실제로 놓친 것들이다:

    --box-grow 3.0    NOTES 잔여를 줄였지만 라벨 옆 리더 화살촉을 같이 지웠다
    심볼 보호(넓게)    밸브를 살렸지만 버블 속 글자가 살아나 run 이 4배가 되었다
    --bridge-gap 18   점선 판정을 늘렸지만 그 +16 은 심볼을 건너뛴 실선이었고
                      심볼-배관 링크를 16개 잃었다

  셋 다 바꾼 수치는 좋아 보였고, 나빠진 쪽은 나중에 따로 재보고서야 알았다.

무엇을 재는가
  단계마다 하나씩 — 글자를 얼마나 지웠나, 심볼을 몇 개 찾았나, run 과 엣지가 몇
  개인가, 심볼이 배관에 몇 개 물렸나, 설명 못 한 잉크가 얼마인가. 마지막 것이
  특히 중요하다: run 이 늘어도 잉크를 더 설명하지 못하면 조각만 늘어난 것이다.
"""
from __future__ import annotations

import json
import os

#: 값이 클수록 좋은 항목. 나머지는 방향을 판단하지 않고 변화만 적는다.
HIGHER_IS_BETTER = {
    'symbols_accepted', 'edges', 'contact_tee', 'symbol_links',
    'symbol_link_runs', 'ink_explained_pct',
}


def collect(**kw):
    return {k: v for k, v in kw.items() if v is not None}


def load(out_dir):
    p = os.path.join(out_dir, 'metrics.json')
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p, encoding='utf-8'))
    except Exception:
        return None


def save(out_dir, m):
    with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8') as fp:
        json.dump(m, fp, ensure_ascii=False, indent=1)


def report(prev, cur):
    """직전 실행과의 차이를 표로. 바뀐 것만 적는다."""
    if not prev:
        return ['(직전 실행 기록 없음 — 이번 것이 기준선이 된다)']
    keys = [k for k in cur if isinstance(cur[k], (int, float))]
    rows = []
    for k in keys:
        a, b = prev.get(k), cur[k]
        if a is None or a == b:
            continue
        d = b - a
        mark = ''
        if k in HIGHER_IS_BETTER:
            mark = ' ↑' if d > 0 else ' ↓'
        rows.append(f'  {k:22} {a:>10} → {b:>10}  ({d:+}){mark}')
    if not rows:
        return ['직전 실행과 모든 수치가 같다']
    return rows

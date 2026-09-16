"""배관-부착 심볼 검출 — ap1000/ 의 v48 을 라이브러리로 부른다.

무엇을 하는가
  배관(carrier)이 한 축 위에서 끊겼다가 다시 이어지는 자리를 찾는다. 그 끊긴
  구간이 인라인 심볼(밸브 등)이 차지한 자리다. 템플릿도 OCR 도 쓰지 않는다.

ap1000/ 원본은 고치지 않는다
  아래 run() 은 AP1000_v48_EXACT_AXIS_SHORT_SIDE_FALLBACK.main() (942-1000행)에서
  파일 I/O 만 걷어낸 것이고 호출 순서는 한 줄도 다르지 않다. 순서가 어긋나면
  v46 정밀집합 보존 → 전체-carrier 보충 → exact-axis 폴백의 3단 누적이 깨진다.
  이식이 맞는지는 개수로 확인한다: page_001 에서 필터 이전 142개
  (CONNECTED 112 · MULTIPART 25 · WHOLE_CARRIER 5) 여야 한다.

bbox 를 배타 구간으로 바꾼다
  v48 의 bbox 는 포함 구간이다 (save_csv 가 width 를 x2-x1+1 로 쓴다). 이 폴더는
  pkg9 기하 쪽에 맞춰 전부 배타 구간 [x0,y0,x1,y1) 으로 통일하므로 +1 해서 낸다.
  놓치면 병합 IoM 이 한 픽셀씩 어긋난다.

필터 임계는 실측으로 정했다 (page_001, 6600x4859)
  v48 이 낸 142개를 v20_17 검출과 대조하니 오검출이 v48 자신의 지표로 갈렸다:

    displacement_span   진짜 최소 50px  /  오검출 중앙값 16px
    symbol_area         진짜 최소 122px /  오검출 3사분위 94px

  span>=13*sw AND symbol_area>=8*sw^2 로 자르면 142 → 97 이 되고, 기존 검출과
  겹치던 94개는 하나도 잃지 않으면서 오검출 45개가 사라진다. 사라지는 것은
  배관 교차 hop(31x16), 리더 화살표 촉(4~9px), 점선 신호선 구간(137x49)이다.
  pkg9 템플릿이 놓쳤던 해칭 밸브 3개(V022A·V022B·V044, 46x85)는 살아남는다.

  임계를 픽셀이 아니라 stroke width 배수로 두는 이유는 sw 가 v48 이 시트에서
  스스로 재는 값(median_skeleton_stroke_radius_px)이라 배율을 따라가기 때문이다.
  page_001 에서 sw=3.8px 이므로 13*sw=50, 8*sw^2=116 이다.

기각한 것을 버리지 않는다
  status='REJECTED' + reject_reason 으로 함께 낸다. 임계를 되돌려 보려면 출력만
  다시 읽으면 되고, 다시 돌릴 필요가 없다.
"""
from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_AP1000 = Path(__file__).resolve().parent / "ap1000"
if str(_AP1000) not in sys.path:
    sys.path.insert(0, str(_AP1000))

import AP1000_v34_AXIS_RELATIVE_FSM_RUN_GRAPH as v34          # noqa: E402
import AP1000_v37_TOLERANT_LONGEST_TRACK_HIERARCHY as v37     # noqa: E402
import AP1000_v44_LATENT_CARRIER_DILATION_BBOX as v44         # noqa: E402
import AP1000_v46_MULTIPART_PIPE_ATTACHED_BBOX as v46         # noqa: E402
import AP1000_v48_EXACT_AXIS_SHORT_SIDE_FALLBACK as v48       # noqa: E402


@dataclass
class Params:
    """v48 의 CLI 기본값 그대로 + 이 폴더가 더한 필터 둘."""
    min_line_length: int = 3
    pipe_witness_factor: float = 8.0
    axis_tolerance_strokes: float = 1.25
    lane_join_strokes: float = 0.75
    lane_min_density: float = 0.72
    max_displacement_strokes: float = 30.0
    internal_island_max_strokes: float = 24.0
    fragment_gap_strokes: float = 6.0
    multipart_min_side_balance: float = 0.30
    two_fragment_max_normal_ratio: float = 1.50
    embedded_carrier_fraction: float = 0.55
    #: 배관이 끊긴 길이의 하한. stroke width 배수.
    min_span_strokes: float = 13.0
    #: 끊긴 구간 안 잉크 픽셀 수의 하한. stroke width^2 배수.
    min_ink_strokes2: float = 8.0
    #: False 면 위 둘을 적용하지 않는다 (이식 무결성 확인용).
    filter_enabled: bool = True

    def as_dict(self):
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


def load(path):
    """이미지 또는 PDF → 회색조. PDF 는 v34 가 PyMuPDF 로 연다."""
    return v34.load(Path(path))


def binarize(gray):
    """잉크=True 인 bool 맵. 기하 쪽은 .astype(uint8) 해서 같은 맵을 쓴다."""
    return v34.binarize(gray)


def run(gray, ink, params):
    """(rows, stats, ctx). rows 는 배타 bbox 를 가진 dict 목록이다.

    ctx 는 render.py 가 v48 의 단계 그림을 그릴 때 쓰는 중간물이다.
    """
    p = params

    # ---- 여기부터 v48.main() 942-1000행과 같은 순서다 ----------------------
    sk = v34.skeletonize(ink)
    base_margin, med = v37.estimate_gap_margin(ink, sk)
    rs, hb0, vb0, hid0, vid0 = v34.extract(sk)
    hb, vb, hid, vid, active, lstats = v44.filter_line_entities(
        rs, hb0, vb0, hid0, vid0, p.min_line_length)
    v34.annotate(rs, hid, vid)

    base_ps, base_stats = v46.precision_proposals(
        rs, active, ink, base_margin, med, p.pipe_witness_factor, p.max_displacement_strokes,
        fragment_gap_strokes=p.fragment_gap_strokes,
        min_side_balance=p.multipart_min_side_balance,
        two_fragment_max_normal_ratio=p.two_fragment_max_normal_ratio)
    base_raw = v46.build_candidates(base_ps, rs, base_stats['pipe_witness_min_px'])
    base_acc = v46.resolve(base_raw, base_margin)
    base_cands = v48.convert_v46_base_candidates(base_raw, base_acc)

    ps, groups, eligible, pstats = v48.build_whole_carrier_proposals(
        rs, active, ink, base_margin, med, p.axis_tolerance_strokes, p.pipe_witness_factor,
        p.max_displacement_strokes, 0.05, p.lane_join_strokes, p.lane_min_density,
        p.internal_island_max_strokes, p.fragment_gap_strokes, p.multipart_min_side_balance,
        p.two_fragment_max_normal_ratio, p.min_line_length, p.embedded_carrier_fraction)
    supp = v48.build_candidates(ps, groups, rs)
    axis_tol = pstats['effective_axis_tolerance_px']

    sw = v48.stroke_width(med)
    # 심볼 안에 박힌 carrier 조각은 같은 성분 구간이다 — 중복 판정보다 먼저 편다.
    for c in base_cands:
        v48.expand_internal_carrier_islands(
            c, groups, eligible, rs, ink, base_margin, axis_tol, sw,
            p.internal_island_max_strokes, p.embedded_carrier_fraction)
    for c in supp:
        v48.expand_internal_carrier_islands(
            c, groups, eligible, rs, ink, base_margin, axis_tol, sw,
            p.internal_island_max_strokes, p.embedded_carrier_fraction)

    supp = [c for c in supp if v48.supplement_not_duplicate(c, base_cands, axis_tol)]
    for i, c in enumerate(supp):
        c.id = len(base_cands) + i
    v48.resolve(supp, axis_tol)
    supp_accepted = [i for i, c in enumerate(supp) if c.status == 'ACCEPTED']

    existing = base_cands + [supp[i] for i in supp_accepted]
    exact_fb, exact_stats = v48.build_exact_axis_near_witness_fallback(
        rs, active, ink, base_margin, med, p.pipe_witness_factor, p.max_displacement_strokes)
    exact_fb = [c for c in exact_fb if v48.supplement_not_duplicate(c, existing, axis_tol)]
    for i, c in enumerate(exact_fb):
        c.id = len(base_cands) + len(supp) + i
    v48.resolve(exact_fb, axis_tol)
    exact_accepted = [i for i, c in enumerate(exact_fb) if c.status == 'ACCEPTED']

    cands = base_cands + supp + exact_fb
    accepted = (set(range(len(base_cands)))
                | {len(base_cands) + i for i in supp_accepted}
                | {len(base_cands) + len(supp) + i for i in exact_accepted})
    # ---- v48.main() 의 검출 부분은 여기서 끝난다 --------------------------

    min_span = p.min_span_strokes * sw
    min_ink = p.min_ink_strokes2 * sw * sw

    rows = []
    n_span = n_ink = 0
    for i, c in enumerate(cands):
        x1, y1, x2, y2 = c.bbox
        row = {'bbox': [int(x1), int(y1), int(x2) + 1, int(y2) + 1],
               'source': 'carrier', 'shape': 'carrier_gap',
               'score': round(float(c.side_balance), 4),
               'score_kind': 'pipe_witness_side_balance_not_class_probability',
               'status': 'ACCEPTED', 'reject_reason': None,
               'geometry': {'geometry_mode': c.geometry_mode, 'orientation': c.o,
                            'proposal_source': c.proposal_source,
                            'displacement_span': int(c.displacement_span),
                            'symbol_area': int(c.symbol_area),
                            'symbol_offaxis_extent': int(c.symbol_offaxis_extent),
                            'gap_count': int(c.gap_count),
                            'internal_group_count': int(c.internal_group_count),
                            'fragment_count': int(c.fragment_count),
                            'before_support': int(c.before_support),
                            'after_support': int(c.after_support),
                            'axis_shift_px': int(abs(c.before_axis - c.after_axis)),
                            'v48_candidate_id': int(c.id)}}
        if i not in accepted:
            # v48 이 스스로 떨어뜨린 것 (중복 등). 개수를 세기 위해 함께 낸다.
            row['status'] = 'REJECTED'
            row['reject_reason'] = c.status if c.status != 'ACCEPTED' else 'NOT_ACCEPTED_BY_V48'
        elif p.filter_enabled and c.displacement_span < min_span:
            row['status'] = 'REJECTED'; row['reject_reason'] = 'SPAN_TOO_SHORT'; n_span += 1
        elif p.filter_enabled and c.symbol_area < min_ink:
            row['status'] = 'REJECTED'; row['reject_reason'] = 'INK_TOO_SMALL'; n_ink += 1
        rows.append(row)

    stats = {}
    stats.update(lstats); stats.update(pstats); stats.update(exact_stats)
    stats.update({
        'median_skeleton_stroke_radius_px': round(float(med), 3),
        'stroke_width_px': round(float(sw), 2),
        'estimated_base_margin_px': int(base_margin),
        'v48_candidates_built': len(cands),
        'v48_accepted_before_filter': len(accepted),
        'v48_accepted_modes_before_filter': dict(sorted(
            Counter(cands[i].geometry_mode for i in accepted).items())),
        'filter_enabled': p.filter_enabled,
        'filter_min_span_px': round(float(min_span), 1),
        'filter_min_ink_px': round(float(min_ink), 1),
        'rejected_span_too_short': n_span,
        'rejected_ink_too_small': n_ink,
        'accepted': sum(1 for r in rows if r['status'] == 'ACCEPTED'),
        'accepted_modes': dict(sorted(Counter(
            r['geometry']['geometry_mode'] for r in rows if r['status'] == 'ACCEPTED').items())),
    })
    ctx = {'rs': rs, 'active': active, 'groups': groups, 'eligible': eligible,
           'proposals': ps, 'cands': cands, 'accepted': sorted(accepted)}
    return rows, stats, ctx

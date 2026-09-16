"""글자를 지우되 글자가 앉은 상자는 지우지 않는다.

digitization-of-piping-and-instrument-diagrams 의 `tools/pid6/textink.py` 를
그대로 옮겼다. 본문은 손대지 않았고 주석만 우리 말로 다시 썼다.

왜 bbox 를 칠하면 안 되는가
  라벨은 배관 **위에** 앉는다. OCR 상자를 흰색으로 칠하면 그 아래 지나가던 배관과
  거기 도착하던 리더가 같이 사라진다. 글자의 잉크와 글자를 둘러싼 사각형은 다른
  것이고, 전자는 직접 골라낼 수 있다.

무엇으로 고르는가 — 연결 성분 단위다
  한 성분이 글자인 조건은 둘이다.

    1. 픽셀의 80% 이상이 어떤 OCR 상자 안에 있다
    2. 그 상자보다 크게 삐져나가지 않는다

  두 번째가 배관을 살린다. 라벨 밑을 지나는 배관은 상자 밖으로 양쪽 다 뻗은 하나의
  긴 성분이라 1번은 통과해도 2번에서 걸린다.
"""
from __future__ import annotations

import cv2
import numpy as np


def _overlap_frac(x1, y1, x2, y2, boxes):
    """이 구역이 보호 상자들에 덮인 비율 (구역 넓이 대비, 최대값)."""
    area = max((x2 - x1) * (y2 - y1), 1)
    best = 0.0
    for b in boxes:
        ix = min(x2, b[2]) - max(x1, b[0])
        iy = min(y2, b[3]) - max(y1, b[1])
        if ix > 0 and iy > 0:
            best = max(best, ix * iy / area)
    return best


def erase_text_ink(ink, boxes, protect=None, margin=0.45, symbol_overlap=0.35,
                   min_area=4):
    """OCR 상자 안의 잉크만 보고 글자를 지운다. (결과, 지운 덩어리 수).

    왜 성분 단위를 버렸는가
      원래는 잉크 전체를 연결 성분으로 나눠 "이 성분이 글자인가" 를 물었다. 그 방식은
      배관에 **닿은** 글자를 원리적으로 다룰 수 없다. page_001 의 V012B 에서 `V` 는
      밸브에 닿고 밸브는 배관에 닿아, 그 글자 한 자가 **49,703px · 2394x2965 짜리
      성분의 일부**가 된다. 그 성분에 걸친 OCR 상자만 310개다. 상자 안쪽 비율도,
      성분 크기도, 획 두께도 전부 그 거대 성분의 값이라 어떤 문턱으로도 안 갈린다.

      그래서 **상자 안에서 잘라 놓고** 본다. 잘라 내면 `V` 는 밸브·배관과 끊어져
      독립 덩어리가 되고, 판정은 하나로 줄어든다.

    판정
      **창 안에서 완결되는** 덩어리가 글자다. 테두리에 닿으면 창 밖으로 이어진다는
      뜻이고, 그러면 배관이거나 배관에 물린 무언가다.

      창은 OCR 상자에 여백을 더해 잡는다. 상자는 글자에 딱 맞으므로 여백이 없으면
      글자 자신이 테두리에 닿아 버린다. 여백을 두면 글자는 창 안쪽에 뜨고, 배관은
      어느 방향으로든 테두리를 넘는다.

      '마주보는 두 변을 잇는가'(관통) 만 보다가 바꿨다. 그 조건은 창 안에서 끝나는
      배관 조각을 전부 글자로 판정해 tee 가 4개까지 무너졌다. 한쪽으로만 나가도
      창 밖으로 이어지는 것은 마찬가지다.

      다만 검출된 심볼과 많이 겹치는 덩어리는 남긴다. OCR 상자가 글자보다 훨씬 크게
      잡혀 심볼을 통째로 삼키는 일이 있고(page_001 의 'D' 상자 178x56 이 밸브 V095A 를
      삼켰다), 그때 심볼이 글자로 지워지면 안 된다. 글자가 심볼 **옆**에 있어 조금
      겹치는 것과, 상자가 심볼을 **품은** 것은 겹친 비율로 갈린다.

    margin: 창을 상자보다 얼마나 키울지 (상자 짧은 변 배수).
    protect: 심볼이 실제로 차지한 자리를 칠한 마스크 (같은 크기, 0/1). **bbox 가 아니다.**
        bbox 로 판정하면 도형이 사각형이 아닌 심볼에서 빈 귀퉁이까지 심볼로 쳐진다 —
        오프페이지 커넥터는 왼쪽이 뾰족한 오각형이라 bbox 안에 빈 삼각형이 생기고,
        거기 놓인 글자 `A` 가 '심볼 몸통' 으로 판정되어 살아남았다.
    symbol_overlap: 덩어리 넓이 대비 심볼 마스크와 겹친 비율의 상한.
    min_area: 이보다 작은 덩어리는 건드리지 않는다 (이진화 얼룩).
    """
    out = ink.copy()
    H, W = ink.shape[:2]
    removed = 0
    for box in boxes:
        pad = max(3, int(round(margin * min(box[2] - box[0], box[3] - box[1]))))
        x1 = max(int(box[0]) - pad, 0)
        y1 = max(int(box[1]) - pad, 0)
        x2 = min(int(box[2]) + pad, W)
        y2 = min(int(box[3]) + pad, H)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        sub = (out[y1:y2, x1:x2] > 0).astype(np.uint8)
        if not sub.any():
            continue
        n, lab = cv2.connectedComponents(sub, 8)
        for rid in range(1, n):
            m = (lab == rid)
            if m.sum() < min_area:
                continue
            if (m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()):
                continue                      # 창 밖으로 이어진다 = 글자가 아니다
            if protect is not None and _is_symbol_body(
                    m, protect[y1:y2, x1:x2], symbol_overlap):
                continue                      # 심볼 몸통이다 — 글자가 아니다
            out[y1:y2, x1:x2][m] = 0
            removed += 1
    return out, removed


def _is_symbol_body(mask, sym, overlap):
    """이 덩어리가 글자가 아니라 **심볼 몸통**인가.

    두 가지를 함께 본다.

      심볼이 차지한 자리와 많이 겹친다      그리고
      그 자리 밖으로도 뻗는다

    둘 다여야 몸통이다. 겹침만 보면 계기 버블 속 코드(`TIA 015A`)까지 지켜 버린다 —
    그것은 심볼 안에 갇힌 **글자**이고 지워야 한다. 밖으로 뻗었다는 것이 "이 잉크는
    배관에 물린 심볼의 일부" 라는 증거다 (OCR 상자가 밸브를 삼킨 V095A 가 그랬다).
    """
    n = int(mask.sum())
    if not n:
        return False
    k = int((mask & (sym > 0)).sum())
    return bool(k and k / n > overlap and k < n)


def grow_boxes(boxes, along=3.0, limit=None):
    """OCR 상자를 글자가 흐르는 방향으로 늘린다. (위 이식분은 손대지 않았다.)

    왜 필요한가
      타일로 나눠 읽으면 타일보다 긴 글줄은 **어느 타일에도 온전히 들어가지 못한다**.
      page_001 의 NOTES 문단이 그렇다 — 한 줄이 1009px 인데 타일이 1280px, 보폭이
      896px 라 포착 창이 202px 밖에 안 된다. 그래서 상자가 타일 경계에서 잘리고,
      "1. THE SYSTEM ..." 이 "HE SYSTEM ..." 으로 들어온다. 잘려 나간 "1. T" 는
      어떤 상자에도 안 들어가므로 지워지지 않고 배관 run 으로 샌다.

      겹침을 키워도 해결되지 않는다. 한 줄이 통째로 들어가려면 보폭이 포착 창보다
      작아야 하는데, 그러려면 타일이 수백 장이 된다.

      잘려 나간 글자는 상자 **바로 옆 같은 줄**에 있다. 그래서 상자를 글줄 방향으로
      만 늘린다. 세로쓰기 라벨은 세로로 늘린다.

    왜 안전한가
      늘린 것은 상자일 뿐이고 글자 판정은 그대로다 — 성분의 80% 가 상자 안에 있고
      상자보다 크게 삐져나가지 않아야 한다. 라벨 밑을 지나는 배관은 늘린 상자보다도
      훨씬 길어서 여전히 걸린다.

    along: 글줄 방향으로 늘릴 양. 상자의 짧은 변(글자 높이) 배수다.
    limit: 절대 상한(px). None 이면 걸지 않는다.
    """
    out = []
    for b in boxes:
        x0, y0, x1, y1 = (int(v) for v in b[:4])
        w, h = x1 - x0, y1 - y0
        d = along * min(w, h)
        if limit is not None:
            d = min(d, limit)
        d = int(round(d))
        if w >= h:
            out.append([x0 - d, y0, x1 + d, y1])
        else:
            out.append([x0, y0 - d, x1, y1 + d])
    return out

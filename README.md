# symbol_detection — 배관 기하 + 닫힌 윤곽 기하로 심볼을 잡는다

pkg9 의 S 단계에서 **크롭을 쓰는 갈래를 전부 걷어낸** 독립 검출기다.
`pkg9` 를 import 하지 않고, GPU 도 Tesseract 도 쓰지 않는다.

```
python symbol_detection/detect.py <img|pdf> <out_dir> [--stages] [--no-carrier-filter]
                                  [--boxed-corner all|stadium|rect] [--erase]
```

저장소 루트에서 실행한다 (pkg9 와 달리 cwd 를 옮길 필요가 없다). PDF 입력은
`ap1000/` 의 v34 가 PyMuPDF 로 연다.

## 두 갈래

| 갈래 | 모듈 | 무엇을 재는가 | page_001 채택 |
|---|---|---|---|
| **carrier** | `carrier.py` → `ap1000/` (AP1000 v48) | 배관이 한 축 위에서 끊겼다 이어지는 자리 = 인라인 심볼 | 88 |
| **frame** | `outline.py` `instrument_frames` | 닫힌 사각/원 윤곽 + 내부 잉크 | 98 |
| **ring** | `outline.py` `dashed_circles` | 점선 큰 원 (Hough 후보 → 반지름별 잉크 프로파일) | 1 |
| **connector** | `boxed.py` `connector_flags` | 장축 단면 테이퍼 프로파일 (오프페이지 커넥터) | 31 |
| **boxed** | `boxed.py` `boxed_tags` | 세로로 긴 닫힌 윤곽 + 내부 잉크 | 30 |

기하 4종은 `pkg9/symbol_outline.py` · `pkg9/symbol_boxed.py` 에서 옮겨 왔고 함수 본문은
그대로다. `ap1000/` 7개 파일은 받은 패키지 그대로이고 **고치지 않는다**.

## 뺀 것과 그 대가

- **템플릿 정합 (`match_references` + `native_bank/` 8장)** — 도면에서 오려낸 템플릿을
  전면 ROI-IoU 로 맞추던 갈래다. 여기 없다. 그 대가로 page_001 의
  **SCREWED CAP 21개가 사라진다** — 배관 끝단이라 양쪽 배관 증거를 요구하는 v48 로는
  구조적으로 못 잡고, 닫힌 윤곽도 아니다. 밸브 92개 중 7개도 못 잡는다.
- **버블 내부 크롭 OCR (`label_frames`)** — 계기 코드(PT/LT/FT…)를 읽어 subtype 을 붙이던
  분류 코드다. 검출 개수에는 영향이 없다 (rows 를 제자리에서 고칠 뿐이다).

대신 전면 템플릿 탐색이 사라져 **수백 초(GPU) → 18초(CPU)** 가 되고, 템플릿이 놓치던
해칭 밸브 3개(V022A · V022B · V044)를 새로 잡는다.

## 박스 태그를 끝단 모양으로 가른다 — `corner.py`

`boxed_tags` 는 닫힌 윤곽 · 내부 잉크 · 장단비 세 가지만 잰다. 끝이 둥근지 각진지는
측정 항목에 없어서 성격이 다른 두 갈래가 같은 `elongated_box` 로 나온다:

| 끝단 | page_001 | 무엇인가 | 그래프에서 |
|---|---|---|---|
| **둥근 (STADIUM)** | 18 | `ECE│ECB` `BTA│BBC` — 배관 위 라인 클래스 경계 | 배관을 자르는 **분할점** |
| **각진 (RECT)** | 10 | `PMS│OP│CMT` `PLS│OP│S` — 점선 신호선으로 밸브 액추에이터에 묶인다 | 밸브에 붙는 **제어 태그** |

`corner.py` 가 장축 단면 프로파일에서 끝단이 평지(90%)를 깨는 길이를 짧은 변으로
나눈다. 반지름 `r` 반원 캡은 `d = .564r = .282 × short`, 각진 벽은 0 이다.
`connector_flags._end_taper` 와 같은 착상이되 **직선 테이퍼를 요구하지 않는다** —
둥근 캡은 호이지 직선이 아니다.

```
각진 10개  cap_ratio = 0.000   (전부 정확히 0)
둥근 20개  cap_ratio = 0.258 ~ 0.320
```

사이가 통째로 비어서 임계 `.12` 는 양쪽 어디에서도 멀다. `boxed.py` 는 pkg9 이식본이라
본문을 고치지 않고, 검출이 끝난 bbox 를 다시 재는 별도 갈래로 뒀다.

**용기는 칸막이로 가른다.** 캡슐형 CORE MAKEUP TANK 도 끝이 둥글어서 형태만으로는
경계 태그와 안 갈린다. 태그는 코드를 칸에 나눠 담고 용기는 속이 비어 있으므로
`boxed_tags` 가 이미 세어 둔 `cells >= 2` 를 함께 건다 — 짧은 변 임계는 배율을
따라가야 하지만 칸막이는 어느 배율에서도 칸막이다.

`--boxed-corner stadium` 은 어긋나는 것을 `CORNER_STYLE` 로 **기각 표시만 하고
버리지 않는다**. 이 필터는 `suppress_covered_carrier` **뒤에** 건다 — 앞에 걸면
기각된 박스가 덮고 있던 carrier 가 되살아나서, 박스를 거르는 일이 carrier 개수를
조용히 바꾼다.

## 리더 선 — `leader.py`

경계 태그는 배관 옆에 떠 있고 **칸막이 한가운데에서 나온 1px 선 하나**로 배관에
묶인다. 태그만 지우면 이 선이 배관에 붙은 짧은 가지로 남아 배관 추적이 그것을
스텁으로 읽는다. 태그와 선은 한 몸이라 함께 지운다.

어디서 끝내는가는 **골격 경로의 갈래 수**로 잰다. 선을 1px 골격으로 얇게 만든 뒤
한 픽셀씩 걸으며 8-이웃 고리의 교차수를 센다 — 끝점 1 · 지나가는 자리 2 · 분기 3 이상.
분기 픽셀은 밟지 않고 멈춘다.

단면 폭만 보는 방법을 먼저 썼다가 버렸다. page_001 에서 두 가지를 못 가린다:

- **B0006** 태그에서 수평으로 14px 나간 뒤 45° 로 꺾여 내려가는 **진짜 리더**다.
  출발점 표류로 자르면 꺾이는 자리에서 잘려 대각선이 남는다.
- **B0022** 수직 리더가 체크밸브에 닿은 뒤 **삼각형 빗변으로 갈아탄다**. 빗변도
  1~2px 라 폭으로는 안 잡히고, 지우개가 밸브를 같이 지운다.

둘 다 방향이 갑자기 바뀌지만 다른 것은 그 자리의 차수다 — 꺾임은 들어온 길과 나갈
길뿐(2)이고 밸브를 만나는 자리는 직진하는 배관까지 셋(3)이다. 교차수는 90° 꺾임을
2 로 내므로 "꺾임은 통과하고 분기만 잡는다" 가 한 수로 성립한다.

page_001 에서 18개 전부 리더 1개씩, 전부 `JUNCTION` 에서 멈춘다 (길이 58~218px).

`--erase` 는 채택된 타원 태그의 bbox 와 리더 경로를 지운 `erased.png` 를 쓴다.
지워진 잉크는 전체의 2.39% 이고, 다른 채택 심볼의 잉크는 깎이지 않는다 — carrier
6행의 bbox 안 잉크가 주는 것은 그 bbox 가 태그·리더를 품고 있었기 때문이지 밸브가
손상된 것이 아니다.

## 선 검출까지 — `pipeline.py`

`detect.py` 는 심볼만 낸다. 텍스트 OCR → 타원 태그 제거 → 심볼 검출 → 선 검출까지
네 단계를 잇는 것이 `pipeline.py` 이고, 그 로직과 실측은 **`PIPELINE.md`** 에 있다.

```
python pipeline.py input/ML103480545_page65.pdf out/full
```

## 글자를 붙인다 — `textmap.py`

`pipeline.py` 가 글자를 **지우는** 데까지라면, `textmap.py` 는 같은 글자를 심볼·배관에
**붙인다**. 다섯 규칙을 순서대로 걸고(심볼 오독 제거 → 심볼 내부 → 커넥터 이웃 →
밸브 태그 → 라인 번호 → 리더 추적), 그 로직과 실측은 **`TEXTMAP.md`** 에 있다.

```
python textmap.py out/full            # symbols.json · lines.json · texts.json 을 고친다
python textmap_render.py out/full     # 01_texts · 09_component_id · 10_text_map png
```

page_001: 텍스트 887 중 **634개**를 붙인다 (심볼 229개 · run 139개).

## 클래스를 내지 않는다

`family` / `kind` 필드를 두지 않는다. 검출기가 기하로 가른 갈래 이름
(`INSTRUMENT FRAME` · `TANK` · `VESSEL` · `BOXED CODE TAG` …)은 `geometry` 안에
추적용으로만 남는다. `shape` 는 형태이지 클래스가 아니다:
`carrier_gap` · `closed_rect` · `closed_circle` · `dashed_circle` · `taper_flag` · `elongated_box`.

## 배관 기하 필터 — 임계는 실측으로 정했다

v48 자체는 page_001 에서 142개를 내는데 그중 45개가 심볼이 아니다. 걸러내는 데
**크롭이 필요 없다** — v48 이 이미 재고 있는 값으로 갈린다:

| | 진짜 (94개) | 오검출 (48개) |
|---|---|---|
| `displacement_span` | 최소 **50** px | 중앙값 16, 3사분위 31 |
| `symbol_area` | 최소 **122** px | 중앙값 62, 3사분위 94 |

그래서 두 하한을 건다. 픽셀이 아니라 **stroke width(`sw`) 배수**다 — `sw` 는 v48 이
시트에서 스스로 재는 값(`median_skeleton_stroke_radius_px`)이라 배율을 따라간다.
page_001 에서 `sw = 3.8px`:

- `--min-span-strokes` 기본 **13.0** → `displacement_span >= 49.4px`
- `--min-ink-strokes2` 기본 **8.0** → `symbol_area >= 115.5px`

결과: 142 → 97. **기존 검출과 겹치던 94개는 하나도 잃지 않고** 오검출 45개가 사라진다.
사라지는 것은 배관 교차 hop(31×16), 리더 화살표 촉(4~9px), 점선 신호선 구간(137×49)이다.

page_001 에서는 `span` 하나가 45개를 다 잡고 `symbol_area` 는 추가로 지우는 것이 없다.
독립적인 두 번째 방어선으로 남겨 뒀다 — 단독으로는 38개를 지운다.

`--no-carrier-filter` 로 끄면 v48 원본과 같은 142개가 나온다 (이식 무결성 확인용).

## 병합 — 기하가 이긴다

carrier 행이 기하 행과 `IoM > 0.5` 면 carrier 쪽을 `COVERED_BY_OUTLINE` 으로 기각한다.
닫힌 윤곽은 "이 잉크가 하나의 도형이다" 를 직접 재고, carrier 는 "배관이 여기서 끊겼다"
는 간접 증거다. 계기 버블이 배관 위에 얹혀 있으면 양쪽이 같은 잉크를 센다.
page_001 에서 9건. 반대 방향은 하지 않는다.

**기각된 것은 버리지 않는다.** `status` + `reject_reason` 으로 출력에 함께 남으므로
임계를 되짚을 때 다시 돌릴 필요가 없다.

## id 는 위치 순번이 아니다

출처별 접두사 + 그 검출기 안의 순번이다: `C`(carrier) `F`(frame) `R`(ring)
`N`(connector) `B`(boxed). 한 갈래의 파라미터를 바꿔도 다른 갈래의 id 가 밀리지 않는다.
좌표순 일련번호였다면 필터 임계 하나에 전체 id 가 조용히 어긋나고, 그 어긋남은
실패하지 않으므로 알아채기 어렵다.

## 산출물

```
<out_dir>/
  symbols.json   {version, input, params, stats, symbols[]}
  symbols.csv    같은 내용의 평면 표 (utf-8-sig, geometry 키를 전부 열로 편다)
  stats.json     갈래별 개수 · 기각 사유별 개수 · 단계별 소요
  overlay.png    도면 위 오버레이. 출처별 색, 기각은 옅은 회색, 좌상단에 범례
  carrier_0*.png --stages 일 때만. v48 내부 단계 4장 (raw HV → witness → 변위 → bbox)
  erased.png     --erase 일 때만. 채택된 타원 태그와 그 리더를 지운 도면
```

`symbols.json` 의 한 행:

```json
{ "id": "C0088", "bbox": [x0, y0, x1, y1],
  "source": "carrier", "shape": "carrier_gap",
  "status": "ACCEPTED", "reject_reason": null,
  "score": 1.0, "score_kind": "pipe_witness_side_balance_not_class_probability",
  "geometry": { "geometry_mode": "MULTIPART", "orientation": "V",
                "displacement_span": 85, "symbol_area": 431, ... } }
```

**bbox 는 전부 배타 구간** `[x0, y0, x1, y1)` 이다 (`width = x1 - x0`).
v48 원본은 포함 구간이라 `carrier.py` 가 `+1` 해서 맞춘다.

## 실측 (page_001_native.png, 6600×4859)

```
채택 248   carrier 88 · frame 98 · connector 31 · boxed 30 · ring 1
기각  54   SPAN_TOO_SHORT 45 · COVERED_BY_OUTLINE 9
소요  18s  binarize 0.2 · geometry 13.4 · carrier 4.7      (GPU 없음)
```

- 필터 OFF 시 carrier 142개, `CONNECTED 112 · MULTIPART 25 · WHOLE_CARRIER 5` —
  받은 패키지를 직접 실행한 결과와 정확히 같다.
- `AP1000_v20_12/outputs_v20_17` 의 VALVE 92개 중 **85개**를 `IoM > 0.3` 으로 덮는다
  (중앙값 0.98). 기하 4종은 INSTRUMENT 96 · CONNECTOR 31 · TAG 28 · TANK 4 · VESSEL 1 로
  pkg9 와 개수가 정확히 일치한다.

## 의존성

`numpy` · `opencv-python` · `scikit-image`(skeletonize — carrier 와 `leader.py`) ·
`PyMuPDF`(PDF 입력).
이 저장소 `.venv` 에 모두 있다. torch · paddle · tesseract 는 필요 없다.

"""OCR 결과 캐시 — 같은 그림을 같은 설정으로 두 번 읽지 않는다.

왜 필요한가
  OCR 이 이 파이프라인에서 가장 비싸다 (page_001 에서 6분 반, 나머지 세 단계를
  합쳐도 30초다). 그런데 고치는 것은 대개 2~4단계라, 캐시가 없으면 임계 하나
  바꿀 때마다 6분을 다시 쓴다.

무엇으로 키를 만드는가
  **그림의 내용**과 OCR 설정이다. 출력 디렉터리로 키를 삼으면 같은 폴더에 다른
  도면을 돌렸을 때 남의 글자를 물려받고, 그 잘못은 조용해서 알아채기 어렵다.
  파일 경로도 키가 못 된다 — 같은 경로의 내용이 바뀔 수 있다.

  그래서 렌더된 페이지 PNG 의 바이트 해시 + tile/overlap/scale 을 키로 쓴다.
  도면이 바뀌면 키가 바뀌고, 설정이 바뀌어도 키가 바뀐다.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / 'cache' / 'ocr'


def key(page_path, tile, overlap, scale):
    h = hashlib.sha256()
    with open(page_path, 'rb') as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b''):
            h.update(chunk)
    h.update(f'|tile={tile}|overlap={overlap}|scale={scale}'.encode())
    return h.hexdigest()[:32]


def path_for(k):
    return CACHE_DIR / f'{k}.json'


def load(k):
    p = path_for(k)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None          # 깨진 캐시는 없는 것으로 본다


def save(k, payload):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = path_for(k)
    # 같은 캐시를 두 프로세스가 동시에 쓸 때 반쯤 쓰인 파일이 남지 않게 한다.
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp, p)
    return p

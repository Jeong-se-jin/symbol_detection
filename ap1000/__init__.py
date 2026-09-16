"""AP1000 v48 배관-부착 심볼 기하 — 외부 패키지를 **바이트 무수정**으로 들여왔다.

출처
  AP1000_v48_EXACT_AXIS_SHORT_SIDE_PACKAGE (2026-09-15 수령, README.txt 동봉)

이 디렉터리의 .py 는 고치지 않는다. 고칠 일이 생기면 carrier.py 쪽에서 감싸거나
파라미터로 연다 — 원본을 그대로 둬야 상류 패키지가 갱신됐을 때 바이트 비교로
무엇이 달라졌는지 볼 수 있다.

7개만 들여왔다. v48 → v46 → v45 → v44 → v37 → v36 → v34 체인이 전부이고,
동봉돼 있던 AP1000_v48_ORIGINAL_REFERENCE.py 는 아무도 import 하지 않아 뺐다.

모듈들은 스스로 sys.path 에 자기 디렉터리를 끼워 넣고 서로를 최상위 이름으로
import 한다 (`import AP1000_v34_... as v34`). 그래서 패키지 상대 import 가 아니라
carrier.py 가 이 디렉터리를 sys.path 에 올린 뒤 최상위로 부른다.
"""

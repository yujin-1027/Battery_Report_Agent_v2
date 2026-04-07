# Battery Report Agent

글로벌 배터리 시장 구조 변화 속 LG에너지솔루션 vs CATL 전략 비교 보고서를 자동 생성하는 AI Agent.

---

## 전체 실행 순서 요약

```
0. data/ 에 PDF 6개 배치
1. .env 설정 (API 키)
2. 의존성 설치
3. Docker로 Qdrant 실행
4. PDF Ingestion (벡터 DB 구축)
5. 에이전트 실행
```

---

## 상세 실행 방법

### 0. 데이터 준비 (PDF 수동 배치)

PDF 파일은 저작권 및 용량 문제로 git에 포함되지 않습니다.
아래 6개 파일을 직접 구해서 `data/` 폴더에 넣어주세요.

```
data/
├── 중국_산업보고서.pdf
├── 한국_배터리산업.pdf
├── 중국_정책.pdf
├── 미국_정책.pdf
├── CATL_ESG.pdf
└── Lgen_ESG.pdf
```

> **파일명을 정확히 맞춰주세요.** `tool/ingest.py`의 `METADATA_TABLE`이 파일명 기준으로
> source_type / company / date 메타데이터를 매핑합니다.
> 파일명이 다르면 경고 후 건너뜁니다.

### 1. 환경 설정

```bash
cp .env.example .env
```

`.env`를 열어 아래 키를 입력합니다.

| 키 | 필수 | 설명 |
|----|------|------|
| `OPENAI_API_KEY` | 필수 | LLM 호출용 |
| `TAVILY_API_KEY` | 필수 | 웹 검색용 ([tavily.com](https://tavily.com) 에서 발급) |
| `LANGSMITH_API_KEY` | 선택 | 실행 추적용 |

### 2. 의존성 설치

```bash
uv venv .venv --python=3.11
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

> `uv`가 없으면: `pip install uv` 또는 `pip install -r requirements.txt` 로 직접 설치

### 3. Qdrant Docker 실행 (벡터 DB 서버)

Qdrant는 RAG(내부 문서 검색)에 사용하는 벡터 DB입니다.
Docker로 로컬에서 실행합니다.

```bash
docker compose up -d
```

정상 실행 확인:
```bash
curl http://localhost:6333/healthz
# 응답: {"status":"ok"} 이면 정상
```

> **Docker가 없다면:** [docker.com](https://www.docker.com/products/docker-desktop) 에서 Docker Desktop 설치
>
> Qdrant 없이 실행해도 됩니다. RAG 검색 결과가 비어 있어도 웹 검색(Tavily)만으로 동작합니다.

### 4. PDF Ingestion (벡터 DB 구축)

`data/` 폴더의 PDF를 읽어 청크 분할 → Qwen3-Embedding-0.6B 임베딩 → Qdrant 저장합니다.

```bash
python tool/ingest.py --reset
```

- `--reset`: Qdrant collection 초기화 후 재구축 (최초 실행 또는 PDF 변경 시 사용)
- 최초 실행 시 임베딩 모델 다운로드(~1GB)가 발생합니다
- 완료 후 `Qdrant 저장: N개` 메시지로 정상 저장 확인

> Step 3 Qdrant가 실행 중이어야 합니다.

### 5. 에이전트 실행

```bash
python main.py
```

또는 코드에서 직접 호출:

```python
from main import run_battery_report

report = run_battery_report(
    "LG에너지솔루션과 CATL의 전략을 비교 분석해주세요.",
    session_id="my-session-001"   # 생략 시 새 세션 자동 생성
)
```

최종 보고서는 `output/battery_report_{session_id}_{timestamp}.md` 에 저장됩니다.

---

## 코드 검토 순서

1. `config.py` — 전역 상수 및 환경 변수
2. `state.py` — BatteryReportState, ResourceItem 타입 정의
3. `tool/web_search.py` — 웹 검색 도구 (Tavily)
4. `tool/rag_retriever.py` — RAG 검색 도구 (Qdrant + Qwen3 임베딩)
5. `tool/ingest.py` — PDF → 벡터 DB ingestion 스크립트
6. `memory/memory_manager.py` — 세션 메모리 로드/저장
7. `prompt/*.py` — 각 에이전트 프롬프트
8. `agent/query_agent.py` — 쿼리 파싱 노드
9. `agent/market_agent.py` — 산업 동향 서브 그래프
10. `agent/lg_agent.py`, `agent/catl_agent.py` — 기업 분석 노드
11. `agent/report_agent.py` — Aggregator + 보고서 작성 + 품질 검사
12. `agent/supervisor_agent.py` — Supervisor + 강제 종료 노드
13. `graph.py` — 전체 그래프 조립
14. `main.py` — 실행 진입점

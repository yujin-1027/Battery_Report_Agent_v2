# Subject

Battery Market Strategy Analysis using Multi-Agent System

A multi-agent based system that analyzes and compares the portfolio diversification strategies of LG Energy Solution and CATL under the EV chasm, using PDF-based retrieval and real-time web search.

## Overview

- Objective :글로벌 배터리 시장의 구조적 변화 속에서 LG에너지솔루션과 CATL의 포트폴리오 다각화 전략을 비교 분석하고, 한국 배터리 산업의 대응 방향과 시사점을 도출하는 전략 분석 보고서를 생성하는 것을 목표로 합니다.
- Method :Supervisor 기반 Multi-Agent Workflow를 설계하고, 내부 PDF 문서 검색을 위한 RAG와 최신 시장 정보 보완을 위한 Web Search를 결합하여 산업 분석, 정책 분석, 기업 분석, Comparative SWOT, 최종 보고서 생성을 수행합니다.
- Tools :
  Python, LangGraph, LangChain, Qdrant, Tavily API, OpenAI API

## Features

- PDF 자료 기반 정보 추출
  - 산업 보고서, 정책 보고서, ESG 보고서 등 내부 문서를 기반으로 핵심 정보 검색
- Web Search 기반 최신 시장 데이터 반영
  - 내부 문서에 없는 최신 시장 동향과 외부 환경 변화 보완
- 기업별 전략 분석
  - LG에너지솔루션과 CATL의 포트폴리오 다각화 전략 및 핵심 경쟁력 비교
- Comparative SWOT 자동 생성
  - 양사의 강점, 약점, 기회, 위협을 비교 관점에서 구조화
- 보고서 자동 생성
  - SUMMARY, 시장 배경, 기업 전략, SWOT, 종합 시사점, REFERENCE를 포함한 보고서 생성
- 확증 편향 방지 전략 :
  - 내부 문서와 외부 웹 자료를 함께 활용
  - 특정 출처에 편중되지 않도록 다중 소스 기반으로 정보 수집
  - Agent 간 Cross-validation 구조를 통해 결과 일관성과 신뢰성 점검
  - 최신 데이터 중심의 Web Search를 적용하여 오래된 정보 반영을 최소화

## Tech Stack

| Category   | Details                        |
| ---------- | ------------------------------ |
| Framework  | LangGraph, LangChain, Python   |
| LLM        | OpenAI API                     |
| Retrieval  | Qdrant                         |
| Embedding  | Qwen/Qwen3-Embedding-0.6B      |
| Web Search | Tavily API                     |
| Loader     | PyPDFLoader                    |
| Splitter   | RecursiveCharacterTextSplitter |

## RAG & Web Search Tool

- RAG Retriever

  - Qdrant 기반 벡터 검색 (PDF 문서)
  - Qwen Embedding (다국어)
  - metadata 필터: `research`, `lg`, `catl`
- Web Search

  - Tavily API 기반 최신 데이터 검색
  - RAG에서 부족한 정보 보완
- Strategy

  - RAG → 실패 시 Web Search fallback

### Component Summary

| Component       | Selection                      | Reason                                                                       |
| --------------- | ------------------------------ | ---------------------------------------------------------------------------- |
| Document Loader | PyPDFLoader                    | 한글·중국어 PDF 텍스트 추출 검증 완료                                       |
| Text Splitter   | RecursiveCharacterTextSplitter | chunk_size=512 기준으로 의미 단위를 유지하여 검색 정확도 향상                |
| Embedding Model | Qwen/Qwen3-Embedding-0.6B      | 한·영·중 다국어를 단일 벡터 공간에서 처리 가능하며 로컬 실행으로 비용 절감 |
| Vector Store    | Qdrant                         | 메타데이터 필터 기반 문서 분리 검색 지원                                     |
| Web Search      | Tavily API                     | 최신 시장 동향과 외부 정보 보완에 활용                                       |

### Search Filter Structure

내부 문서는 ingestion 시 메타데이터를 태깅하여 에이전트별로 분리 검색합니다.

| filter_key   | Target Documents          | Agent                 |
| ------------ | ------------------------- | --------------------- |
| `research` | 산업·정책 리서치 PDF     | Market Analysis Agent |
| `lg`       | LG에너지솔루션 ESG 보고서 | LG Analysis Agent     |
| `catl`     | CATL ESG 보고서           | CATL Analysis Agent   |

### Fallback Strategy

RAG가 빈 결과를 반환할 경우, 에이전트는 자동으로 Web Search 기반 분석으로 전환됩니다.
이를 통해 벡터 DB 미구성 상황에서도 분석 흐름이 중단되지 않도록 설계하였습니다.

## Agents

- Planner Agent : 전체 Task를 분배하고 Workflow 흐름을 제어
- Market Analysis Agent : 배터리 산업 및 정책 환경 분석 수행
- LG Analysis Agent : LG에너지솔루션의 포트폴리오 전략과 핵심 경쟁력 분석
- CATL Analysis Agent : CATL의 포트폴리오 전략과 핵심 경쟁력 분석
- Comparison Agent : 두 기업의 전략 비교 및 Comparative SWOT 작성
- Report Agent : 분석 결과를 종합하여 최종 보고서 형식으로 정리

## Agents

- Query Agent
  사용자 쿼리를 파싱하고 의도, 대상 기업, 유효성을 판별합니다.
- Market Analysis Agent
  산업 동향 분석과 정책 분석을 수행하는 서브그래프 구조의 에이전트입니다.
- LG Analysis Agent
  Web Search와 RAG를 활용해 LG에너지솔루션의 전략, SWOT, 재무 정보를 수집하고 구조화합니다.
- CATL Analysis Agent
  Web Search와 RAG를 활용해 CATL의 전략, SWOT, 재무 정보를 수집하고 구조화합니다.
- Report Writer Agent통합된 분석 결과를 바탕으로 최종 Markdown 보고서를 작성합니다.
- Supervisor Agent
  전체 파이프라인의 진행 상태를 확인하고 다음 실행 단계를 라우팅합니다.

## Architecture

Supervisor Pattern 기반 Multi-Agent Workflow

##Flow
Query
→ Planner Agent
→ Market Analysis Agent
→ LG Analysis Agent / CATL Analysis Agent
→ Comparison Agent
→ Report Agent

## Directory Structure

```text
├── data/                  # PDF 문서
├── agents/                # Agent 모듈
├── prompts/               # 프롬프트 템플릿
├── vector_db/             # 벡터 DB 및 임베딩 저장소
├── outputs/               # 생성된 보고서 저장
├── main.py                # 실행 스크립트
└── README.md
```

## 상세 실행 방법 (Setup)

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

## Contributors

- 김철희 : LangGraph 설계
- 박소윤 : Vector DB 구축, PDF Parsing
- 신준용 : Vector DB 구축, PDF Parsing
- 최유진 : Prompt Engineering, LangGraph 설계

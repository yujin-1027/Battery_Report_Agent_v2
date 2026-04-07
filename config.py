"""
[config.py]
개념: 프로젝트 전역 상수 및 환경 설정 관리 모듈
기능: 모델명, 경로, 외부 서비스(Qdrant, 웹 검색) 접속 정보, 재시도 횟수 등
     코드 전반에서 공통으로 참조하는 단일 설정 파일.
     환경 변수는 .env에서 로드하며, 기본값도 여기서 관리.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# ── 모델 ─────────────────────────────────────────────────────────────────────
MODEL_NAME        = "gpt-4o"
MODEL_TEMPERATURE = 0

# ── 재시도 상한 (Supervisor가 이 횟수 초과 시 END_WARNING 선택) ───────────────
MAX_RETRIES = 3

# ── 경로 ─────────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
DATA_DIR         = BASE_DIR / "data"           # RAG 원본 PDF 위치
OUTPUT_DIR       = BASE_DIR / "output"         # 최종 보고서 저장 위치
MEMORY_STORE_DIR = BASE_DIR / "memory" / "store"  # 세션 메모리 JSON 저장 위치

# 런타임에 필요한 디렉터리 자동 생성
OUTPUT_DIR.mkdir(exist_ok=True)
MEMORY_STORE_DIR.mkdir(parents=True, exist_ok=True)

# ── Qdrant (RAG, Docker Container) ───────────────────────────────────────────
QDRANT_HOST       = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "battery_research")

# ── Embedding 모델 (Qwen3-Embedding-0.6B, 로컬 실행) ─────────────────────────
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")
EMBEDDING_VECTOR_DIM = 1024
EMBEDDING_BATCH_SIZE = 32

# ── 검색 설정 ─────────────────────────────────────────────────────────────────
WEB_SEARCH_MAX_RESULTS = 5   # 웹 검색 결과 최대 건수
RAG_TOP_K              = 5   # RAG 유사도 검색 상위 K

# ── 리소스 요약 최대 길이 ─────────────────────────────────────────────────────
RESOURCE_SUMMARY_MAX_CHARS = 500

# ── LangSmith 추적 (선택) ─────────────────────────────────────────────────────
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "battery-report-agent")

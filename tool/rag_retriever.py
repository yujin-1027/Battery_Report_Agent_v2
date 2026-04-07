"""
[tool/rag_retriever.py]
개념: Qdrant 기반 RAG(벡터 검색) 도구 인터페이스
기능:
  - RagResult            : RAG 검색 결과 단건의 기대 타입 정의
  - get_rag_retriever_tool() : LangChain Tool 객체 반환

임베딩: Qwen/Qwen3-Embedding-0.6B (로컬, sentence-transformers)
        OpenAI Embeddings 미사용 — API 키 불필요
연결 대상: Qdrant Docker Container (기본 localhost:6333)
Collection: battery_research (vector dim=1024, distance=COSINE)

필터 키 (filter_key 파라미터):
  "research" — source_type=research 문서 (산업/정책 리서치)
  "lg"       — company=lg 문서 (LG에너지솔루션)
  "catl"     — company=catl 문서 (CATL)
  ""         — 필터 없음 (전체 검색)
"""

import torch
from typing import TypedDict, Optional
from langchain_core.tools import tool
from config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION,
    EMBEDDING_MODEL_NAME, EMBEDDING_VECTOR_DIM, RAG_TOP_K,
)


class RagResult(TypedDict):
    """
    RAG 검색 단건 결과 기대 타입.
    다른 팀 구현 시 이 스키마를 준수해야 함.
    """
    doc_id:   str    # Qdrant 문서 ID
    content:  str    # 청크 원문 → ResourceItem.raw_content
    source:   str    # 출처 파일명 → ResourceItem.source_url
    score:    float  # 코사인 유사도 점수 (0~1)
    metadata: dict   # source_type, company, date, page_number 등


# ── 공유 리소스 초기화 (모듈 로드 시 1회) ────────────────────────────────────

_model = None
_client = None
_init_error = None


def _init():
    global _model, _client, _init_error
    if _model is not None or _init_error is not None:
        return

    try:
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient

        device = (
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
        print(f"[RAG] 임베딩 모델 로드 완료 — {EMBEDDING_MODEL_NAME} ({device})")

        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5)
        _client.get_collections()   # 연결 확인
        print(f"[RAG] Qdrant 연결 성공 — {QDRANT_HOST}:{QDRANT_PORT}/{QDRANT_COLLECTION}")

    except Exception as e:
        _init_error = str(e)
        print(f"[RAG WARNING] 초기화 실패 ({e}). RAG 검색은 빈 결과를 반환합니다.")


def _build_filter(filter_key: str):
    """filter_key에 따라 Qdrant Filter 생성."""
    if not filter_key:
        return None
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        if filter_key == "research":
            return Filter(must=[FieldCondition(key="source_type", match=MatchValue(value="research"))])
        elif filter_key in ("lg", "catl"):
            return Filter(must=[FieldCondition(key="company", match=MatchValue(value=filter_key))])
    except ImportError:
        pass
    return None


def rag_retrieve(query: str, filter_key: str = "", top_k: int = RAG_TOP_K) -> list[RagResult]:
    """
    Qdrant에서 쿼리와 유사한 문서 청크를 검색하여 RagResult 리스트로 반환.
    filter_key: "research" | "lg" | "catl" | "" (전체)
    """
    _init()
    if _model is None or _client is None:
        return []

    vec = _model.encode(query).tolist()
    response = _client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=vec,
        query_filter=_build_filter(filter_key),
        limit=top_k,
        with_payload=True,
    )

    results: list[RagResult] = []
    for pt in response.points:
        payload = pt.payload or {}
        results.append(RagResult(
            doc_id=str(pt.id),
            content=payload.get("text", ""),
            source=payload.get("source_domain", payload.get("source", "unknown")),
            score=pt.score,
            metadata={k: v for k, v in payload.items() if k != "text"},
        ))
    return results


def get_rag_retriever_tool(top_k: int = RAG_TOP_K):
    """
    LangGraph ToolNode에 등록할 LangChain Tool 반환.
    에이전트는 filter_key로 검색 범위를 좁힐 수 있음.
    """
    _init()  # 임포트 시점에 미리 초기화

    @tool
    def rag_search(query: str, filter_key: str = "") -> str:
        """
        배터리 산업 관련 내부 문서(PDF 보고서, 리서치 자료)에서 관련 정보를 검색합니다.
        Qdrant 벡터 DB에서 쿼리와 가장 유사한 문서 청크를 반환합니다.

        filter_key 옵션:
          "research" — 산업/정책 리서치 문서만 검색
          "lg"       — LG에너지솔루션 문서만 검색
          "catl"     — CATL 문서만 검색
          ""         — 전체 문서 검색 (기본값)
        """
        if _init_error:
            return f"[RAG] 초기화 실패 ({_init_error}). 웹 검색 결과를 활용하세요."

        results = rag_retrieve(query, filter_key=filter_key, top_k=top_k)

        if not results:
            return "관련 내부 문서를 찾지 못했습니다."

        parts = []
        for r in results:
            filename = r["metadata"].get("filename", "") or r["source"].split("/")[-1]
            page = r["metadata"].get("page", r["metadata"].get("page_number", ""))
            chunk_index = r["metadata"].get("chunk_index", "")
            date = r["metadata"].get("date", "")
            loc = f" p.{page}" if page != "" else ""
            loc += f" chunk_{chunk_index}" if chunk_index != "" else ""
            parts.append(
                f"[출처: {filename}{loc}]"
                f"{f' [{date}]' if date else ''} "
                f"[유사도: {r['score']:.3f}]\n{r['content']}"
            )

        return "\n\n---\n\n".join(parts)

    return rag_search

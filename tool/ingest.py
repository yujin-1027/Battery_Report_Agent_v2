"""
[tool/ingest.py]
개념: PDF → Qdrant 벡터 DB ingestion 스크립트
기능:
  - battery_report_agent/data/ 폴더의 PDF를 읽어 청크 분할
  - Qwen3-Embedding-0.6B 로 임베딩 후 Qdrant에 업로드
  - collection 없으면 자동 생성 (size=1024, COSINE)

실행:
  python tool/ingest.py              # data/ 폴더 전체 PDF
  python tool/ingest.py --reset      # collection 초기화 후 재ingestion

PDF 메타데이터: 아래 METADATA_TABLE 딕셔너리로 관리.
파일명이 테이블에 없으면 경고 후 건너뜁니다.
"""

import argparse
import time
import unicodedata
import uuid
import torch
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# battery_report_agent 루트를 sys.path에 추가 (tool/ 하위에서 실행 시 대비)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION,
    EMBEDDING_MODEL_NAME, EMBEDDING_VECTOR_DIM, EMBEDDING_BATCH_SIZE,
    DATA_DIR,
)

# 파일명 → 메타데이터 매핑 테이블
# 파일명(확장자 포함)을 key로, 메타데이터를 value로 관리
# 테이블에 없는 파일은 경고 후 건너뜀
METADATA_TABLE: dict[str, dict] = {
    "중국_산업보고서.pdf":  {"source_type": "research", "date": "2025-09"},
    "한국_배터리산업.pdf":  {"source_type": "research", "date": "2024-12"},
    "중국_정책.pdf":        {"source_type": "research", "date": "2024-01"},
    "미국_정책.pdf":        {"source_type": "research", "date": "2024-06"},
    "CATL_ESG.pdf":         {"source_type": "company",  "company": "catl", "date": "2026-03"},
    "Lgen_ESG.pdf":         {"source_type": "company",  "company": "lg",   "date": "2024-06"},
}


def setup_collection(client: QdrantClient, reset: bool = False):
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION in existing:
        if reset:
            print(f"[ingest] collection '{QDRANT_COLLECTION}' 초기화")
            client.delete_collection(QDRANT_COLLECTION)
        else:
            print(f"[ingest] collection '{QDRANT_COLLECTION}' 기존 유지 (--reset 으로 재생성)")
            return

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_VECTOR_DIM, distance=Distance.COSINE),
    )
    print(f"[ingest] collection '{QDRANT_COLLECTION}' 생성 완료 (dim={EMBEDDING_VECTOR_DIM})")


def ingest_pdf(
    path: Path,
    model: SentenceTransformer,
    client: QdrantClient,
    splitter: RecursiveCharacterTextSplitter,
) -> int:
    filename = unicodedata.normalize("NFC", path.name)
    meta = METADATA_TABLE[filename]  # run()에서 사전 검증 후 호출됨

    docs = PyPDFLoader(str(path)).load()
    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["source_type"] = meta["source_type"]
        chunk.metadata["source_domain"] = str(path)
        chunk.metadata["filename"] = filename
        chunk.metadata["date"] = meta.get("date", "")
        chunk.metadata["chunk_index"] = i
        if "company" in meta:
            chunk.metadata["company"] = meta["company"]

    texts = [c.page_content for c in chunks]
    vectors = model.encode(texts, batch_size=EMBEDDING_BATCH_SIZE, show_progress_bar=False).tolist()

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={**chunk.metadata, "text": chunk.page_content},
        )
        for vec, chunk in zip(vectors, chunks)
    ]

    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=QDRANT_COLLECTION, points=points[i:i + batch_size])

    return len(chunks)


def run(reset: bool = False):
    # ── 리소스 초기화 ─────────────────────────────────────────────────────────
    device = (
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[ingest] 디바이스: {device}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

    setup_collection(client, reset=reset)

    # ── PDF 탐색 ──────────────────────────────────────────────────────────────
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"[ingest] {DATA_DIR} 에 PDF 없음. data/ 폴더에 PDF를 넣고 다시 실행하세요.")
        return

    # macOS NFD 파일명을 NFC로 정규화해서 METADATA_TABLE 키와 비교
    def nfc(s: str) -> str:
        return unicodedata.normalize("NFC", s)

    # 테이블에 없는 파일 경고
    unknown = [p.name for p in pdf_files if nfc(p.name) not in METADATA_TABLE]
    if unknown:
        print(f"[ingest] WARNING: METADATA_TABLE에 없는 파일 — 건너뜀: {unknown}")
    pdf_files = [p for p in pdf_files if nfc(p.name) in METADATA_TABLE]

    if not pdf_files:
        print("[ingest] 처리할 PDF 없음. METADATA_TABLE 파일명을 확인하세요.")
        return

    print(f"\n[ingest] {len(pdf_files)}개 PDF 처리 시작")
    total = 0
    for path in pdf_files:
        t0 = time.time()
        n = ingest_pdf(path, model, client, splitter)
        elapsed = time.time() - t0
        print(f"  {path.name[:55]:<55} → {n:>4}청크  {elapsed:.1f}s")
        total += n

    count = client.count(collection_name=QDRANT_COLLECTION).count
    print(f"\n[ingest] 완료 — 총 {total}청크 / Qdrant 저장: {count}개")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF → Qdrant ingestion")
    parser.add_argument("--reset", action="store_true", help="collection 초기화 후 재ingestion")
    args = parser.parse_args()
    run(reset=args.reset)

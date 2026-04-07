"""
[memory/memory_manager.py]
개념: 세션 기반 대화 컨텍스트 영속성 관리
기능:
  - memory_load_node   : session_id 로 이전 대화 컨텍스트 로드 (LangGraph 노드)
  - memory_update_node : 최종 보고서를 세션 파일에 저장 (LangGraph 노드)
저장 방식:
  현재: memory/store/{session_id}.json 파일로 관리 (단순, 빠른 구현)
  예정: step 2 에서 Redis 또는 DB 로 교체 예정 → load/save 함수만 수정하면 됨
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage
from config import MEMORY_STORE_DIR, RESOURCE_SUMMARY_MAX_CHARS


# ── MemoryLoadNode ────────────────────────────────────────────────────────────

def memory_load_node(state: dict) -> dict:
    """
    session_id 로 이전 세션 컨텍스트를 로드.
    - 기존 세션: 저장된 context_summary 를 previous_context 에 반영
    - 신규 세션: session_id 를 UUID 로 새로 생성, previous_context = ""
    """
    # session_id 없으면 새 세션 생성
    session_id = state.get("session_id") or str(uuid.uuid4())
    memory_path: Path = MEMORY_STORE_DIR / f"{session_id}.json"

    if memory_path.exists():
        with open(memory_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        previous_context = saved.get("context_summary", "")
        print(f"[MemoryLoad] 이전 세션 로드 — session_id={session_id}")
    else:
        previous_context = ""
        print(f"[MemoryLoad] 신규 세션 생성 — session_id={session_id}")

    return {
        "session_id":       session_id,
        "previous_context": previous_context,
        "memory_loaded":    True,
    }


# ── MemoryUpdateNode ──────────────────────────────────────────────────────────

def memory_update_node(state: dict) -> dict:
    """
    최종 보고서를 세션 파일에 저장.
    - context_summary : 보고서 앞 500자 → 다음 세션의 previous_context 로 활용
    - report_preview  : 2000자 미리보기 (디버깅·검색용)
    - used_sources    : 보고서에 사용된 URL 목록
    """
    session_id   = state.get("session_id", str(uuid.uuid4()))
    final_report = state.get("final_report") or state.get("report_draft", "")
    used_sources = state.get("used_sources", [])

    # 다음 세션을 위한 500자 요약
    context_summary = final_report[:RESOURCE_SUMMARY_MAX_CHARS] if final_report else ""

    memory_data = {
        "session_id":      session_id,
        "updated_at":      datetime.now().isoformat(),
        "context_summary": context_summary,
        "used_sources":    used_sources,
        "report_preview":  final_report[:2000],   # 미리보기 (2000자)
    }

    memory_path: Path = MEMORY_STORE_DIR / f"{session_id}.json"
    with open(memory_path, "w", encoding="utf-8") as f:
        json.dump(memory_data, f, ensure_ascii=False, indent=2)

    print(f"[MemoryUpdate] 세션 저장 완료 — {memory_path}")

    return {
        "final_report":  final_report,
        "memory_updated": True,
        "messages": [
            HumanMessage(content="최종 보고서가 저장되었습니다.", name="MemoryUpdate")
        ],
    }

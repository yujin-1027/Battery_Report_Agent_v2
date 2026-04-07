"""
[main.py]
개념: 배터리 보고서 에이전트 실행 진입점
기능:
  - run_battery_report() : 사용자 쿼리를 입력받아 전체 파이프라인 실행
                           진행 상황을 스트리밍으로 출력하고 최종 보고서를 파일로 저장
  - __main__             : 기본 예시 쿼리로 에이전트 직접 실행
"""

import uuid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from graph import graph
from agent.report_agent import save_report_to_file

load_dotenv(override=True)


def run_battery_report(
    user_query: str,
    session_id: str | None = None,
) -> str:
    """
    배터리 보고서 에이전트를 실행하고 최종 보고서를 반환.

    Args:
        user_query : 사용자 질의 (예: "LG vs CATL 전략 비교 보고서 작성해줘")
        session_id : 세션 ID (None 이면 새 UUID 자동 생성 → 신규 세션 시작)

    Returns:
        최종 보고서 Markdown 문자열
    """
    session_id = session_id or str(uuid.uuid4())

    config = RunnableConfig(
        recursion_limit=50,                          # 노드 재귀 상한 (충분히 넉넉히)
        configurable={"thread_id": session_id},      # MemorySaver 세션 식별
    )

    # ─ 초기 State 설정 ────────────────────────────────────────────────────────
    initial_state = {
        # 에이전트 메시지 채널
        "messages":           [HumanMessage(content=user_query)],
        "next":               "",

        # 쿼리
        "user_query":         user_query,
        "intent":             "",
        "companies":          [],
        "is_valid":           False,
        "invalid_reason":     "",

        # 메모리
        "session_id":         session_id,
        "previous_context":   "",
        "memory_loaded":      False,

        # 시장 분석
        "market_trends":      [],
        "regulations":        [],
        "industry_resources": [],
        "policy_resources":   [],
        "market_done":        False,

        # LG 분석
        "lg_strategy":        [],
        "lg_swot":            {},
        "lg_financials":      {},
        "lg_resources":       [],
        "lg_done":            False,

        # CATL 분석
        "catl_strategy":      [],
        "catl_swot":          {},
        "catl_financials":    {},
        "catl_resources":     [],
        "catl_done":          False,

        # Aggregator / Critic
        "comparative_swot":   {},
        "bias_check_passed":  False,
        "bias_feedback":      "",

        # 보고서
        "report_draft":       "",
        "report_done":        False,

        # 품질 검사
        "quality_passed":     False,
        "quality_feedback":   "",
        "retry_count":        0,
        "evaluation_result":  {},

        # 최종
        "final_report":       "",
        "memory_updated":     False,
        "used_sources":       [],
    }

    print(f"\n{'='*60}")
    print(f"배터리 보고서 에이전트 시작")
    print(f"세션 ID : {session_id}")
    print(f"쿼리    : {user_query}")
    print(f"{'='*60}\n")

    # ─ 그래프 스트리밍 실행 ──────────────────────────────────────────────────
    final_state = None
    for chunk in graph.stream(initial_state, config, stream_mode="values"):
        # 각 스텝의 최신 메시지 출력 (진행 상황 모니터링)
        if chunk.get("messages"):
            last_msg = chunk["messages"][-1]
            sender   = getattr(last_msg, "name", "Unknown")
            preview  = last_msg.content[:120].replace("\n", " ")
            print(f"  [{sender}] {preview}")
        final_state = chunk

    if final_state is None:
        print("[오류] 그래프 실행 실패 — final_state 없음")
        return "오류: 그래프 실행 실패"

    # ─ 최종 보고서 추출 ──────────────────────────────────────────────────────
    final_report = (
        final_state.get("final_report")          # APPROVED 경로: MemoryUpdate 저장
        or final_state.get("report_draft")        # END_WARNING 경로: 초안 그대로
        or "보고서 생성에 실패했습니다."
    )

    # 유효하지 않은 쿼리인 경우
    if not final_state.get("is_valid", True) and not final_report.strip():
        reason = final_state.get("invalid_reason", "배터리 산업과 무관한 쿼리")
        final_report = f"[거부] 해당 요청을 처리할 수 없습니다.\n사유: {reason}"

    # ─ 파일 저장 ─────────────────────────────────────────────────────────────
    if final_report and "실패" not in final_report and "거부" not in final_report:
        save_report_to_file(final_report, session_id=session_id[:8])

    # ─ 요약 출력 ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("최종 보고서 (앞 500자 미리보기)")
    print(f"{'='*60}")
    print(final_report[:500] + ("..." if len(final_report) > 500 else ""))
    print(f"\n총 {len(final_report):,}자")

    return final_report


# ── 직접 실행 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    QUERY = (
        "LG에너지솔루션과 CATL의 전략을 비교 분석하여 "
        "한국 배터리 산업의 생존 방향을 도출하는 보고서를 작성해주세요."
    )
    run_battery_report(QUERY)

"""
[graph.py]
개념: 배터리 보고서 에이전트 메인 그래프 조립
기능:
  - build_graph() : 전체 노드·엣지를 연결하고 컴파일된 StateGraph 반환
  - graph         : 싱글턴 컴파일 그래프 (main.py 등에서 import 하여 사용)

[그래프 구조 요약]
  START
    → QueryTransform (쿼리 파싱)
      → (is_valid=False) → END (거부 메시지 반환)
      → (is_valid=True)  → MemoryLoad
                           → Supervisor ─┬→ MarketAnalysis → Supervisor
                                         ├→ LGAnalysis     → Supervisor
                                         ├→ CATLAnalysis → Aggregator → Supervisor
                                         ├→ ReportWriter → QualityChecker → Supervisor
                                         ├→ APPROVED → MemoryUpdate → END
                                         └→ END_WARNING → EndWithWarning → END

[설계 원칙]
  - Supervisor 가 모든 라우팅 결정권 보유 (reference 패턴 동일)
  - Aggregator, QualityChecker 는 각 분석 완료 후 자동 실행 (Supervisor 우회)
  - MemorySaver 로 체크포인트 저장 → 세션 재개 가능
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import BatteryReportState
from agent.query_agent     import query_transform_node
from agent.supervisor_agent import supervisor_node, end_with_warning_node
from agent.market_agent    import market_analysis_node
from agent.lg_agent        import lg_analysis_node
from agent.catl_agent      import catl_analysis_node
from agent.report_agent    import aggregator_node, report_writer_node, quality_checker_node
from memory.memory_manager import memory_load_node, memory_update_node


# ── 라우팅 함수 ───────────────────────────────────────────────────────────────

def _get_supervisor_next(state: BatteryReportState) -> str:
    """Supervisor 가 state["next"] 에 기록한 라우팅 대상 반환"""
    return state["next"]


def _check_query_valid(state: BatteryReportState) -> str:
    """쿼리 유효성 결과에 따라 분기"""
    return "valid" if state.get("is_valid", False) else "invalid"


# ── 그래프 빌더 ───────────────────────────────────────────────────────────────

def build_graph():
    """
    배터리 보고서 에이전트 전체 파이프라인을 조립하고 컴파일.
    MemorySaver 체크포인터로 세션 간 상태 유지 (thread_id 로 식별).
    """
    workflow = StateGraph(BatteryReportState)

    # ─ 노드 등록 ─────────────────────────────────────────────────────────────
    workflow.add_node("QueryTransform",  query_transform_node)
    workflow.add_node("MemoryLoad",      memory_load_node)
    workflow.add_node("Supervisor",      supervisor_node)
    workflow.add_node("MarketAnalysis",  market_analysis_node)
    workflow.add_node("LGAnalysis",      lg_analysis_node)
    workflow.add_node("CATLAnalysis",    catl_analysis_node)
    workflow.add_node("Aggregator",      aggregator_node)      # CATLAnalysis 후 자동
    workflow.add_node("ReportWriter",    report_writer_node)
    workflow.add_node("QualityChecker",  quality_checker_node) # ReportWriter 후 자동
    workflow.add_node("MemoryUpdate",    memory_update_node)
    workflow.add_node("EndWithWarning",  end_with_warning_node)

    # ─ 진입 엣지 ─────────────────────────────────────────────────────────────
    workflow.add_edge(START, "QueryTransform")

    # ─ 쿼리 유효성 분기 ──────────────────────────────────────────────────────
    workflow.add_conditional_edges(
        "QueryTransform",
        _check_query_valid,
        {
            "valid":   "MemoryLoad",  # 유효 → 파이프라인 진입
            "invalid": END,           # 무관 쿼리 → 즉시 종료
        }
    )

    # ─ 메모리 로드 → Supervisor ───────────────────────────────────────────────
    workflow.add_edge("MemoryLoad", "Supervisor")

    # ─ Supervisor → 6방향 분기 ───────────────────────────────────────────────
    workflow.add_conditional_edges(
        "Supervisor",
        _get_supervisor_next,
        {
            "MarketAnalysis": "MarketAnalysis",
            "LGAnalysis":     "LGAnalysis",
            "CATLAnalysis":   "CATLAnalysis",
            "ReportWriter":   "ReportWriter",
            "APPROVED":       "MemoryUpdate",    # 승인 → 저장 → END
            "END_WARNING":    "EndWithWarning",   # 한계 초과 → 강제 종료
        }
    )

    # ─ 분석 노드 → Supervisor 복귀 ───────────────────────────────────────────
    workflow.add_edge("MarketAnalysis", "Supervisor")
    workflow.add_edge("LGAnalysis",     "Supervisor")

    # ─ CATL → Aggregator (자동) → Supervisor ─────────────────────────────────
    workflow.add_edge("CATLAnalysis",  "Aggregator")
    workflow.add_edge("Aggregator",    "Supervisor")

    # ─ 보고서 작성 → 품질 검사 (자동) → Supervisor ────────────────────────────
    workflow.add_edge("ReportWriter",  "QualityChecker")
    workflow.add_edge("QualityChecker", "Supervisor")

    # ─ 종료 엣지 ─────────────────────────────────────────────────────────────
    workflow.add_edge("MemoryUpdate",  END)
    workflow.add_edge("EndWithWarning", END)

    return workflow.compile(checkpointer=MemorySaver())


# ── 싱글턴 그래프 인스턴스 (모듈 임포트 시 1회 생성) ─────────────────────────
graph = build_graph()

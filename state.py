"""
[state.py]
개념: 배터리 보고서 에이전트 전체 파이프라인이 공유하는 단일 State 정의
기능:
  - ResourceItem  : 조사 자료 1건의 원문·요약·출처를 구조화한 단위 타입
  - BatteryReportState : 전체 노드가 읽고 쓰는 LangGraph TypedDict 상태

[리소스 관리 전략 — step 별 변화 예정]
  step 1 (현재): raw_content 에 원문 전체 텍스트 저장
  step 2 (예정): raw_content 대신 별도 DB(Vector DB 등)에 저장 후
                  id(UUID)로 참조. id 필드를 미리 포함해 설계.
"""

import operator
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


# ── 기본 자료 단위 ─────────────────────────────────────────────────────────────

class ResourceItem(TypedDict):
    """
    조사 자료 단건 구조체.
    - id         : UUID (step 2에서 DB 기본 키로 활용 예정)
    - raw_content: 원문 전체 텍스트 (step 2에서는 DB 조회 결과)
    - summary    : 500자 이내 요약문
    - source_url : 출처 URL
    - usage_note : 이 자료가 분석에서 어떤 주장/근거에 사용됐는지 한 줄 설명
    """
    id: str
    raw_content: str
    summary: str
    source_url: str
    usage_note: str


# ── 전체 파이프라인 State ──────────────────────────────────────────────────────

class BatteryReportState(TypedDict):
    # ── 에이전트 간 메시지 누적 (Supervisor 라우팅 판단에 활용) ─────────────────
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str   # Supervisor가 결정한 다음 라우팅 대상 노드명

    # ── 1. QueryTransformNode ──────────────────────────────────────────────────
    user_query:     str        # 사용자 원본 쿼리
    intent:         str        # 파싱 의도 (예: "배터리_전략_비교")
    companies:      list[str]  # 분석 대상 기업 (예: ["LG에너지솔루션", "CATL"])
    is_valid:       bool       # 쿼리 유효성 검사 통과 여부
    invalid_reason: str        # is_valid=False 시 사유

    # ── 2. MemoryLoadNode ──────────────────────────────────────────────────────
    session_id:       str   # 세션 식별자 (UUID)
    previous_context: str   # 이전 세션 요약 컨텍스트 (없으면 "")
    memory_loaded:    bool

    # ── 3. MarketAnalysisNode (서브 그래프: IndustryAnalysis + PolicyAnalysis) ──
    industry_resources: list[ResourceItem]  # 산업 분석 조사 결과
    policy_resources:   list[ResourceItem]  # 정책 분석 조사 결과
    market_trends:      list[str]           # ["캐즘 심화", "ESS 성장", ...]
    regulations:        list[str]           # ["IRA", "CRMA", ...]
    market_done:        bool

    # ── 4. LGAnalysisNode ─────────────────────────────────────────────────────
    lg_resources:  list[ResourceItem]
    lg_strategy:   list[str]
    lg_swot:       dict   # {"S": "...", "W": "...", "O": "...", "T": "..."}
    lg_financials: dict   # {"revenue": "...", "operating_profit": "..."}
    lg_done:       bool

    # ── 5. CATLAnalysisNode ───────────────────────────────────────────────────
    catl_resources:  list[ResourceItem]
    catl_strategy:   list[str]
    catl_swot:       dict
    catl_financials: dict
    catl_done:       bool

    # ── 6. AggregatorNode (확증 편향 Critic 포함) ──────────────────────────────
    bias_check_passed: bool  # 양면 검토 통과 여부
    bias_feedback:     str   # 미통과 시 구체적 피드백

    # ── 7. ReportWriterNode ───────────────────────────────────────────────────
    comparative_swot: dict   # {"LG에너지솔루션": {...SWOT}, "CATL": {...SWOT}}
    report_draft:     str    # 보고서 초안 (Markdown)
    report_done:      bool

    # ── 8. QualityCheckerNode (Supervisor 품질 검증용 데이터) ──────────────────
    quality_passed:   bool
    quality_feedback: str   # 재작성 시 반영할 지시
    retry_count:      int   # 재시도 횟수 (MAX_RETRIES 초과 시 END_WARNING)
    evaluation_result: dict  # QualityResult.model_dump() + score 저장

    # ── 9. MemoryUpdateNode ───────────────────────────────────────────────────
    final_report:   str        # 최종 확정 보고서
    memory_updated: bool
    used_sources:   list[str]  # 보고서에 실제 사용된 웹 URL 목록
    rag_sources:    list[str]  # RAG 내부문서 출처 (파일명+페이지) 목록

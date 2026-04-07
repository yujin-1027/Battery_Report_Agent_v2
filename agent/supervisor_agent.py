"""
[agent/supervisor_agent.py]
개념: 전체 파이프라인 라우팅 결정권 보유 Supervisor
기능:
  - QualityResult       : 보고서 품질 체크리스트 스키마 (Pydantic)
  - compute_quality_score : 체크리스트 항목 평균 점수 계산
  - supervisor_node     : 현재 State를 보고 다음 단계 결정 (LangGraph 노드)
  - end_with_warning_node: 최대 재시도 초과 시 강제 종료 (LangGraph 노드)

설계 원칙:
  - Supervisor는 직접 분석·작성하지 않고 항상 위임 (reference 패턴 동일)
  - QualityResult 스키마에서 체크리스트 자동 생성 → 기준 변경 시 스키마만 수정
  - 품질 검사 결과를 메시지에 주입해 Supervisor 판단 근거 제공
"""

from typing import Literal
from pydantic import BaseModel, Field
from pydantic_core import PydanticUndefined
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from config import MODEL_NAME, MODEL_TEMPERATURE, MAX_RETRIES


# ── 라우팅 옵션 (상수로 단일 관리) ───────────────────────────────────────────

ROUTING_OPTIONS = [
    "MarketAnalysis",   # 산업 동향·정책 분석
    "LGAnalysis",       # LG에너지솔루션 분석
    "CATLAnalysis",     # CATL 분석
    "ReportWriter",     # 보고서 작성·재작성
    "APPROVED",         # 승인 → MemoryUpdate → END
    "END_WARNING",      # 최대 재시도 초과 → 강제 종료
]


# ── 품질 평가 체크리스트 스키마 ───────────────────────────────────────────────

class QualityResult(BaseModel):
    """
    보고서 품질 체크리스트.
    ★ 모든 bool 항목이 True 일 때만 passed=True.
    기준 추가·변경 시 이 클래스만 수정하면
    체크리스트 텍스트·점수 계산 모두 자동 반영.
    """
    passed: bool = Field(description="True if ALL criteria are met")

    # ── 체크 항목 ─────────────────────────────────────────────────────────────
    has_executive_summary: bool = Field(
        description="Executive Summary(경영진 요약) 섹션이 있는가"
    )
    has_market_analysis: bool = Field(
        description="글로벌 배터리 시장 동향 및 캐즘 분석이 포함되어 있는가"
    )
    has_company_comparison: bool = Field(
        description="LG에너지솔루션과 CATL의 전략 비교 분석이 있는가"
    )
    has_swot_analysis: bool = Field(
        description="양사의 SWOT 분석이 구체적으로 기술되어 있는가"
    )
    has_strategic_implications: bool = Field(
        description="한국 배터리 산업에 대한 전략적 시사점이 제시되어 있는가"
    )
    has_references: bool = Field(
        description="References 섹션에 출처 URL이 1개 이상 포함되어 있는가"
    )

    issues: list[str] = Field(
        default_factory=list,
        description="미충족 항목 목록. 전부 통과 시 빈 리스트."
    )
    retry_instruction: str = Field(
        default="",
        description="재작성 시 반드시 보완할 구체적 지시. 통과 시 빈 문자열."
    )


# ── 점수 계산 ─────────────────────────────────────────────────────────────────

def compute_quality_score(result: QualityResult) -> float:
    """
    QualityResult 의 bool 체크 항목(passed 제외)의 단순 평균.
    항목 추가·제거 시 자동으로 반영됨.
    """
    criteria = [
        getattr(result, name)
        for name, field in QualityResult.model_fields.items()
        if (
            field.annotation is bool
            and field.default is PydanticUndefined
            and name != "passed"
        )
    ]
    return sum(criteria) / len(criteria) if criteria else 0.0


# ── 체크리스트 텍스트 자동 생성 ──────────────────────────────────────────────

def _build_criteria_text() -> str:
    """QualityResult bool 항목 → 번호 매긴 체크리스트 문자열"""
    lines = []
    for i, (name, field) in enumerate(QualityResult.model_fields.items(), 1):
        if (
            field.annotation is bool
            and field.default is PydanticUndefined
            and name != "passed"
        ):
            lines.append(f"  {i}. {name:<35}: {field.description}")
    return "\n".join(lines)


def _criteria_count() -> int:
    """체크 항목(passed 제외) 개수"""
    return sum(
        1 for name, field in QualityResult.model_fields.items()
        if field.annotation is bool and field.default is PydanticUndefined and name != "passed"
    )


# ── Supervisor 시스템 프롬프트 동적 생성 ──────────────────────────────────────

def _build_supervisor_system_prompt() -> str:
    """
    MAX_RETRIES, ROUTING_OPTIONS 등 코드 상수를 참조해 프롬프트 생성.
    상수 변경 시 프롬프트도 자동 반영.
    """
    return (
        "당신은 배터리 산업 보고서 생성 파이프라인의 Supervisor입니다.\n"
        "모든 라우팅 결정권은 당신에게 있으며, 직접 분석·작성하지 않고 항상 위임합니다.\n\n"

        "## 워크플로우 순서\n"
        "1. MarketAnalysis — 글로벌 배터리 산업 동향 + 정책 분석\n"
        "2. LGAnalysis     — LG에너지솔루션 전략·SWOT·재무 분석\n"
        "3. CATLAnalysis   — CATL 전략·SWOT·재무 분석\n"
        "   (CATLAnalysis 후 Aggregator 자동 실행 → 당신에게 복귀)\n"
        "4. ReportWriter   — 통합 데이터로 보고서 작성\n"
        "   (ReportWriter 후 QualityChecker 자동 실행 → 당신에게 복귀)\n\n"

        "## 라우팅 판단 기준\n"
        "- market_done=False                           → MarketAnalysis\n"
        "- market_done=True,  lg_done=False            → LGAnalysis\n"
        "- lg_done=True,      catl_done=False          → CATLAnalysis\n"
        "- catl_done=True,    report_done=False        → ReportWriter\n"
        "- report_done=True,  quality_passed=True      → APPROVED\n"
        f"- report_done=True,  quality_passed=False,   retry < {MAX_RETRIES}"
        "  → ReportWriter (피드백 반영 재작성)\n"
        f"- retry_count >= {MAX_RETRIES}               → END_WARNING\n\n"

        "## 반드시 준수\n"
        "- 직접 분석·작성 금지 — 항상 위임\n"
        f"- 선택지: {ROUTING_OPTIONS}"
    )


# ── 라우팅 스키마 ─────────────────────────────────────────────────────────────

class RouteResponse(BaseModel):
    next: Literal[
        "MarketAnalysis", "LGAnalysis", "CATLAnalysis",
        "ReportWriter", "APPROVED", "END_WARNING"
    ]


# ── LLM (모듈 로드 시 1회) ────────────────────────────────────────────────────

_llm = ChatOpenAI(model=MODEL_NAME, temperature=MODEL_TEMPERATURE)


# ── Supervisor 노드 ───────────────────────────────────────────────────────────

def supervisor_node(state: dict) -> dict:
    """
    현재 State를 분석해 다음 라우팅 대상을 결정하는 LangGraph 노드.
    - 품질 평가 결과(evaluation_result)를 메시지에 주입해 판단 근거 제공
    - State 요약도 메시지에 주입해 Supervisor가 전체 현황 파악 가능
    """
    messages    = list(state.get("messages", []))
    eval_result = state.get("evaluation_result", {})
    retry_count = state.get("retry_count", 0)
    n_criteria  = _criteria_count()

    # ─ 품질 평가 결과 주입 (있을 때)
    if eval_result:
        score = eval_result.get("score", 0.0)
        messages.append(HumanMessage(
            content=(
                f"[QualityChecker 평가 결과 — 당신이 판단하세요]\n"
                f"통과 여부   : {'PASS' if eval_result.get('passed') else 'FAIL'}\n"
                f"점수        : {score:.2f} "
                f"({int(score * n_criteria)}/{n_criteria} 항목 통과)\n"
                f"실패 항목   : {', '.join(eval_result.get('issues', []))}\n"
                f"재작성 지시 : {eval_result.get('retry_instruction', '')}\n"
                f"현재 재시도 : {retry_count}/{MAX_RETRIES}회"
            ),
            name="QualityChecker"
        ))

    # ─ State 진행 현황 요약 주입
    messages.append(HumanMessage(
        content=(
            f"[State 현황]\n"
            f"market_done={state.get('market_done', False)}, "
            f"lg_done={state.get('lg_done', False)}, "
            f"catl_done={state.get('catl_done', False)}, "
            f"report_done={state.get('report_done', False)}, "
            f"quality_passed={state.get('quality_passed', False)}, "
            f"retry_count={retry_count}"
        ),
        name="StateTracker"
    ))

    # ─ Supervisor 판단
    prompt = ChatPromptTemplate.from_messages([
        ("system", _build_supervisor_system_prompt()),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "다음 단계를 선택하세요. 선택지: {options}"),
    ])

    chain  = prompt | _llm.with_structured_output(RouteResponse, method="function_calling")
    result = chain.invoke({"messages": messages, "options": str(ROUTING_OPTIONS)})

    print(f"[Supervisor] → {result.next}")
    return {"next": result.next}


# ── 강제 종료 노드 ────────────────────────────────────────────────────────────

def end_with_warning_node(state: dict) -> dict:
    """
    최대 재시도(MAX_RETRIES) 초과 시 Supervisor가 선택하는 강제 종료 노드.
    최종 점수·미통과 항목·경고 메시지를 기록하고 종료.
    """
    eval_result = state.get("evaluation_result", {})
    score       = eval_result.get("score", 0.0)

    warning_content = (
        f"[경고] 최대 재시도({MAX_RETRIES}회) 초과로 강제 종료됩니다.\n"
        f"최종 점수  : {score:.2f}\n"
        f"미통과 항목: {', '.join(eval_result.get('issues', []))}\n"
        "현재까지 작성된 보고서가 최종 산출물로 제공됩니다. 품질이 기준에 미달할 수 있습니다."
    )
    print(f"[EndWithWarning] {warning_content}")

    return {
        "final_report": state.get("report_draft", ""),
        "messages": [HumanMessage(content=warning_content, name="System")],
    }

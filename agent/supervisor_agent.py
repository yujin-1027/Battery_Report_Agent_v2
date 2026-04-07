"""
[agent/supervisor_agent.py]
개념: 전체 파이프라인 라우팅 결정권 보유 Supervisor
기능:
  - QualityResult       : 보고서 품질 체크리스트 스키마 (Pydantic)
  - QUALITY_WEIGHTS     : 항목별 가중치 (논문 기반 + 구조 항목)
  - QUALITY_THRESHOLDS  : 항목별 Pass 임계값
  - compute_quality_score : 가중 평균 점수 계산
  - supervisor_node     : 현재 State를 보고 다음 단계 결정 (LangGraph 노드)
  - end_with_warning_node: 최대 재시도 초과 시 강제 종료 (LangGraph 노드)

설계 원칙:
  - Supervisor는 직접 분석·작성하지 않고 항상 위임 (reference 패턴 동일)
  - QualityResult 스키마에서 체크리스트 자동 생성 → 기준 변경 시 스키마만 수정
  - 품질 검사 결과를 메시지에 주입해 Supervisor 판단 근거 제공

평가 기준 출처:
  - RAGAS (2023) arXiv:2309.15217  → Faithfulness, Context Relevance, Answer Relevance
  - FreshLLMs (2023) arXiv:2310.03214 → Temporal Relevance, Source Diversity
  - FRAMES (2024) arXiv:2409.12941   → Factuality, Consistency
  - 팀 Criteria 자체 기준            → Executive Summary, Company Comparison, Strategic Implications, Format Compliance
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

# ── 전체 통과 임계값 ──────────────────────────────────────────────────────────
PASS_THRESHOLD = 0.75  # 가중 평균 점수가 이 값 이상이면 passed=True


# ── 품질 평가 체크리스트 스키마 ───────────────────────────────────────────────

class QualityResult(BaseModel):
    """
    보고서 품질 평가 스키마 (점수 기반, 0.0 ~ 1.0).

    ── 논문 기반 항목 (7개) ──────────────────────────────────────────
    [RAGAS 2023 / arXiv:2309.15217]
      - faithfulness_score    : F = |지지된 주장| / |전체 주장|
      - context_relevance_score: CR = 필요한 문장 수 / 전체 문장 수
      - answer_relevance_score : 보고서가 요청 질문에 답하는 정도

    [FreshLLMs 2023 / arXiv:2310.03214]
      - temporal_relevance_score : 소스 중 2년 이내 비율
      - source_diversity_score   : 인용 도메인 다양성 (단일 쏠림 방지)

    [FRAMES 2024 / arXiv:2409.12941]
      - factuality_score    : LG/CATL 각각 2개 이상 출처 통합 여부
      - consistency_score   : SWOT W/T 항목에 반론/리스크 포함 비율

    ── 팀 Criteria 자체 기준 항목 (4개) ─────────────────────────────
      - executive_summary_score   : Summary ≤ 500자 충족 여부
      - company_comparison_score  : LG vs CATL 비교 분석 완성도
      - strategic_implication_score: 한국 배터리 산업 시사점 포함 여부
      - format_compliance_score   : SWOT 표 형식 + B2B 문체 준수 여부

    ★ 가중 평균 점수 >= PASS_THRESHOLD(0.75) 일 때 passed=True
    기준 추가·변경 시 이 클래스 + QUALITY_WEIGHTS 만 수정하면 자동 반영.
    """
    passed: bool = Field(
        description=f"가중 평균 점수가 {PASS_THRESHOLD} 이상이면 True"
    )

    # ── 논문 기반 항목 ────────────────────────────────────────────────────────

    # [RAGAS 2023] Faithfulness: F = |지지된 주장| / |전체 주장|
    faithfulness_score: float = Field(
        description=(
            "[RAGAS 2023] 보고서 내 전체 주장 중 출처로 뒷받침된 주장의 비율. "
            "Pass 조건: 0.8 이상 (80%% 이상 클레임에 출처 매핑)"
        )
    )

    # [RAGAS 2023] Context Relevance: CR = 필요한 문장 수 / 전체 문장 수
    context_relevance_score: float = Field(
        description=(
            "[RAGAS 2023] 검색된 청크 중 보고서 작성에 실제로 활용된 비율. "
            "Pass 조건: 0.7 이상 (검색 결과의 70%% 이상이 유효하게 사용됨)"
        )
    )

    # [RAGAS 2023] Answer Relevance: 보고서가 요청 질문에 답하는 정도
    answer_relevance_score: float = Field(
        description=(
            "[RAGAS 2023] 보고서가 사용자 요청(LG vs CATL 전략 비교)에 "
            "직접적으로 답하는 정도. Pass 조건: 0.8 이상"
        )
    )

    # [FreshLLMs 2023] Temporal Relevance: 소스 중 3년 이내 비율
    temporal_relevance_score: float = Field(
        description=(
            "[FreshLLMs 2023] 인용된 전체 출처 중 최근 3년 이내 자료의 비율. "
            "Pass 조건: 0.8 이상 (출처의 80%% 이상이 3년 이내)"
        )
    )

    # [FreshLLMs 2023] Source Diversity: 인용 도메인 다양성
    source_diversity_score: float = Field(
        description=(
            "[FreshLLMs 2023] 인용 출처의 도메인 다양성. "
            "단일 도메인 비율이 33%% 이하이고 3개 이상 도메인 사용 시 1.0. "
            "Pass 조건: 0.7 이상"
        )
    )

    # [FRAMES 2024] Factuality: LG/CATL 각각 멀티소스 근거 통합 여부
    factuality_score: float = Field(
        description=(
            "[FRAMES 2024] LG에너지솔루션과 CATL 각각에 대해 "
            "2개 이상의 독립 출처가 통합되었는지 여부. "
            "Pass 조건: 0.8 이상 (양사 모두 멀티소스 근거 보유)"
        )
    )

    # [FRAMES 2024] Consistency: SWOT W/T에 반론/리스크 포함 비율
    consistency_score: float = Field(
        description=(
            "[FRAMES 2024] SWOT의 W(약점)/T(위협) 항목에 "
            "상반된 시각 또는 리스크 관점이 포함된 비율. "
            "Pass 조건: 0.75 이상 (W/T 항목의 75%% 이상에 반론 포함)"
        )
    )

    # ── 팀 Criteria 자체 기준 항목 ────────────────────────────────────────────

    # [팀 기준] Executive Summary ≤ 500자
    executive_summary_score: float = Field(
        description=(
            "[팀 Criteria] Executive Summary 섹션이 존재하고 500자 이내인지 여부. "
            "없으면 0.0 / 있으나 500자 초과 시 0.5 / 충족 시 1.0"
        )
    )

    # [팀 기준] LG vs CATL 비교 분석 완성도
    company_comparison_score: float = Field(
        description=(
            "[팀 Criteria] LG에너지솔루션과 CATL의 전략 비교 분석이 "
            "양사 모두 포함되어 있는 정도. "
            "Pass 조건: 0.8 이상 (양사 비교 섹션 + Comparative SWOT 표 존재)"
        )
    )

    # [팀 기준] 한국 배터리 산업 시사점
    strategic_implication_score: float = Field(
        description=(
            "[팀 Criteria] 한국 배터리 산업의 생존 방향에 대한 "
            "구체적 시사점이 제시되어 있는 정도. "
            "Pass 조건: 0.8 이상 (실행 가능한 시사점 1개 이상)"
        )
    )

    # [팀 기준] SWOT 표 존재 여부
    format_compliance_score: float = Field(
        description=(
            "[팀 Criteria] SWOT 분석이 표(Table) 형식으로 작성되었는지 여부. "
            "Pass 조건: 1.0 (SWOT 표 존재)"
        )
    )

    issues: list[str] = Field(
        default_factory=list,
        description="Pass 조건 미충족 항목 목록. 전부 통과 시 빈 리스트."
    )
    retry_instruction: str = Field(
        default="",
        description="재작성 시 반드시 보완할 구체적 지시. 통과 시 빈 문자열."
    )


# ── 항목별 가중치 (합계 = 1.0) ────────────────────────────────────────────────
# 논문 기반 항목에 더 높은 가중치 부여
# 변경 시 합계가 1.0이 되도록 유지할 것

QUALITY_WEIGHTS: dict[str, float] = {
    # 논문 기반 (총 0.65)
    "faithfulness_score":        0.15,  # RAGAS - 근거성 (핵심)
    "context_relevance_score":   0.08,  # RAGAS - 검색 정밀도
    "answer_relevance_score":    0.10,  # RAGAS - 답변 관련성
    "temporal_relevance_score":  0.12,  # FreshLLMs - 최신성 (핵심)
    "source_diversity_score":    0.10,  # FreshLLMs - 편향 방지
    "factuality_score":          0.05,  # FRAMES - 멀티소스 사실성
    "consistency_score":         0.05,  # FRAMES - 일관성

    # 팀 Criteria 자체 기준 (총 0.35)
    "executive_summary_score":      0.05,  # Summary ≤ 500자
    "company_comparison_score":     0.15,  # LG vs CATL 비교 (핵심)
    "strategic_implication_score":  0.10,  # 시사점
    "format_compliance_score":      0.05,  # 형식 준수
}

# ── 항목별 Pass 임계값 ────────────────────────────────────────────────────────
QUALITY_THRESHOLDS: dict[str, float] = {
    "faithfulness_score":           0.80,
    "context_relevance_score":      0.70,
    "answer_relevance_score":       0.80,
    "temporal_relevance_score":     0.80,
    "source_diversity_score":       0.70,
    "factuality_score":             0.80,
    "consistency_score":            0.75,
    "executive_summary_score":      0.80,
    "company_comparison_score":     0.80,
    "strategic_implication_score":  0.80,
    "format_compliance_score":      1.00,
}


# ── 점수 계산 ─────────────────────────────────────────────────────────────────

def compute_quality_score(result: QualityResult) -> float:
    """
    QualityResult의 float 항목에 QUALITY_WEIGHTS 가중치를 적용한 가중 평균 반환.
    - QUALITY_WEIGHTS에 없는 항목은 자동 제외
    - 항목 추가·제거 시 QUALITY_WEIGHTS만 수정하면 자동 반영
    """
    total_weight = 0.0
    weighted_sum = 0.0

    for name, weight in QUALITY_WEIGHTS.items():
        score = getattr(result, name, None)
        if score is not None:
            weighted_sum += score * weight
            total_weight += weight

    return weighted_sum / total_weight if total_weight > 0 else 0.0


def get_failed_items(result: QualityResult) -> list[str]:
    """
    QUALITY_THRESHOLDS 기준으로 미달 항목 반환.
    issues 필드 자동 생성에 활용.
    """
    failed = []
    for name, threshold in QUALITY_THRESHOLDS.items():
        score = getattr(result, name, None)
        if score is not None and score < threshold:
            failed.append(
                f"{name}: {score:.2f} (기준 {threshold:.2f} 미달)"
            )
    return failed


# ── 체크리스트 텍스트 자동 생성 ──────────────────────────────────────────────

def _build_criteria_text() -> str:
    """QualityResult float 항목 → 번호 매긴 체크리스트 문자열"""
    lines = []
    for i, (name, field) in enumerate(QualityResult.model_fields.items(), 1):
        if (
            field.annotation is float
            and name in QUALITY_WEIGHTS
        ):
            threshold = QUALITY_THRESHOLDS.get(name, "-")
            weight    = QUALITY_WEIGHTS.get(name, 0.0)
            lines.append(
                f"  {i}. {name:<35}: "
                f"Pass >= {threshold} | 가중치 {weight:.2f} | {field.description[:50]}..."
            )
    return "\n".join(lines)


def _criteria_count() -> int:
    """가중치 항목 개수"""
    return len(QUALITY_WEIGHTS)


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
        f"- report_done=True,  quality_passed=True      → APPROVED\n"
        f"- report_done=True,  quality_passed=False,   retry < {MAX_RETRIES}"
        "  → ReportWriter (피드백 반영 재작성)\n"
        f"- retry_count >= {MAX_RETRIES}               → END_WARNING\n\n"

        "## 품질 평가 기준 (가중 평균 점수 기반)\n"
        f"- 전체 통과 임계값: {PASS_THRESHOLD} 이상\n"
        "- 항목별 Pass 조건은 각 필드 description 참조\n\n"

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

        # 항목별 점수 상세 출력
        item_details = []
        for name, threshold in QUALITY_THRESHOLDS.items():
            item_score = eval_result.get(name, None)
            if item_score is not None:
                status = "✓" if item_score >= threshold else "✗"
                item_details.append(
                    f"  {status} {name}: {item_score:.2f} (기준 {threshold:.2f})"
                )

        messages.append(HumanMessage(
            content=(
                f"[QualityChecker 평가 결과 — 당신이 판단하세요]\n"
                f"통과 여부     : {'PASS' if eval_result.get('passed') else 'FAIL'}\n"
                f"가중 평균 점수 : {score:.2f} (통과 기준: {PASS_THRESHOLD})\n"
                f"항목별 점수   :\n" + "\n".join(item_details) + "\n"
                f"실패 항목     : {', '.join(eval_result.get('issues', []))}\n"
                f"재작성 지시   : {eval_result.get('retry_instruction', '')}\n"
                f"현재 재시도   : {retry_count}/{MAX_RETRIES}회"
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

    chain  = prompt | _llm.with_structured_output(RouteResponse)
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
        f"최종 가중 평균 점수 : {score:.2f} (통과 기준: {PASS_THRESHOLD})\n"
        f"미통과 항목        : {', '.join(eval_result.get('issues', []))}\n"
        "현재까지 작성된 보고서가 최종 산출물로 제공됩니다. 품질이 기준에 미달할 수 있습니다."
    )
    print(f"[EndWithWarning] {warning_content}")

    return {
        "final_report": state.get("report_draft", ""),
        "messages": [HumanMessage(content=warning_content, name="System")],
    }
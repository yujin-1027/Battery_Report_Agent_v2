"""
[agent/lg_agent.py]
개념: LG에너지솔루션 전략·SWOT·재무 분석 에이전트
기능:
  - lg_analysis_node : Web + RAG 도구로 LG에너지솔루션 정보 수집 및 구조화
                       이전 시장 분석 결과(market_trends, regulations)를 컨텍스트로 활용
  - 반환 필드: lg_strategy, lg_swot, lg_financials, lg_resources, lg_done
"""

import uuid

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from config import MODEL_NAME, MODEL_TEMPERATURE, RESOURCE_SUMMARY_MAX_CHARS
from state import ResourceItem
from tool.web_search import get_web_search_tool
from tool.rag_retriever import get_rag_retriever_tool
from prompt.company_prompts import LG_RESEARCH_PROMPT


# ── 구조화 출력 스키마 ────────────────────────────────────────────────────────

class CompanyAnalysisOutput(BaseModel):
    """LG에너지솔루션 분석 LLM 구조화 추출 결과"""
    strategy: list[str] = Field(
        description="핵심 사업 전략 목록 (각 항목 1~2문장, 최소 5개)"
    )
    swot: dict = Field(
        description=(
            "SWOT 분석. 반드시 {'S': str, 'W': str, 'O': str, 'T': str} 형태로. "
            "각 항목은 2~3문장으로 구체적으로 기술."
        )
    )
    financials: dict = Field(
        description=(
            "주요 재무 지표. 권장 키: revenue(매출), operating_profit(영업이익), "
            "market_share(시장점유율), order_backlog(수주잔고). "
            "없는 항목은 '정보 없음'으로 기재."
        )
    )
    resources: list[dict] = Field(
        description=(
            "수집된 자료 목록. 각 항목은 "
            "raw_content(원문), summary(500자 이내), source_url(URL) 포함."
        )
    )


# ── 도구 및 LLM 초기화 (모듈 로드 시 1회) ────────────────────────────────────

_web_tool = get_web_search_tool()
_rag_tool = get_rag_retriever_tool()
_tools    = [_web_tool, _rag_tool]
_llm      = ChatOpenAI(model=MODEL_NAME, temperature=MODEL_TEMPERATURE)

# React Agent — 모듈 로드 시 1회 생성
_lg_research_agent = create_react_agent(
    model=_llm,
    tools=_tools,
    prompt=LG_RESEARCH_PROMPT,
)


# ── 헬퍼: ResourceItem 생성 ───────────────────────────────────────────────────

def _make_resource(raw_content: str, summary: str, source_url: str) -> ResourceItem:
    return ResourceItem(
        id=str(uuid.uuid4()),
        raw_content=raw_content,
        summary=summary[:RESOURCE_SUMMARY_MAX_CHARS],
        source_url=source_url,
    )


# ── 노드 함수 ─────────────────────────────────────────────────────────────────

def lg_analysis_node(state: dict) -> dict:
    """
    LG에너지솔루션 전략·SWOT·재무 정보 수집 및 구조화 LangGraph 노드.
    - 시장 분석 결과(market_trends, regulations)를 컨텍스트로 전달해
      시장 맥락에서 LG 전략을 평가하도록 유도
    """
    market_trends = state.get("market_trends", [])
    regulations   = state.get("regulations", [])

    # 시장 맥락을 간략히 추가 (토큰 절약: 상위 3개만)
    context_hint = ""
    if market_trends:
        context_hint = (
            f"\n\n[참고 시장 맥락]\n"
            f"주요 동향: {market_trends[:3]}\n"
            f"주요 규제: {regulations[:3]}"
        )

    research_result = _lg_research_agent.invoke({
        "messages": [HumanMessage(content=(
            f"LG에너지솔루션의 최신 사업 전략, SWOT 분석, 재무 현황을 상세히 조사하세요."
            f"{context_hint}"
        ))]
    })
    research_text = research_result["messages"][-1].content

    # 구조화 추출
    extractor = _llm.with_structured_output(CompanyAnalysisOutput)
    parsed: CompanyAnalysisOutput = extractor.invoke([
        SystemMessage(content=(
            "아래 LG에너지솔루션 리서치 결과를 구조화하세요.\n"
            "swot 은 반드시 {'S': ..., 'W': ..., 'O': ..., 'T': ...} 형태로.\n"
            "각 resource summary 는 500자 이내."
        )),
        HumanMessage(content=research_text),
    ])

    lg_resources = [
        _make_resource(
            raw_content=r.get("raw_content", ""),
            summary=r.get("summary", ""),
            source_url=r.get("source_url", ""),
        )
        for r in parsed.resources
    ]

    print(f"[LGAnalysis] 전략 {len(parsed.strategy)}개, 자료 {len(lg_resources)}건")
    return {
        "lg_strategy":  parsed.strategy,
        "lg_swot":      parsed.swot,
        "lg_financials": parsed.financials,
        "lg_resources": lg_resources,
        "lg_done":      True,
        "messages": [HumanMessage(
            content=(
                f"[LGAnalysis 완료] "
                f"전략 {len(parsed.strategy)}개 | 자료 {len(lg_resources)}건 수집"
            ),
            name="LGAnalysis"
        )],
    }

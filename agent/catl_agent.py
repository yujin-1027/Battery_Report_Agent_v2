"""
[agent/catl_agent.py]
개념: CATL 전략·SWOT·재무 분석 에이전트
기능:
  - catl_analysis_node : Web + RAG 도구로 CATL 정보 수집 및 구조화
                         시장 분석 + LG 분석 결과를 컨텍스트로 활용해 비교 관점 강화
  - 반환 필드: catl_strategy, catl_swot, catl_financials, catl_resources, catl_done
"""

import json
import re
import uuid

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from config import MODEL_NAME, MODEL_TEMPERATURE, RESOURCE_SUMMARY_MAX_CHARS
from state import ResourceItem
from tool.web_search import get_web_search_tool
from tool.rag_retriever import get_rag_retriever_tool
from prompt.company_prompts import CATL_RESEARCH_PROMPT


# ── 구조화 출력 스키마 ────────────────────────────────────────────────────────

class CompanyAnalysisOutput(BaseModel):
    """CATL 분석 LLM 구조화 추출 결과"""
    strategy: list[str] = Field(
        description="핵심 사업 전략 목록 (각 항목 1~2문장, 최소 5개)"
    )
    swot: dict = Field(
        default_factory=dict,
        description=(
            "SWOT 분석. 반드시 {'S': str, 'W': str, 'O': str, 'T': str} 형태로. "
            "각 항목은 2~3문장으로 구체적으로 기술."
        )
    )
    financials: dict = Field(
        default_factory=dict,
        description=(
            "주요 재무 지표. 권장 키: revenue(매출), operating_profit(영업이익), "
            "market_share(시장점유율), order_backlog(수주잔고). "
            "없는 항목은 '정보 없음'으로 기재."
        )
    )
    resources: list[dict] = Field(
        default=[],
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
_catl_research_agent = create_react_agent(
    model=_llm,
    tools=_tools,
    prompt=CATL_RESEARCH_PROMPT,
)


# ── 헬퍼: ResourceItem 생성 ───────────────────────────────────────────────────

def _make_resource(raw_content: str, summary: str, source_url: str, usage_note: str = "") -> ResourceItem:
    return ResourceItem(
        id=str(uuid.uuid4()),
        raw_content=raw_content,
        summary=summary[:RESOURCE_SUMMARY_MAX_CHARS],
        source_url=source_url,
        usage_note=usage_note,
    )


def _extract_resources_from_messages(messages: list) -> list[ResourceItem]:
    resources: list[ResourceItem] = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        raw = msg.content if isinstance(msg.content, str) else str(msg.content)
        parsed_as_json = False
        if raw.strip().startswith("["):
            try:
                items = json.loads(raw)
                if isinstance(items, list) and items and isinstance(items[0], dict):
                    parsed_as_json = True
                    for item in items:
                        url     = item.get("url", "")
                        content = item.get("content", "")
                        title   = item.get("title", "")
                        if not url and not content:
                            continue
                        resources.append(_make_resource(
                            raw_content=content,
                            summary=(title or content)[:RESOURCE_SUMMARY_MAX_CHARS],
                            source_url=url,
                        ))
            except (json.JSONDecodeError, ValueError):
                pass
        if not parsed_as_json:
            for block in raw.split("\n\n---\n\n"):
                block = block.strip()
                if not block:
                    continue
                lines = block.split("\n", 1)
                header = lines[0]
                body   = lines[1].strip() if len(lines) > 1 else block
                source_url = ""
                score_info = ""
                m = re.match(r"\[출처: ([^\]]+)\]", header)
                if m:
                    source_url = m.group(1)
                dm = re.search(r"\[(\d{4}-\d{2})\]", header)
                date_info = f" ({dm.group(1)})" if dm else ""
                sm = re.search(r"\[유사도: ([^\]]+)\]", header)
                score_info = f" 유사도:{sm.group(1)}" if sm else ""
                resources.append(_make_resource(
                    raw_content=body,
                    summary=body[:RESOURCE_SUMMARY_MAX_CHARS],
                    source_url=source_url + date_info + score_info,
                ))
    return resources


# ── 노드 함수 ─────────────────────────────────────────────────────────────────

def catl_analysis_node(state: dict) -> dict:
    """
    CATL 전략·SWOT·재무 정보 수집 및 구조화 LangGraph 노드.
    - 시장 분석 + LG 분석 결과를 컨텍스트로 전달해
      LG와의 비교 관점에서 CATL을 평가하도록 유도
    """
    market_trends = state.get("market_trends", [])
    lg_strategy   = state.get("lg_strategy", [])

    # 비교 관점 컨텍스트 (토큰 절약: 상위 3개만)
    context_hint = ""
    if market_trends:
        context_hint = (
            f"\n\n[비교 맥락]\n"
            f"시장 동향: {market_trends[:3]}\n"
            f"LG 전략(요약): {lg_strategy[:2]}"
        )

    research_result = _catl_research_agent.invoke({
        "messages": [HumanMessage(content=(
            f"CATL의 최신 사업 전략, SWOT 분석, 재무 현황을 상세히 조사하세요."
            f"{context_hint}"
        ))]
    })
    research_text = research_result["messages"][-1].content

    # 도구 호출 결과에서 직접 resources 추출
    catl_resources = _extract_resources_from_messages(research_result["messages"])

    # strategy/swot/financials만 LLM으로 구조화
    extractor = _llm.with_structured_output(CompanyAnalysisOutput, method="function_calling")
    parsed: CompanyAnalysisOutput = extractor.invoke([
        SystemMessage(content=(
            "아래 CATL 리서치 결과를 구조화하세요.\n"
            "swot 은 반드시 {'S': ..., 'W': ..., 'O': ..., 'T': ...} 형태로.\n"
            "resources 필드는 빈 리스트로 반환하세요 (출처는 별도 처리됩니다)."
        )),
        HumanMessage(content=research_text),
    ])

    # RAG 리소스에 usage_note 부여 (RAG 청크만 대상)
    rag_only = [r for r in catl_resources if not r["source_url"].startswith("http")]
    if rag_only:
        chunks_desc = "\n".join(
            f"[{i}] 출처: {r['source_url']}\n내용: {r['raw_content'][:300]}"
            for i, r in enumerate(rag_only)
        )
        note_result = _llm.invoke([
            SystemMessage(content=(
                "아래는 CATL 분석에 사용된 내부 문서 청크 목록입니다.\n"
                "각 청크가 분석의 어떤 주장이나 근거를 뒷받침하는 데 쓰였는지 "
                "한 줄(20자 이내)로 설명하세요.\n"
                "반드시 '[0] 설명', '[1] 설명' 형식으로, 번호 순서대로 출력하세요."
            )),
            HumanMessage(content=(
                f"리서치 요약:\n{research_text[:1000]}\n\n"
                f"청크 목록:\n{chunks_desc}"
            )),
        ])
        notes = {}
        for line in note_result.content.splitlines():
            m = re.match(r"\[(\d+)\]\s*(.+)", line.strip())
            if m:
                notes[int(m.group(1))] = m.group(2).strip()
        for i, r in enumerate(rag_only):
            r["usage_note"] = notes.get(i, "")

    print(f"[CATLAnalysis] 전략 {len(parsed.strategy)}개, 자료 {len(catl_resources)}건")
    return {
        "catl_strategy":  parsed.strategy,
        "catl_swot":      parsed.swot,
        "catl_financials": parsed.financials,
        "catl_resources": catl_resources,
        "catl_done":      True,
        "messages": [HumanMessage(
            content=(
                f"[CATLAnalysis 완료] "
                f"전략 {len(parsed.strategy)}개 | 자료 {len(catl_resources)}건 수집"
            ),
            name="CATLAnalysis"
        )],
    }

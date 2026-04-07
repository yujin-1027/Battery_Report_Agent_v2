"""
[agent/market_agent.py]
개념: 산업 동향 분석 에이전트 (서브 그래프 구조)
기능:
  서브 그래프 내부 노드:
    - industry_analysis_node : 글로벌 배터리 산업 동향 수집 (Web + RAG)
    - policy_analysis_node   : 배터리 관련 규제·정책 수집 (Web + RAG)
    - market_summary_node    : 수집 결과 완료 표시 및 메시지 생성
  메인 그래프 인터페이스:
    - market_subgraph        : 세 노드를 연결한 컴파일된 서브 그래프
    - market_analysis_node   : 메인 그래프(graph.py)에서 호출하는 래퍼 노드 함수

[서브 그래프 설계 이유]
  스펙에서 "산업분석과 정책 분석은 node로" 명시 → 독립 노드로 분리.
  서브 그래프로 캡슐화하여 메인 그래프에서는 단일 노드처럼 사용 가능.
"""

import json
import re
import uuid
import operator
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from config import MODEL_NAME, MODEL_TEMPERATURE, RESOURCE_SUMMARY_MAX_CHARS
from state import ResourceItem
from tool.web_search import get_web_search_tool
from tool.rag_retriever import get_rag_retriever_tool
from prompt.market_prompts import INDUSTRY_ANALYSIS_PROMPT, POLICY_ANALYSIS_PROMPT


# ── 서브 그래프 전용 State ────────────────────────────────────────────────────
# 메인 BatteryReportState 의 부분 집합 — LangGraph가 자동으로 메인 State에 병합

class MarketAnalysisState(TypedDict):
    messages:           Annotated[Sequence[BaseMessage], operator.add]
    industry_resources: list[ResourceItem]
    policy_resources:   list[ResourceItem]
    market_trends:      list[str]
    regulations:        list[str]
    market_done:        bool


# ── 구조화 출력 스키마 ────────────────────────────────────────────────────────

class IndustryAnalysisOutput(BaseModel):
    """산업 분석 LLM 구조화 추출 결과"""
    trends: list[str] = Field(description="주요 배터리 산업 동향 목록 (각 항목 1~2문장)")
    resources: list[dict] = Field(
        default=[],
        description=(
            "수집된 자료 목록. 각 항목은 "
            "raw_content(원문), summary(500자 이내), source_url(URL) 포함"
        )
    )


class PolicyAnalysisOutput(BaseModel):
    """정책 분석 LLM 구조화 추출 결과"""
    regulations: list[str] = Field(description="주요 정책·규제 목록 (각 항목 1~2문장)")
    resources: list[dict] = Field(
        default=[],
        description=(
            "수집된 자료 목록. 각 항목은 "
            "raw_content(원문), summary(500자 이내), source_url(URL) 포함"
        )
    )


# ── 도구 및 LLM 초기화 (모듈 로드 시 1회) ────────────────────────────────────

_web_tool = get_web_search_tool()
_rag_tool = get_rag_retriever_tool()
_tools    = [_web_tool, _rag_tool]
_llm      = ChatOpenAI(model=MODEL_NAME, temperature=MODEL_TEMPERATURE)

# React Agent (도구 사용 가능한 리서치 에이전트) — 모듈 로드 시 1회 생성
_industry_agent = create_react_agent(
    model=_llm,
    tools=_tools,
    prompt=INDUSTRY_ANALYSIS_PROMPT,
)
_policy_agent = create_react_agent(
    model=_llm,
    tools=_tools,
    prompt=POLICY_ANALYSIS_PROMPT,
)


# ── 헬퍼: ResourceItem 생성 ───────────────────────────────────────────────────

def _make_resource(raw_content: str, summary: str, source_url: str, usage_note: str = "") -> ResourceItem:
    """dict → ResourceItem 변환 (id UUID 자동 생성)"""
    return ResourceItem(
        id=str(uuid.uuid4()),
        raw_content=raw_content,
        summary=summary[:RESOURCE_SUMMARY_MAX_CHARS],
        source_url=source_url,
        usage_note=usage_note,
    )


def _extract_resources_from_messages(messages: list) -> list[ResourceItem]:
    """
    ReAct agent 메시지 목록에서 ToolMessage를 파싱해 ResourceItem 리스트 반환.
    - 웹 검색(Tavily): [{"url": ..., "content": ...}, ...] 형태 JSON
    - RAG: "[출처: 파일명 p.N] ..." 형태 텍스트
    """
    resources: list[ResourceItem] = []

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        raw = msg.content if isinstance(msg.content, str) else str(msg.content)

        # 웹 검색 결과: JSON 배열 시도
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

        # RAG 결과: "---" 구분 블록 (JSON 파싱 실패 시 또는 JSON이 아닌 경우)
        if not parsed_as_json:
            for block in raw.split("\n\n---\n\n"):
                block = block.strip()
                if not block:
                    continue
                # "[출처: 파일명 p.N] [유사도: 0.XXX]\n본문..." 형태 파싱
                source_url = ""
                score_info = ""
                lines = block.split("\n", 1)
                header = lines[0]
                body   = lines[1].strip() if len(lines) > 1 else block

                m = re.match(r"\[출처: ([^\]]+)\]", header)
                if m:
                    source_url = m.group(1)
                dm = re.search(r"\[(\d{4}-\d{2})\]", header)
                date_info = f" ({dm.group(1)})" if dm else ""
                sm = re.search(r"\[유사도: ([^\]]+)\]", header)
                if sm:
                    score_info = f" 유사도:{sm.group(1)}"

                resources.append(_make_resource(
                    raw_content=body,
                    summary=(body[:RESOURCE_SUMMARY_MAX_CHARS]),
                    source_url=source_url + date_info + score_info,
                ))

    return resources


# ── 서브 그래프 노드 1: 산업 분석 ────────────────────────────────────────────

def industry_analysis_node(state: MarketAnalysisState) -> dict:
    """
    글로벌 배터리 산업 동향 조사.
    React Agent(web + RAG)로 리서치 후 LLM 구조화 추출.
    """
    # React Agent 리서치
    research_result = _industry_agent.invoke({
        "messages": [HumanMessage(content=(
            "글로벌 배터리 산업의 최신 동향을 조사하세요. "
            "특히 전기차 캐즘, ESS 시장 성장, HEV 전환 트렌드, "
            "주요 기업 수주 및 가동률 현황에 집중하세요."
        ))]
    })
    research_text = research_result["messages"][-1].content

    # 도구 호출 결과에서 직접 resources 추출 (LLM 요약 텍스트에 출처가 소실되므로)
    industry_resources = _extract_resources_from_messages(research_result["messages"])

    # trends만 LLM으로 구조화 (resources는 코드에서 처리)
    extractor = _llm.with_structured_output(IndustryAnalysisOutput, method="function_calling")
    parsed: IndustryAnalysisOutput = extractor.invoke([
        SystemMessage(content=(
            "아래 리서치 결과에서 배터리 산업 동향 목록만 구조화하세요.\n"
            "resources 필드는 빈 리스트로 반환하세요 (출처는 별도 처리됩니다)."
        )),
        HumanMessage(content=research_text),
    ])

    # RAG 리소스에 usage_note 부여
    rag_only = [r for r in industry_resources if not r["source_url"].startswith("http")]
    if rag_only:
        chunks_desc = "\n".join(
            f"[{i}] 출처: {r['source_url']}\n내용: {r['raw_content'][:300]}"
            for i, r in enumerate(rag_only)
        )
        note_result = _llm.invoke([
            SystemMessage(content=(
                "아래는 배터리 산업 동향 분석에 사용된 내부 문서 청크 목록입니다.\n"
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

    print(f"[IndustryAnalysis] 동향 {len(parsed.trends)}개, 자료 {len(industry_resources)}건")
    return {
        "market_trends":      parsed.trends,
        "industry_resources": industry_resources,
        "messages": [HumanMessage(content="산업 분석 완료", name="IndustryAnalysis")],
    }


# ── 서브 그래프 노드 2: 정책 분석 ────────────────────────────────────────────

def policy_analysis_node(state: MarketAnalysisState) -> dict:
    """
    배터리 관련 규제·정책 조사.
    React Agent(web + RAG)로 리서치 후 LLM 구조화 추출.
    """
    research_result = _policy_agent.invoke({
        "messages": [HumanMessage(content=(
            "배터리·전기차 산업에 영향을 미치는 주요 정책과 규제를 조사하세요. "
            "미국 IRA, EU CRMA, 중국 보조금 정책, 탄소중립 2050 등에 집중하세요."
        ))]
    })
    research_text = research_result["messages"][-1].content

    # 도구 호출 결과에서 직접 resources 추출
    policy_resources = _extract_resources_from_messages(research_result["messages"])

    extractor = _llm.with_structured_output(PolicyAnalysisOutput, method="function_calling")
    parsed: PolicyAnalysisOutput = extractor.invoke([
        SystemMessage(content=(
            "아래 리서치 결과에서 배터리 관련 정책·규제 목록만 구조화하세요.\n"
            "resources 필드는 빈 리스트로 반환하세요 (출처는 별도 처리됩니다)."
        )),
        HumanMessage(content=research_text),
    ])

    # RAG 리소스에 usage_note 부여
    rag_only = [r for r in policy_resources if not r["source_url"].startswith("http")]
    if rag_only:
        chunks_desc = "\n".join(
            f"[{i}] 출처: {r['source_url']}\n내용: {r['raw_content'][:300]}"
            for i, r in enumerate(rag_only)
        )
        note_result = _llm.invoke([
            SystemMessage(content=(
                "아래는 배터리 관련 정책·규제 분석에 사용된 내부 문서 청크 목록입니다.\n"
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

    print(f"[PolicyAnalysis] 정책 {len(parsed.regulations)}개, 자료 {len(policy_resources)}건")
    return {
        "regulations":     parsed.regulations,
        "policy_resources": policy_resources,
        "messages": [HumanMessage(content="정책 분석 완료", name="PolicyAnalysis")],
    }


# ── 서브 그래프 노드 3: 마켓 요약 ────────────────────────────────────────────

def market_summary_node(state: MarketAnalysisState) -> dict:
    """서브 그래프 완료 표시 및 완료 요약 메시지 생성"""
    trends_count = len(state.get("market_trends", []))
    regs_count   = len(state.get("regulations", []))
    ind_count    = len(state.get("industry_resources", []))
    pol_count    = len(state.get("policy_resources", []))

    return {
        "market_done": True,
        "messages": [HumanMessage(
            content=(
                f"[MarketAnalysis 서브 그래프 완료]\n"
                f"산업 동향 {trends_count}개 | 정책·규제 {regs_count}개 | "
                f"산업 자료 {ind_count}건 | 정책 자료 {pol_count}건"
            ),
            name="MarketSummary"
        )],
    }


# ── 서브 그래프 조립 ──────────────────────────────────────────────────────────

_market_builder = StateGraph(MarketAnalysisState)
_market_builder.add_node("IndustryAnalysis", industry_analysis_node)
_market_builder.add_node("PolicyAnalysis",   policy_analysis_node)
_market_builder.add_node("MarketSummary",    market_summary_node)

_market_builder.add_edge(START,              "IndustryAnalysis")
_market_builder.add_edge("IndustryAnalysis", "PolicyAnalysis")
_market_builder.add_edge("PolicyAnalysis",   "MarketSummary")
_market_builder.add_edge("MarketSummary",    END)

market_subgraph = _market_builder.compile()   # 재사용 가능한 컴파일된 서브 그래프


# ── 메인 그래프 래퍼 노드 ─────────────────────────────────────────────────────

def market_analysis_node(state: dict) -> dict:
    """
    메인 그래프(graph.py)에서 호출하는 래퍼.
    market_subgraph 를 실행하고 결과를 메인 BatteryReportState 에 반영.
    """
    sub_result = market_subgraph.invoke({
        "messages":           list(state.get("messages", [])),
        "industry_resources": [],
        "policy_resources":   [],
        "market_trends":      [],
        "regulations":        [],
        "market_done":        False,
    })

    # 서브 그래프 결과를 메인 State 필드로 반환
    return {
        "market_trends":      sub_result.get("market_trends", []),
        "regulations":        sub_result.get("regulations", []),
        "industry_resources": sub_result.get("industry_resources", []),
        "policy_resources":   sub_result.get("policy_resources", []),
        "market_done":        sub_result.get("market_done", False),
        "messages": [HumanMessage(
            content=(
                sub_result["messages"][-1].content
                if sub_result.get("messages")
                else "MarketAnalysis 완료"
            ),
            name="MarketAnalysis"
        )],
    }

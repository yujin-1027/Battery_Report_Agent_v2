"""
[agent/report_agent.py]
개념: 보고서 생성 파이프라인 노드 모음
기능:
  - aggregator_node      : 수집 데이터 통합 + 확증 편향 검토(Critic) + used_sources 수집
  - report_writer_node   : 통합 데이터로 Markdown 보고서 작성
  - quality_checker_node : QualityResult 기준으로 보고서 품질 자동 평가
  - save_report_to_file  : 최종 보고서를 output/ 폴더에 .md 파일로 저장

배치 순서 (graph.py):
  CATLAnalysis → [Aggregator 자동] → Supervisor
  ReportWriter → [QualityChecker 자동] → Supervisor
"""

import re
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config import MODEL_NAME, MODEL_TEMPERATURE, OUTPUT_DIR
from agent.supervisor_agent import QualityResult, compute_quality_score
from prompt.report_prompts import REPORT_WRITER_PROMPT, BIAS_CHECK_PROMPT


# ── LLM (모듈 로드 시 1회) ────────────────────────────────────────────────────

_llm = ChatOpenAI(model=MODEL_NAME, temperature=MODEL_TEMPERATURE)


# ── 편향 검토 스키마 ──────────────────────────────────────────────────────────

class BiasCheckResult(BaseModel):
    """확증 편향 검토 결과"""
    passed: bool = Field(description="True if analysis is balanced and unbiased")
    feedback: str = Field(
        default="",
        description="편향 발견 시 구체적 사례 및 개선 방향. 통과 시 빈 문자열."
    )


# ── Aggregator 노드 ───────────────────────────────────────────────────────────

def aggregator_node(state: dict) -> dict:
    """
    CATLAnalysis 완료 후 자동 실행되는 집계 노드.
    1. LG + CATL + 시장 분석 결과를 comparative_swot 으로 통합
    2. 확증 편향 검토 수행 (bias_check_passed, bias_feedback 업데이트)
    3. 전체 사용 출처 URL 수집 (used_sources)
    """
    lg_swot      = state.get("lg_swot", {})
    catl_swot    = state.get("catl_swot", {})
    lg_strategy  = state.get("lg_strategy", [])
    catl_strategy = state.get("catl_strategy", [])

    # ─ 확증 편향 검토
    bias_checker = _llm.with_structured_output(BiasCheckResult, method="function_calling")
    bias_result: BiasCheckResult = bias_checker.invoke([
        SystemMessage(content=BIAS_CHECK_PROMPT),
        HumanMessage(content=(
            f"[LG에너지솔루션]\n"
            f"전략: {lg_strategy}\n"
            f"SWOT: {lg_swot}\n\n"
            f"[CATL]\n"
            f"전략: {catl_strategy}\n"
            f"SWOT: {catl_swot}\n\n"
            "위 두 기업 분석이 균형 있게 이루어졌는지 검토하세요."
        )),
    ])

    # ─ 비교 SWOT 구성
    comparative_swot = {
        "LG에너지솔루션": lg_swot,
        "CATL":           catl_swot,
    }

    # ─ 전체 출처 수집: 웹(URL)과 RAG(파일명) 구분, 섹션 태깅
    section_resources = {
        "시장 분석":    state.get("industry_resources", []),
        "정책·규제":    state.get("policy_resources", []),
        "LG에너지솔루션": state.get("lg_resources", []),
        "CATL":         state.get("catl_resources", []),
    }

    web_sources: list[str] = []
    # rag_sources: {파일명 → {"sections": set, "score": float}}
    rag_map: dict[str, dict] = {}
    web_seen: set[str] = set()

    for section, resources in section_resources.items():
        for r in resources:
            s = r.get("source_url", "").strip() if isinstance(r, dict) else getattr(r, "source_url", "").strip()
            if not s:
                continue
            if s.startswith("http"):
                if s not in web_seen:
                    web_seen.add(s)
                    web_sources.append(s)
            else:
                # 파일명과 유사도 분리: "Lgen_ESG.pdf (2024-06) 유사도:0.852"
                score_m = re.search(r"유사도:([\d.]+)", s)
                score = float(score_m.group(1)) if score_m else 0.0
                file_key = re.sub(r"\s*유사도:[\d.]+", "", s).strip()  # 유사도 제거한 파일명
                usage_note = r.get("usage_note", "") if isinstance(r, dict) else getattr(r, "usage_note", "")
                if file_key not in rag_map:
                    rag_map[file_key] = {"sections": set(), "score": score, "usage_note": usage_note}
                else:
                    rag_map[file_key]["score"] = max(rag_map[file_key]["score"], score)
                    # 가장 구체적인 usage_note 유지 (빈 문자열보다 내용 있는 것 우선)
                    if usage_note and not rag_map[file_key]["usage_note"]:
                        rag_map[file_key]["usage_note"] = usage_note
                rag_map[file_key]["sections"].add(section)

    # 섹션별 그룹으로 정리된 RAG 출처 목록 생성
    rag_sources: list[str] = []
    for file_key, info in rag_map.items():
        sections_str = ", ".join(sorted(info["sections"]))
        score_str = f"유사도:{info['score']:.3f}" if info["score"] > 0 else ""
        entry = f"{file_key}"
        if score_str:
            entry += f" {score_str}"
        entry += f" [사용 섹션: {sections_str}]"
        if info.get("usage_note"):
            entry += f" → {info['usage_note']}"
        rag_sources.append(entry)

    used_sources = web_sources  # 기존 필드는 URL만 유지 (report_writer가 링크로 사용)

    bias_status = "통과" if bias_result.passed else f"미통과 — {bias_result.feedback[:80]}"
    print(f"[Aggregator] 편향 검토={bias_status} | 웹출처 {len(web_sources)}건 | RAG출처 {len(rag_map)}건(중복제거)")

    return {
        "comparative_swot":  comparative_swot,
        "bias_check_passed": bias_result.passed,
        "bias_feedback":     bias_result.feedback,
        "used_sources":      used_sources,
        "rag_sources":       rag_sources,
        "messages": [HumanMessage(
            content=(
                f"[Aggregator 완료] "
                f"편향 검토={'통과' if bias_result.passed else '미통과'} | "
                f"웹출처 {len(web_sources)}건 | RAG출처 {len(rag_sources)}건"
            ),
            name="Aggregator"
        )],
    }


# ── 보고서 작성 노드 ──────────────────────────────────────────────────────────

def report_writer_node(state: dict) -> dict:
    """
    수집·집계된 데이터로 Markdown 보고서를 작성하는 LangGraph 노드.
    재작성 시 quality_feedback(이전 검토 지시)와 bias_feedback(편향 주의)를 프롬프트에 주입.
    """
    quality_feedback = state.get("quality_feedback", "")
    bias_feedback    = state.get("bias_feedback", "")
    bias_passed      = state.get("bias_check_passed", True)

    # 피드백 섹션 구성 (재작성 시에만 추가)
    feedback_section = (
        f"\n\n## ⚠️ 이전 검토 피드백 (반드시 반영)\n{quality_feedback}"
        if quality_feedback else ""
    )
    bias_section = (
        f"\n\n## ⚠️ 확증 편향 주의사항 (반드시 수정)\n{bias_feedback}"
        if bias_feedback and not bias_passed else ""
    )

    # 수집된 모든 데이터를 보고서 작성 요청에 포함
    web_sources = state.get('used_sources', [])
    rag_sources = state.get('rag_sources', [])
    report_request = (
        f"## 시장 동향\n{state.get('market_trends', [])}\n\n"
        f"## 규제·정책\n{state.get('regulations', [])}\n\n"
        f"## LG에너지솔루션\n"
        f"- 전략: {state.get('lg_strategy', [])}\n"
        f"- SWOT: {state.get('lg_swot', {})}\n"
        f"- 재무: {state.get('lg_financials', {})}\n\n"
        f"## CATL\n"
        f"- 전략: {state.get('catl_strategy', [])}\n"
        f"- SWOT: {state.get('catl_swot', {})}\n"
        f"- 재무: {state.get('catl_financials', {})}\n\n"
        f"## 웹 출처 (URL — 링크로 표시)\n{web_sources}\n\n"
        f"## RAG 내부문서 출처 (파일명+페이지 — 텍스트로 표시)\n{rag_sources}"
        f"{feedback_section}{bias_section}"
    )

    response = _llm.invoke([
        SystemMessage(content=REPORT_WRITER_PROMPT),
        HumanMessage(content=report_request),
    ])
    report_draft = response.content

    print(f"[ReportWriter] 보고서 {len(report_draft):,}자 작성 완료")
    return {
        "report_draft":     report_draft,
        "report_done":      True,
        "quality_passed":   False,    # 작성 직후 품질 검사 미수행 상태로 초기화
        "evaluation_result": {},      # 이전 평가 결과 초기화
        "messages": [HumanMessage(
            content=f"[ReportWriter 완료] {len(report_draft):,}자 보고서 작성",
            name="ReportWriter"
        )],
    }


# ── 품질 검사 노드 ────────────────────────────────────────────────────────────

def quality_checker_node(state: dict) -> dict:
    """
    작성된 보고서를 QualityResult 체크리스트로 자동 평가하는 LangGraph 노드.
    평가 결과(evaluation_result)를 State 에 저장 후 Supervisor 로 복귀.
    재시도·종료 결정은 Supervisor 가 담당.
    """
    report_draft  = state.get("report_draft", "")
    current_retry = state.get("retry_count", 0)

    checker = _llm.with_structured_output(QualityResult, method="function_calling")
    result: QualityResult = checker.invoke([
        SystemMessage(content=(
            "당신은 배터리 산업 보고서의 품질 평가 전문가입니다.\n"
            "아래 보고서를 체크리스트 기준으로 엄격하게 평가하세요.\n"
            "passed=True 는 모든 항목이 True 일 때만 가능합니다.\n"
            "미충족 항목은 issues 에 구체적으로 기재하고,\n"
            "retry_instruction 에 재작성 시 반드시 보완할 지시를 작성하세요."
        )),
        HumanMessage(content=f"평가할 보고서:\n\n{report_draft}"),
    ])

    score = compute_quality_score(result)

    print(
        f"[QualityChecker] {'PASS' if result.passed else 'FAIL'} | "
        f"점수: {score:.2f} | 실패 항목: {result.issues}"
    )
    return {
        "quality_passed":   result.passed,
        "quality_feedback": result.retry_instruction,
        "retry_count":      current_retry + (0 if result.passed else 1),  # 실패 시 카운트 증가
        "evaluation_result": {**result.model_dump(), "score": score},
        "messages": [HumanMessage(
            content=(
                f"[QualityChecker] {'PASS' if result.passed else 'FAIL'} | "
                f"점수: {score:.2f}"
            ),
            name="QualityChecker"
        )],
    }


# ── 보고서 파일 저장 유틸리티 ─────────────────────────────────────────────────

def save_report_to_file(report: str, session_id: str = "default") -> Path:
    """
    최종 보고서를 output/ 폴더에 Markdown 파일로 저장.
    파일명: battery_report_{session_id}_{timestamp}.md
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = OUTPUT_DIR / f"battery_report_{session_id}_{timestamp}.md"
    filename.write_text(report, encoding="utf-8")
    print(f"[보고서 저장] {filename}")
    return filename

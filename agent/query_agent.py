"""
[agent/query_agent.py]
개념: 사용자 쿼리 파싱 및 유효성 검사 노드
기능:
  - QueryParseResult : LLM 구조화 출력 스키마 (Pydantic)
  - query_transform_node : 쿼리 → 의도(intent) · 기업(companies) · 유효성(is_valid) 추출
  유효하지 않은 쿼리(배터리/에너지 산업 무관)는 즉시 END 로 분기됨 (graph.py 참고)
"""

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import MODEL_NAME, MODEL_TEMPERATURE


# ── 파싱 결과 스키마 ──────────────────────────────────────────────────────────

class QueryParseResult(BaseModel):
    """사용자 쿼리 파싱 구조화 결과"""
    intent: str = Field(
        description=(
            "파싱된 의도. 다음 중 하나: "
            "'배터리_전략_비교' / '산업_동향_분석' / '기업_분석' / '기타'"
        )
    )
    companies: list[str] = Field(
        description=(
            "언급된 기업명 목록. 예: ['LG에너지솔루션', 'CATL']. "
            "언급 없으면 빈 리스트."
        )
    )
    is_valid: bool = Field(
        description=(
            "배터리·에너지 산업·전기차·ESS 관련 분석 요청이면 True. "
            "완전히 무관한 요청(날씨, 요리 등)이면 False."
        )
    )
    invalid_reason: str = Field(
        default="",
        description="is_valid=False 일 때 구체적인 사유. 통과 시 빈 문자열."
    )


# ── LLM 초기화 (모듈 로드 시 1회) ────────────────────────────────────────────

_llm    = ChatOpenAI(model=MODEL_NAME, temperature=MODEL_TEMPERATURE)
_parser = _llm.with_structured_output(QueryParseResult)  # 구조화 출력 바인딩


# ── 노드 함수 ─────────────────────────────────────────────────────────────────

def query_transform_node(state: dict) -> dict:
    """
    사용자 쿼리를 파싱하고 유효성을 검사하는 LangGraph 노드.
    반환 필드: intent, companies, is_valid, invalid_reason
    """
    user_query = state["user_query"]

    result: QueryParseResult = _parser.invoke([
        SystemMessage(content=(
            "당신은 배터리·에너지 산업 보고서 요청 분석 전문가입니다.\n"
            "사용자 쿼리에서 아래 4가지를 정확히 추출하세요:\n\n"
            "1. intent: 요청 의도 유형\n"
            "   - 배터리_전략_비교 : 두 기업 이상의 전략을 비교 요청\n"
            "   - 산업_동향_분석  : 시장·트렌드 분석 요청\n"
            "   - 기업_분석      : 특정 기업 단독 분석 요청\n"
            "   - 기타           : 위에 해당하지 않는 경우\n\n"
            "2. companies: 언급된 기업명 목록 (정식 명칭으로 통일)\n"
            "   예) 'LG엔솔' → 'LG에너지솔루션', '닝더스다이' → 'CATL'\n\n"
            "3. is_valid: 배터리·전기차·에너지 저장·ESS·배터리 소재 관련이면 True\n\n"
            "4. invalid_reason: is_valid=False 일 때 사유"
            "규칙:"
            "- 배터리 산업, EV, ESS, 관련 에너지 기업과 관련된 쿼리만 허용"
            "- 쿼리가 모호한 경우, 무효로 표시하기 전에 변형 시도"
            ""
        )),
        HumanMessage(content=user_query),
    ])

    print(
        f"[QueryTransform] intent={result.intent}, "
        f"companies={result.companies}, is_valid={result.is_valid}"
    )

    return {
        "intent":         result.intent,
        "companies":      result.companies,
        "is_valid":       result.is_valid,
        "invalid_reason": result.invalid_reason,
    }

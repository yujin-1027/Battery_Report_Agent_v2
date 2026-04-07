"""
[tool/web_search.py]
개념: 웹 검색 도구 인터페이스 및 Stub
기능:
  - WebSearchResult : 검색 결과 단건의 기대 타입 정의
  - get_web_search_tool() : LangChain Tool 객체 반환
  현재: TavilySearch를 기본 구현체로 사용
  교체: 다른 팀 구현체 완성 시 이 함수 내부만 교체하면 전체 에이전트에 반영

[다른 팀을 위한 기대 출력 타입 — WebSearchResult]
  title   : str  — 기사/페이지 제목
  url     : str  — 출처 URL (ResourceItem.source_url 로 연결)
  content : str  — 원문 전체 텍스트 (ResourceItem.raw_content 로 연결)
  snippet : str  — 200자 이내 짧은 발췌 요약
"""

from typing import TypedDict
from langchain_community.tools.tavily_search import TavilySearchResults
from config import WEB_SEARCH_MAX_RESULTS


class WebSearchResult(TypedDict):
    """
    웹 검색 단건 결과 기대 타입.
    다른 팀 구현 시 이 스키마를 준수해야 함.
    """
    title:   str   # 기사/페이지 제목
    url:     str   # 출처 URL → ResourceItem.source_url
    content: str   # 원문 전체 텍스트 → ResourceItem.raw_content
    snippet: str   # 200자 이내 발췌 → ResourceItem.summary 초안으로 활용


def get_web_search_tool(max_results: int = WEB_SEARCH_MAX_RESULTS):
    """
    웹 검색 LangChain Tool 반환.
    - 현재: TavilySearchResults (Tavily API 키 필요)
    - 교체: 다른 팀 구현체로 교체 시 이 함수만 수정
    """
    return TavilySearchResults(
        max_results=max_results,
        # include_raw_content=True 로 설정하면 전체 원문 포함 (토큰 증가 주의)
    )

QUERY_PARSING_PROMPT = """
You are a query parser for a B2B battery industry research system.

Analyze the user's query and extract the following:
- intent: the analytical goal (e.g., "배터리_전략_비교")
- companies: list of companies mentioned (e.g., ["LG에너지솔루션", "CATL"])
- is_valid: whether the query is relevant to battery industry analysis
- invalid_reason: if invalid, explain why

Rules:
- Only accept queries related to battery industry, EV, ESS, or related energy companies
- If the query is ambiguous, attempt transformation before marking invalid
- Output in JSON format only

Query: {user_query}
"""

"""
번역:
당신은 B2B 배터리 산업 연구 시스템의 쿼리 파서입니다.
사용자의 쿼리를 분석하여 다음을 추출하세요:
- intent: 분석 목표 (예: "배터리_전략_비교")
- companies: 언급된 기업 목록 (예: ["LG에너지솔루션",
    "CATL"])
- is_valid: 쿼리가 배터리 산업 분석과 관련 있는지 여부
- invalid_reason: 관련 없는 경우 그 이유 설명
규칙:
- 배터리 산업, EV, ESS, 관련 에너지 기업과 관련된 쿼리만 허용
- 쿼리가 모호한 경우, 무효로 표시하기 전에 변형 시도
- JSON 형식으로 출력

"""
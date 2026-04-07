REPORT_GENERATION_PROMPT = """
You are a B2B research report writer.

Using the provided market analysis, company analyses (LG에너지솔루션 & CATL), 
and comparative data, generate a structured report.

Format requirements:
- Writing style: formal B2B Korean (B2B 서비스 기준 문체)
- Summary section: half page or less
- SWOT comparison: TABLE format (not prose)
- Markdown output

Report structure:
1. Executive Summary (반 페이지 이하)
2. 글로벌 배터리 시장 현황
3. LG에너지솔루션 전략 분석
4. CATL 전략 분석
5. Comparative SWOT (표)
6. 종합 시사점 및 한국 배터리 산업 생존 방향
7. References (실제 사용된 URL만)

Input:
- market_trends: {market_trends}
- regulations: {regulations}
- lg_summary: {lg_summary}          ← 구조 파악용
- catl_summary: {catl_summary}      ← 구조 파악용
- lg_raw_sources: {lg_raw_sources}  ← 근거 작성용 (필수)
- catl_raw_sources: {catl_raw_sources} ← 근거 작성용 (필수)
- market_raw_sources: {market_raw_sources}
- policy_raw_sources: {policy_raw_sources}

Note: 보고서의 모든 주장은 raw_sources 기반으로 작성하고,
      summary는 구조 참고용으로만 사용할 것
"""
"""Dense 문서 구성 실험. 원문과 BM25 필드는 변경하지 않는다."""
import re

PROFILES = {
    "baseline": ("servNm", "target_text", "benefit_text", "apply_text"),
    "content": ("servNm", "summary_text", "benefit_text", "target_text", "criteria_text"),
    "core": ("servNm", "summary_text", "benefit_text"),
}
_NO_INFO = re.compile(r"(?:해당\s*없음|없음|해당\s*사항\s*없음|붙임\s*참조|상동|지원\s*대상(?:의\s*내용)?을\s*참고해\s*주시기\s*바랍니다)[.!\s]*")


def build_texts(table, profile="baseline"):
    """baseline은 기존 문자열을 그대로 보존. 새 구성은 안내문과 중복 필드만 정리."""
    if profile not in PROFILES:
        raise ValueError(f"알 수 없는 Dense 문서 구성: {profile}")
    fields = PROFILES[profile]
    missing = [field for field in fields if field not in table]
    if missing:
        raise ValueError(f"필수 검색 칼럼 없음: {missing}")
    values = table[list(fields)].fillna("")
    if profile == "baseline":
        return values.agg(" ".join, axis=1).tolist()
    documents = []
    for row in values.itertuples(index=False, name=None):
        parts, seen = [], set()
        for value in row:
            text = re.sub(r"\s+", " ", str(value)).strip()
            if not text or _NO_INFO.fullmatch(text) or text in seen:
                continue
            seen.add(text)
            parts.append(text)
        documents.append("\n".join(parts))
    return documents

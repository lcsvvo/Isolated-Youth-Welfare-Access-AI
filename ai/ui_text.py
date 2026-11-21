"""Conservative ending changes for UI summaries; source fields are never mutated."""
import re

_VERBS = "지원|제공|운영|지급|감면|면제|대여|연계|실시|진행|보장|확인|선정|부담|도모|제고|향상"

def _has_final(text):
    return bool(text) and "가" <= text[-1] <= "힣" and (ord(text[-1])-ord("가")) % 28 != 0

def _sentence(text):
    body = text.strip().rstrip(".!。")
    if not body:
        return text
    for old,new in (("있습니다","있어요"),("없습니다","없어요"),("됩니다","돼요"),("합니다","해요"),
                    ("있음","있어요"),("없음","없어요"),("가능함","가능해요"),("필요함","필요해요")):
        if body.endswith(old):
            return body[:-len(old)]+new+"."
    if body.endswith("입니다"):
        stem=body[:-3]
        return stem+("이에요" if _has_final(stem) else "예요")+"."
    if re.search(r"(?:해요|돼요|어요|아요|예요|이에요|세요|니다)$",body):
        return body+"."
    match = re.fullmatch(r"(.*?)("+_VERBS+r")함",body)
    if match:
        return match[1]+match[2]+"해요."
    match = re.fullmatch(r"(.+?)\s+("+_VERBS+r")",body)
    if match:
        stem,verb=match.groups()
        particle="" if stem.endswith(("을","를")) else ("을" if _has_final(stem) else "를")
        return stem+particle+" "+verb+"해요."
    # Do not guess the grammar or eligibility meaning of arbitrary source text.
    return "안내 내용은 ‘"+body+"’예요."

def friendly_summary(text):
    text=str(text or "").strip()
    if not text:
        return "상세 화면에서 서비스 안내를 확인해 주세요."
    return "\n".join(_sentence(line) for line in re.split(r"\n+|(?<=[.!?])\s+(?=[가-힣])",text) if line.strip())

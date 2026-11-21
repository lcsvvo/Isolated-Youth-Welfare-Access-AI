"""GPU·모델 없이 파이프라인 연결만 확인하기 위한 가짜 모델.
진짜 모델과 함수 이름이 같아서, --mock만 빼면 그대로 진짜 모델로 바뀐다."""
import re
import zlib
import numpy as np

_DOMAIN_WORDS = {
    "미취업(구직)": ["취업", "일자리", "구직", "면접"], "주거": ["집", "월세", "전세", "이사"],
    "건강": ["우울", "상담", "병원", "잠"], "관계참여": ["친구", "외로", "사람"],
    "경제": ["돈", "생활비", "빚"], "교육훈련": ["자격증", "공부", "교육"],
}


class MockEmbedder:
    def encode(self, texts):
        vecs = np.zeros((len(texts), 512), dtype=np.float32)
        for i, t in enumerate(texts):
            for j in range(len(t) - 1):
                vecs[i, zlib.crc32(t[j:j + 2].encode()) % 512] += 1
        return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)


class MockLLM:
    def chat_json(self, system, user, task=None, retries=1):
        if task == "crisis":
            return {"crisis": False, "evidence": ""}
        if task == "extract":
            text = user.split("입력:")[-1].strip()
            sents = [s for s in re.split(r"(?<=[.?!])\s*", text) if s]
            needs = []
            for d, words in _DOMAIN_WORDS.items():
                hit = next((s for s in sents if any(w in s for w in words)), None)
                if hit:
                    needs.append({"domain": d, "level": "main", "evidence": hit})
            outing = next((s for s in sents if "밖" in s and "부담" in s), None)
            return {"needs": needs, "activity_range": "집 안" if outing else "판단 불가",
                    "activity_evidence": outing, "constraints": [], "age": None, "region_sido": None}
        if task == "explain":
            src = user.split("[서비스 원문]")[-1].strip().split("\n")[0]
            return {"reason": "입력하신 상황과 관련된 서비스예요.",
                    "how_to_apply": "원문의 신청 방법을 확인해 주세요.",
                    "check_needed": ["나이·지역 조건"], "evidence": src[:40]}
        return {}

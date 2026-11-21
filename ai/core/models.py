"""모델을 '불러오는' 곳. 다른 코드는 모델 종류를 몰라도 encode()/chat_json()만 부르면 된다.
모델을 바꾸고 싶으면 이 파일(과 config.yaml)만 고치면 된다."""
from core.utils import parse_json


class Embedder:
    """임베딩 모델: 문장 → 숫자 벡터."""

    def __init__(self, name, cache_dir=None, *, device=None, batch_size=32,
                 max_length=None, revision=None, local_files_only=False):
        import hashlib
        import json
        from pathlib import Path
        import sentence_transformers
        from sentence_transformers import SentenceTransformer
        if batch_size < 1 or (max_length is not None and max_length < 1):
            raise ValueError("batch_size와 max_length는 양수여야 합니다")
        self.batch_size = batch_size
        self.model = SentenceTransformer(name, cache_folder=cache_dir,
                                         device=device, revision=revision,
                                         local_files_only=local_files_only)
        if max_length is not None:
            self.model.max_seq_length = max_length
        config = getattr(getattr(self.model[0], "auto_model", None), "config", None)
        local = Path(name)
        files = [(str(p.relative_to(local)), p.stat().st_size, p.stat().st_mtime_ns)
                 for p in sorted(local.rglob("*")) if p.is_file()] if local.is_dir() else None
        self.cache_key = hashlib.sha256(json.dumps({
            "name": str(local.resolve()) if local.is_dir() else name,
            "revision": getattr(config, "_commit_hash", None) or revision,
            "local_files": files, "max_length": self.model.max_seq_length,
            "sentence_transformers": sentence_transformers.__version__,
            "normalize": True,
        }, sort_keys=True).encode()).hexdigest()

    def encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True,
                                 batch_size=self.batch_size, show_progress_bar=False)


class LLM:
    """언어 모델: 지시문 + 입력 → 답변. Colab T4는 bf16 미지원이라 float16 사용."""

    def __init__(self, name, cache_dir=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, cache_dir=cache_dir, torch_dtype=torch.float16, device_map="auto")

    def chat(self, system, user, max_new_tokens=512):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        inputs = self.tok.apply_chat_template(msgs, add_generation_prompt=True,
                                              return_dict=True, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def chat_json(self, system, user, task=None, retries=1):
        # TODO: vLLM guided decoding으로 바꾸면 JSON 깨짐이 거의 사라진다.
        # greedy 디코딩이라 같은 길이로 다시 돌리면 같은 출력이 나옴 → 재시도마다 생성 길이를 두 배로
        # (kanana_v3 추출 실패 10건이 모두 13~29초로 긴 생성 → 출력이 512토큰에서 잘린 것으로 추정)
        last_err, self.last_raw = None, None
        for i in range(retries + 1):
            raw = None
            try:
                raw = self.chat(system, user, max_new_tokens=512 * 2 ** i)
                return parse_json(raw)
            except Exception as e:
                last_err, self.last_raw = e, raw   # 평가에서 실패 원인을 보려고 마지막 원문을 남김
        raise last_err


def load_models(cfg, mock=False):
    if mock:
        from core.mock import MockEmbedder, MockLLM
        return MockEmbedder(), MockLLM()
    cache = cfg["paths"].get("cache_dir")
    import os
    batch_size = int(os.environ.get("BANGEOL_EMBED_BATCH_SIZE", "32"))
    return Embedder(cfg["models"]["embedder"], cache, batch_size=batch_size), LLM(cfg["models"]["llm"], cache)

"""2-b 의미 검색. 모델·문서·순서를 검증한 캐시만 재사용한다."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np


def cache_path_for(template, profile):
    """문서 구성별 캐시 경로. config의 `{profile}` 자리에 구성 이름을 넣는다.

    구성마다 파일을 나눠 baseline·core 전환 시 서로의 캐시를 덮어쓰지 않게 한다.
    `{profile}`이 없는 경로는 그대로 쓰며, 이때도 지문 검사로 다른 구성의 임베딩은 재사용하지 않는다.
    """
    if not template:
        return None
    if not isinstance(profile, str) or not profile or any(c in profile for c in "/\\{}"):
        raise ValueError(f"Dense 문서 구성 이름이 잘못되었습니다: {profile!r}")
    return str(template).replace("{profile}", profile)


class DenseIndex:
    def __init__(self, embedder, ids, texts, cache_path=None):
        self.embedder, self.ids = embedder, list(ids)
        texts = list(texts)
        if len(self.ids) != len(texts):
            raise ValueError("서비스 ID와 문서 수가 다릅니다")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("서비스 ID가 중복되었습니다")
        if not all(isinstance(t, str) for t in texts):
            raise ValueError("검색 문서는 문자열이어야 합니다")
        identity = getattr(embedder, "cache_key", None)
        # 모델 식별 정보가 없는 외부 embedder는 디스크 캐시를 사용하지 않는다.
        self.cache_path = Path(cache_path) if cache_path and identity and texts else None
        payload = {"version": 1, "model": identity,
                   "documents": list(zip(map(str, self.ids), texts))}
        self.fingerprint = hashlib.sha256(json.dumps(
            payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        self.cache_hit = False
        if not texts:
            self.emb = np.empty((0, 0), dtype=np.float32)
            return
        if self.cache_path and self.cache_path.exists():
            try:
                # 기존 메타데이터 없는 .npy 캐시는 안전하게 다시 계산한다.
                with np.load(self.cache_path, allow_pickle=False) as cached:
                    if cached["fingerprint"].item() == self.fingerprint:
                        self.emb = self._vectors(cached["emb"], len(texts))
                        self.cache_hit = True
            except (OSError, ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile):
                pass
        if not self.cache_hit:
            self.emb = self._vectors(embedder.encode(texts), len(texts))
            if self.cache_path:
                self._save()

    @staticmethod
    def _vectors(value, rows):
        vectors = np.asarray(value, dtype=np.float32)
        if (vectors.ndim != 2 or vectors.shape[0] != rows or
                vectors.shape[1] == 0 or not np.isfinite(vectors).all()):
            raise ValueError("임베딩 크기 또는 값이 잘못되었습니다")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("영벡터 임베딩은 검색할 수 없습니다")
        return vectors / norms

    def _save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        name = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.cache_path.parent,
                                             suffix=".tmp", delete=False) as stream:
                name = stream.name
                np.savez(stream, fingerprint=self.fingerprint, emb=self.emb)
            os.replace(name, self.cache_path)
        finally:
            if name and os.path.exists(name):
                os.unlink(name)

    def search_with_scores(self, query, k=100):
        """정규화 내적(코사인 유사도). 동점은 입력 서비스 순서를 유지한다."""
        if not isinstance(query, str) or not query.strip() or k <= 0 or not self.ids:
            return []
        q = self._vectors(self.embedder.encode([query]), 1)[0]
        if q.shape[0] != self.emb.shape[1]:
            raise ValueError("질의와 문서 임베딩 차원이 다릅니다")
        scores = self.emb @ q
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.ids[i], float(scores[i])) for i in order]

    def search(self, query, k=100):
        return [sid for sid, _ in self.search_with_scores(query, k)]

    def search_batch_with_scores(self, queries, k=100):
        """개발 평가용 일괄 검색. 빈 질의는 같은 위치에 빈 결과를 반환한다."""
        queries = list(queries)
        results = [[] for _ in queries]
        valid = [i for i, query in enumerate(queries) if isinstance(query, str) and query.strip()]
        if k <= 0 or not self.ids or not valid:
            return results
        vectors = self._vectors(self.embedder.encode([queries[i] for i in valid]), len(valid))
        if vectors.shape[1] != self.emb.shape[1]:
            raise ValueError("질의와 문서 임베딩 차원이 다릅니다")
        for position, vector in zip(valid, vectors):
            scores = self.emb @ vector
            order = np.argsort(-scores, kind="stable")[:k]
            results[position] = [(self.ids[i], float(scores[i])) for i in order]
        return results

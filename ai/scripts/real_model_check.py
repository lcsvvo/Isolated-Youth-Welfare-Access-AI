"""GPU smoke check with synthetic examples; never substitutes mock models."""
import json
import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def measure_flow(pipe, explain):
    from confirm.questions import pending
    started = time.perf_counter()
    normal = pipe.prepare("서울에 사는 25살이고 월세와 생활비가 걱정돼요.")
    report = {"prepare_seconds": round(time.perf_counter() - started, 2)}
    started = time.perf_counter()
    if normal["status"] == "awaiting_confirm":
        normal = pipe.resume(normal, {"type": "finish_confirm"})
    question_seconds = []
    report["confirm_seconds"] = round(time.perf_counter() - started, 2)
    for _ in range(8):
        if normal["status"] != "awaiting_answer":
            break
        started = time.perf_counter()
        question = pending(normal)
        normal = pipe.resume(normal, {"type": "answer", "target": question["slot"],
                                     "value": None, "answer_type": "건너뜀"})
        question_seconds.append(round(time.perf_counter() - started, 3))
    report["answer_seconds"] = question_seconds
    started = time.perf_counter()
    out = pipe.search(normal, explain_mode="source_first")
    report["normal"] = {"status": out["status"], "retrieval_status": out["retrieval_status"],
                        "cards": len(out["cards"]), "search_seconds": round(time.perf_counter()-started, 2),
                        "service_ids": [c["servId"] for c in out["cards"]],
                        "source_first": all(c.get("explanation_pending") for c in out["cards"]),
                        "reasons_present": all(bool(c.get("reason")) for c in out["cards"]),
                        "timings_ms": out.get("timings_ms", {})}
    if explain and out["cards"]:
        started = time.perf_counter()
        card = pipe.explain_service(normal, out, out["cards"][0]["servId"])
        report["detail"] = {"seconds": round(time.perf_counter() - started, 2),
                            "explanation_kind": card.get("explanation_kind"), "reason_present": bool(card.get("reason"))}
    return report


def main(explain=False):
    import torch
    from pipeline import Pipeline
    if not torch.cuda.is_available():
        raise RuntimeError("GPU가 연결되지 않았습니다. Colab 런타임에서 T4를 선택해 주세요.")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    pipe = Pipeline(mock=False)
    report = {"gpu": torch.cuda.get_device_name(0), "models": pipe.cfg["models"],
              "load_seconds": round(time.perf_counter() - started, 2),
              "load_gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
              "load_gpu_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
              "services": len(pipe.services), "q1_services": len(pipe.review.q1), "q2_pairs": len(pipe.review.q2)}
    for label in ("first_run", "warm_run"):
        torch.cuda.reset_peak_memory_stats()
        measured = measure_flow(pipe, explain)
        measured["gpu_peak_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        measured["gpu_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
        report[label] = measured
    unsure = pipe.prepare("모르겠어요")
    report["unsure"] = {"status": unsure["status"], "needs": len(unsure["confirmed_profile"].get("needs", []))}
    crisis = pipe.prepare("지금 자해할 생각이 있어요")
    stopped = pipe.search(crisis)
    report["crisis"] = {"status": crisis["status"], "searched": stopped["searched"], "cards": len(stopped["cards"])}
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    for label in ("first_run", "warm_run"):
        normal = report[label]["normal"]
        if (normal["retrieval_status"] not in ("ok", "partial") or not 1 <= normal["cards"] <= 5
                or not normal["reasons_present"] or not normal["source_first"]):
            raise RuntimeError(f"{label} 목록과 원문 안내를 확인해 주세요.")
    if crisis["status"] != "crisis_stop" or stopped["searched"] or stopped["cards"]:
        raise RuntimeError("위기 문장의 추천 중단 확인이 필요합니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explain", action="store_true", help="상세 1건의 AI 설명 시간도 측정")
    main(explain=parser.parse_args().explain)

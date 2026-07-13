"""실제 Cerebras로 상황 분류·생성 품질과 지연시간을 측정한다."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.models import GenerateRequest, Target, Tone
from app.service import (
    ExcuseGenerationService,
    apply_severity_guardrails,
    count_sentences,
    validate_grounding,
    validate_situation_fit,
)


DEFAULT_CASES = ROOT / "evals" / "situation_cases.json"
DEFAULT_REPORT = ROOT / "evals" / "latest-situation-report.json"
HUMOR_MARKERS = ("ㅋㅋ", "ㅎㅎ", "우주의 기운", "외계인", "마법", "운명")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--concurrency", type=int, default=2)
    return parser.parse_args()


def load_cases(path: Path, limit: int | None, ids: list[str] | None) -> list[dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if ids:
        selected = set(ids)
        cases = [case for case in cases if case["id"] in selected]
    if limit:
        cases = cases[:limit]
    return cases


async def evaluate_case(case: dict[str, Any], settings: Settings) -> dict[str, Any]:
    request = GenerateRequest(
        situation=case["situation"],
        target=Target(case["target"]),
        targetDescription=case.get("targetDescription"),
        tone=Tone(case["tone"]),
    )
    service = ExcuseGenerationService(settings)
    original_classify = service.client.classify_situation
    original_generate = service.client.generate
    metrics: dict[str, Any] = {
        "classificationMs": 0,
        "generationCalls": 0,
        "profile": None,
        "generatedCandidates": [],
    }

    async def tracked_classify(*args: Any, **kwargs: Any):
        started = time.perf_counter()
        profile = await original_classify(*args, **kwargs)
        metrics["classificationMs"] = round((time.perf_counter() - started) * 1000)
        metrics["profile"] = profile
        return profile

    async def tracked_generate(*args: Any, **kwargs: Any):
        metrics["generationCalls"] += 1
        generated = await original_generate(*args, **kwargs)
        metrics["generatedCandidates"].append(
            {"excuse": generated.excuse, "replyOptions": generated.replyOptions}
        )
        return generated

    service.client.classify_situation = tracked_classify
    service.client.generate = tracked_generate
    started = time.perf_counter()
    try:
        result = await service.generate(request, f"eval-{case['id']}")
        total_ms = round((time.perf_counter() - started) * 1000)
        profile = apply_severity_guardrails(metrics["profile"], request)
        fit_issues = validate_situation_fit(result, profile)
        grounding_issues = validate_grounding(result, request)
        serious_humor = profile.severity.value == "SERIOUS" and any(
            marker in result.excuse for marker in HUMOR_MARKERS
        )
        return {
            **case,
            "actualSeverity": profile.severity.value,
            "severityMatched": profile.severity.value == case["expectedSeverity"],
            "profile": profile.model_dump(mode="json"),
            "excuse": result.excuse,
            "length": len(result.excuse.strip()),
            "sentenceCount": count_sentences(result.excuse),
            "fitIssues": fit_issues,
            "groundingIssues": grounding_issues,
            "seriousHumorViolation": serious_humor,
            "generationCalls": metrics["generationCalls"],
            "regenerated": metrics["generationCalls"] > 1,
            "classificationMs": metrics["classificationMs"],
            "totalMs": total_ms,
            "error": None,
            "generatedCandidates": metrics["generatedCandidates"],
        }
    except Exception as exc:  # 평가 전체를 중단하지 않고 실패 사례를 보고서에 남긴다.
        return {
            **case,
            "actualSeverity": None,
            "severityMatched": False,
            "profile": None,
            "excuse": None,
            "length": 0,
            "sentenceCount": 0,
            "fitIssues": [],
            "groundingIssues": [],
            "seriousHumorViolation": False,
            "generationCalls": metrics["generationCalls"],
            "regenerated": metrics["generationCalls"] > 1,
            "classificationMs": metrics["classificationMs"],
            "totalMs": round((time.perf_counter() - started) * 1000),
            "error": f"{type(exc).__name__}: {exc}",
            "generatedCandidates": metrics["generatedCandidates"],
        }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [result for result in results if result["error"] is None]
    latencies = [result["totalMs"] for result in completed]
    by_expected: dict[str, dict[str, Any]] = {}
    for severity in ("LIGHT", "NORMAL", "SERIOUS"):
        group = [result for result in results if result["expectedSeverity"] == severity]
        matched = sum(result["severityMatched"] for result in group)
        by_expected[severity] = {
            "count": len(group),
            "matched": matched,
            "accuracy": round(matched / len(group), 3) if group else None,
            "regenerationRate": round(
                sum(result["regenerated"] for result in group) / len(group), 3
            ) if group else None,
        }
    serious = [
        result
        for result in completed
        if result["expectedSeverity"] == "SERIOUS"
    ]
    light = [
        result for result in completed if result["expectedSeverity"] == "LIGHT"
    ]
    classification_latencies = [
        result["classificationMs"] for result in completed
    ]
    return {
        "caseCount": len(results),
        "completedCount": len(completed),
        "errorCount": len(results) - len(completed),
        "severityAccuracy": round(
            sum(result["severityMatched"] for result in results) / len(results), 3
        ) if results else 0,
        "fitPassRate": round(
            sum(not result["fitIssues"] for result in completed) / len(completed), 3
        ) if completed else 0,
        "groundingPassRate": round(
            sum(not result["groundingIssues"] for result in completed) / len(completed), 3
        ) if completed else 0,
        "regenerationRate": round(
            sum(result["regenerated"] for result in completed) / len(completed), 3
        ) if completed else 0,
        "seriousHumorViolations": sum(
            result["seriousHumorViolation"] for result in completed
        ),
        "seriousThreeSentenceRate": round(
            sum(result["sentenceCount"] >= 3 for result in serious) / len(serious), 3
        ) if serious else None,
        "lightAtMostTwoSentenceRate": round(
            sum(result["sentenceCount"] <= 2 for result in light) / len(light), 3
        ) if light else None,
        "averageClassificationMs": round(statistics.mean(classification_latencies))
        if classification_latencies else None,
        "averageTotalMs": round(statistics.mean(latencies)) if latencies else None,
        "p95TotalMs": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]
        if latencies else None,
        "byExpectedSeverity": by_expected,
    }


async def main() -> None:
    args = parse_args()
    settings = Settings()
    if not settings.cerebras_api_key:
        raise SystemExit("CEREBRAS_API_KEY가 필요합니다.")
    cases = load_cases(args.cases, args.limit, args.ids)
    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    async def limited(case: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            result = await evaluate_case(case, settings)
            status = "PASS" if result["severityMatched"] and not result["error"] else "CHECK"
            print(
                f"[{status}] {result['id']} expected={result['expectedSeverity']} "
                f"actual={result['actualSeverity']} calls={result['generationCalls']} "
                f"latency={result['totalMs']}ms"
            )
            return result

    results = await asyncio.gather(*(limited(case) for case in cases))
    report = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "model": settings.cerebras_model,
        "summary": summarize(results),
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"report={args.report}")


if __name__ == "__main__":
    asyncio.run(main())

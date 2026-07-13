import re
from enum import StrEnum

from app.models import ExcuseResult


class SafetyAction(StrEnum):
    ALLOW = "ALLOW"
    TRANSFORM = "TRANSFORM"
    BLOCK = "BLOCK"


# Keep these patterns narrow: ordinary requests may contain "거짓말" or
# "변명", so broad keyword blocking would create unnecessary false positives.
BLOCK_PATTERNS = (
    r"신분(?:증)?\s*(?:위조|도용|사칭)",
    r"(?:가짜|위조)\s*(?:영수증|증거|서류)",
    r"증거\s*(?:조작|삭제|은폐)",
    r"사기\s*(?:치는|방법|수법|치는법)",
    r"(?:죽여|해치|폭행|협박)\s*(?:버려|방법|하는법)?",
    r"자해\s*(?:방법|하는법|도구)",
    r"(?:혐오|차별)\s*(?:표현|발언|문구)",
    r"api[_ -]?key|authorization\s*:\s*bearer|sk-[a-zA-Z0-9_-]{12,}",
)

TRANSFORM_PATTERNS = (
    r"책임\s*피하",
    r"들키지\s*않",
    r"완벽하게\s*속",
    r"추궁을\s*피",
)


def classify_input(*texts: str | None) -> SafetyAction:
    content = "\n".join(text or "" for text in texts).lower()
    if any(re.search(pattern, content, re.IGNORECASE) for pattern in BLOCK_PATTERNS):
        return SafetyAction.BLOCK
    if any(re.search(pattern, content, re.IGNORECASE) for pattern in TRANSFORM_PATTERNS):
        return SafetyAction.TRANSFORM
    return SafetyAction.ALLOW


def validate_output_safety(result: ExcuseResult) -> None:
    output = "\n".join(
        [
            result.excuse,
            result.recommendedAction,
            result.likelyFollowUp,
            *result.replyOptions,
            *result.riskFactors,
            *(item.when for item in result.aftermath),
            *(item.question for item in result.aftermath),
            *result.remember,
        ]
    )
    if classify_input(output) == SafetyAction.BLOCK:
        raise ValueError("generated content violates the safety policy")
    if "```" in output or "[SYSTEM" in output.upper() or "[MEMORY_DATA" in output.upper():
        raise ValueError("generated content contains internal prompt or markdown markers")

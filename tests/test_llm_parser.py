"""제공자 응답의 관대한 파싱 회귀 테스트."""

import json

from app.llm import CerebrasClient


def test_parser_keeps_excuse_when_optional_fields_are_missing() -> None:
    body = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {"excuse": "회의 시간을 잘못 봤어요. 지금 바로 들어갈게요."},
                        ensure_ascii=False,
                    )
                },
            }
        ]
    }

    result = CerebrasClient._parse_result(body)

    assert result.excuse == "회의 시간을 잘못 봤어요. 지금 바로 들어갈게요."
    assert result.replyOptions == [result.excuse, result.excuse]
    assert result.successRate == 50


def test_parser_keeps_plain_text_response() -> None:
    body = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "지금 확인해서 바로 공유할게요."},
            }
        ]
    }

    result = CerebrasClient._parse_result(body)

    assert result.excuse == "지금 확인해서 바로 공유할게요."

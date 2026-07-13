"""제공자 응답의 관대한 파싱 회귀 테스트."""

import json
from types import SimpleNamespace

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


def test_payload_uses_cerebras_strict_structured_output() -> None:
    client = SimpleNamespace(
        settings=SimpleNamespace(
            cerebras_model="test-model",
            temperature=0.7,
            reasoning_effort="low",
        )
    )

    payload = CerebrasClient._build_payload(client, "system", "user", 300)

    response_format = payload["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "tongchoo_excuse_v1"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["aftermath"]["items"]["additionalProperties"] is False

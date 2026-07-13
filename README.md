# TongChoo AI Server

FastAPI service that receives sanitized context from the Spring Web server and calls OpenAI with non-streaming Structured Outputs.

## Responsibilities

- Prompt construction for create/evolve/reply modes
- Input safety classification and output safety validation
- OpenAI SDK 호출, JSON 파싱, 스키마 검증
- 요청 ID·모델·토큰 사용량 로깅과 명확한 provider 오류 응답
- No direct database access

Spring remains responsible for authentication, DB memory queries, XP/grade rules, ERD persistence, and the final domain transaction.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8001
```

`OPENAI_API_KEY` is required for the generation endpoint. `/health` does not call OpenAI.

The AI call uses the official OpenAI Python SDK directly. Structured Outputs handles the JSON shape, and Pydantic performs the final application-level validation.

## Internal API

`POST /internal/v1/excuses/generate`

```json
{
  "mode": "REPLY",
  "situation": "팀 회의에 20분 늦었다",
  "target": "TEAM_LEAD",
  "tone": "SLICK",
  "memory": "최근 변명 참고 데이터",
  "rootExcuse": "회의 시작 시간을 잘못 봤어요.",
  "conversation": [
    {"role": "assistant", "content": "회의 시작 시간을 잘못 봤어요."},
    {"role": "user", "content": "그래도 왜 미리 말하지 않았어요?"}
  ],
  "currentExcuse": "회의 시작 시간을 잘못 봐서 20분 늦었어요.",
  "incomingMessage": "그래도 왜 미리 말하지 않았어요?",
  "roundNumber": 3,
  "evolveDirection": null
}
```

`conversation` is the current branch selected by Spring from the root excuse. It accepts up to 10 turns with `user` or `assistant` roles. `CREATE` does not need a conversation, `EVOLVE` requires `currentExcuse`, and `REPLY` requires `incomingMessage` plus an existing excuse or conversation context.

The Spring-facing response uses the same core shape as `ExcuseResponse`: `analysis` contains `successRate`, `realism`, `persuasion`, `suspicionLevel`, and `riskFactors`; `aftermath` contains `dayOffset`, `when`, `question`, and `collapseRate`. `id`, parent IDs, XP, complexity warnings, and timestamps remain Spring-owned fields.

The response also includes `recommendedAction`, `likelyFollowUp`, and 2-3 `replyOptions` so the user can choose a natural message and prepare for the next question. The former flat provider response is available only at `/internal/v1/excuses/generate/raw` for diagnostics.

For Spring's evolve request, `direction` is accepted as an alias for `evolveDirection`. Spring should still enrich the internal request with `situation`, `target`, `tone`, and the current branch context because the AI server does not access the database.

Set `INTERNAL_SERVICE_TOKEN` in deployed environments. Spring sends it as `Authorization: Bearer <token>`.

Every generation request may send `X-Request-ID`. If omitted, FastAPI creates one and returns it in the response header. The same ID is used in API and OpenAI logs.

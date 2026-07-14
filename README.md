# TongChoo AI Server

FastAPI AI server for TongChoo. Spring assembles the context and owns the database;
FastAPI calls Cerebras and returns a strict JSON response.

## Responsibilities

- create/reply prompt construction
- Cerebras `gpt-oss-120b` chat completion
- non-streaming Structured Outputs JSON schema validation
- request-ID based logging
- no direct database access

Spring remains responsible for authentication, DB memory queries, reply
lineage, XP/grade rules, ERD persistence, and the final domain transaction. See
[SPRING_INTEGRATION.md](./SPRING_INTEGRATION.md) for the exact integration contract.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8001
```

## Test

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Set `CEREBRAS_API_KEY` before calling a generation endpoint. `/health` only checks
configuration and does not call Cerebras.

The default generation policy is:

- `stream=false`
- `max_completion_tokens=1400`
- at most 2 attempts inside one provider operation
- at most 4 provider calls across classification, quality regeneration, and retries
- 18 seconds for the complete FastAPI request (Spring waits 20 seconds)
- if the first response is truncated, retry once with 1800 tokens
- `reasoning_effort=low`
- `memory` maximum 12,000 characters

### Situation-aware CREATE/REPLY

Before CREATE generation, FastAPI classifies the original situation as `LIGHT`,
`NORMAL`, or `SERIOUS` with a separate Structured Output call. Classification
uses a low temperature of `0.1` by default and considers actual impact, recovery
difficulty, relationship formality, accountability, next action, and whether
humor is safe. Ambiguous situations default to `NORMAL`.

The resulting `SituationProfile` supplies sentence and character ranges to both
the prompt and a local quality gate:

- `LIGHT`: 1–2 sentences, 20–100 characters
- `NORMAL`: 2–3 sentences, 60–180 characters
- `SERIOUS`: 3–5 sentences, 120–350 characters

Spring stores the CREATE response's `situationSeverity` on the conversation.
Every REPLY sends that persisted value back, so FastAPI skips classification and
cannot downgrade a serious conversation based on the latest short message. REPLY
then permits a shorter response. A result that misses its range or
required next action is regenerated once by
default with specific correction reasons. If both valid JSON candidates still
miss some quality rules, the higher-scoring non-empty result is returned rather
than failing the whole CREATE request.

Configure this flow with `CLASSIFICATION_TEMPERATURE` and
`SITUATION_QUALITY_MAX_ATTEMPTS`.

The reproducible real-provider benchmark, fixed 21-case dataset, and latest
result are available in [`evals/`](./evals/README.md). The service also removes
unsupported generated time promises and rejects candidates that invent a cause
or claim a concrete impact absent from the input.

### REPLY quality gate

For `REPLY`, the server compares the three generated candidates with
the previous assistant turns and with each other. The default similarity limit
is 0.9 and can be changed with `REPLY_SIMILARITY_THRESHOLD`. Clear why/when/how
questions are also checked for a corresponding reason, time, or action answer.

When a result fails these checks, FastAPI regenerates it once by default with the
rejection reasons supplied as correction context. The number of quality attempts
is controlled by `REPLY_QUALITY_MAX_ATTEMPTS`. If the final replacement still
fails, it returns HTTP 422 with `REPLY_QUALITY_REJECTED`. Spring should
send the current conversation branch on every REPLY call so prior answers can be
compared.

### Aftermath quality gate

`aftermath[].question` must be a direct question that probes a claim, cause,
time, evidence, promise, or contradiction in the generated `excuse`. Generic
work follow-ups such as asking for the meeting agenda or the next schedule are
rejected. CREATE regenerates an invalid aftermath up to
`AFTERMATH_QUALITY_MAX_ATTEMPTS`; REPLY applies the same aftermath validation as
part of its reply quality gate. CREATE returns the higher-scoring grounded
candidate when only non-critical structure rules remain; if every candidate
adds unsupported facts, it returns HTTP 422 with `GROUNDING_QUALITY_REJECTED`.

## Internal API

Spring should call one of these endpoints:

```text
POST /internal/v1/excuses/create
POST /internal/v1/excuses/reply
```

The compatibility endpoint `/internal/v1/excuses/generate` accepts the generic
request with an explicit `mode`. The raw provider-shaped response is available
only at `/internal/v1/excuses/generate/raw` for diagnostics.

### Create

```json
{
  "situation": "팀 회의에 20분 늦었다",
  "target": "TEAM_LEAD",
  "tone": "MILD"
}
```

### Reply

```json
{
  "situation": "팀 회의에 20분 늦었다",
  "target": "TEAM_LEAD",
  "tone": "MILD",
  "situationSeverity": "NORMAL",
  "currentExcuse": "회의 시작 시간을 잘못 봐서 20분 늦었어요.",
  "incomingMessage": "그래도 왜 미리 말하지 않았어요?",
  "conversation": [
    {"role": "assistant", "message": "회의 시작 시간을 잘못 봤어요."}
  ],
  "roundNumber": 3
}
```

`conversation` contains only turns before `currentExcuse`; the latest incoming
message is sent only as `incomingMessage`. This removes repeated prompt context
and accepts up to 10 prior turns. `roundNumber` accepts 1~5 because the service limits reply preparation
to five rounds.
The Spring-facing response matches the Java client contract: `excuse`,
`replyOptions`, score fields, `suspicionLevel`, `situationSeverity`, `riskFactors`,
`rememberItems`, and `aftermaths`. `replyOptions` always contains exactly three
distinct candidates and keeps the generated order:
short/direct, polite/responsible, then light relationship-repair wording.
IDs, reply lineage, XP, complexity warnings, and timestamps remain Spring-owned
fields.

Set `INTERNAL_SERVICE_TOKEN` in deployed environments. Spring sends it as
`Authorization: Bearer <token>`.

Every request may send `X-Request-ID`. If omitted, FastAPI creates one and returns
it in the response header.

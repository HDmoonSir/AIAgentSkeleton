# Integration Guide

이 골격은 "외부 AI 모델 API를 도구로 호출하는 에이전트"를 그대로 두고,
**모델 API 어댑터와 도구만 교체**하면 실제 서비스가 되도록 설계되어 있다.

## 경계 원칙

- 코어(`aiagent/`)는 구체 구현을 모른다. 포트(추상)만 받는다.
- 구체 의존성(HTTP 클라이언트, 모델 API 어댑터, 대화 스토어)은 **진입점(`main.py` lifespan)**에서만 생성한다.
- 설정 분리: 추론 LLM은 코어(`LLM_*`), 외부 모델 API/서버는 진입점(`MODEL_API_*`, `APP_*`).

## 교체 절차

### 1. 모델 API 어댑터 구현

`InferenceModelClient`(`aiagent/ports/inference_model.py`) 포트를 구현한다.
기본 제공 `HttpInferenceModelClient`를 실제 API에 맞게 수정하거나, 새 어댑터를 작성한다.

- 요청 변환: `_build_payload`
- 응답 변환: `_parse_response`
- 인프라 오류는 `InferenceModelError`로 감싼다.

작성한 어댑터를 `main.py` lifespan에서 생성해 `assemble_orchestrator`에 주입한다.

### 2. 도구 추가/변경

`aiagent/tools/tool_registry.py`의 `build_tools`에서 포트를 주입받아 LangChain 도구를 등록한다.
도구 이름 상수를 정의하고, 의도별 시나리오(`agent_orchestrator.DICT_SCENARIO_TOOL_NAMES`)에 매핑한다.

### 3. 의도 추가

`aiagent/constants.py`의 `IntentType`·`DICT_INTENT_DESCRIPTION`에 의도를 추가하고,
오케스트레이터 시나리오 매핑과 프롬프트(`prompt_builder.py`)를 확장한다.

### 4. 대화 이력 백엔드 교체

`ConversationStore`(`app/agent/conversation_store.py`)를 DB 구현으로 교체하고
lifespan에서 주입 대상을 바꾼다. 코어/라우트는 수정하지 않는다.

## 상태

| 구성요소 | 상태 |
|----------|------|
| 에이전트 골격(LangGraph 4노드) | 제공 |
| 모델 호출 포트/계약 | 제공 |
| HTTP 어댑터 | 예시 제공 (실제 API 스키마에 맞춰 수정 필요) |
| 도구(run_model_inference) | 예시 제공 |
| 대화 이력 | 메모리 구현 (운영 시 DB로 교체) |

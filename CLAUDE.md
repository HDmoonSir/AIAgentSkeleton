# Agent AI Skeleton

외부 AI 모델 API를 **도구(tool)로 호출하는** LangGraph + FastAPI 에이전트 서비스 골격.
도메인 로직은 비어 있고, 모델 어댑터와 도구만 교체하면 실제 서비스가 되도록 설계됐다.

## 경계 원칙 (가장 중요)

- 코어(`aiagent/`)는 DB·프레임워크·모델 구현을 **모른다**. 포트(추상)만 의존한다.
- 구체 의존성(HTTP 클라이언트·모델 API 어댑터·대화 스토어)은 **진입점(`main.py` lifespan)에서만** 생성·주입한다(composition root).
- 설정 분리: 추론 LLM은 코어(`LLM_*`, `aiagent/config/`), 외부 모델 API·서버는 진입점(`MODEL_API_*`·`APP_*`, `app/core/config.py`).
- FastAPI 라우트는 transport만 담당하고, 핵심 로직은 `aiagent/`로 내린다.

## 데이터 흐름

```
사용자 → FastAPI(/chat) → AgentFacade → AgentOrchestrator(LangGraph)
            의도분류 → 도구계획 → 도구실행 → 답변생성
                                  └ run_model_inference → InferenceModelClient(포트) → 외부 모델 API
```

## 폴더 구조

```
main.py                              조립 루트 (모든 의존성 생성·주입)
aiagent/                             프레임워크 독립 코어
├─ constants.py                      IntentType · 기본값
├─ config/{schemas,loader}.py        추론 LLM 설정 (LLM_*)
├─ ports/inference_model.py          외부 모델 API 호출 경계(포트)
├─ tools/tool_registry.py            포트를 도구로 노출 (build_tools)
├─ agents/                           agent_orchestrator · facade · prompt_builder · agent_state
├─ schemas/agent_io.py               계층 간 데이터 계약
└─ core/factory.py                   assemble_orchestrator(...)
app/                                 FastAPI 어댑터 (transport)
├─ api/routes.py                     /health · /chat
├─ core/{config,dependencies}.py     서버/모델 API 설정 · DI
├─ agent/conversation_store.py       대화 이력 보관 경계
└─ adapters/http_inference_client.py 포트의 HTTP 구현
frontend/index.html                  데모 채팅 UI (/ui)
tests/test_agent_core.py             코어 순수 로직 테스트
```

## 교체 지점 (기능 추가는 여기서)

| 무엇 | 어디 |
|------|------|
| 외부 모델 API 호출 | `app/adapters/http_inference_client.py` (`_build_payload`·`_parse_response`) |
| 도구 추가 | `aiagent/tools/tool_registry.py`의 `build_tools` |
| 의도 추가 | `aiagent/constants.py` + `agent_orchestrator.DICT_SCENARIO_TOOL_NAMES` |
| 대화 이력 백엔드 | `app/agent/conversation_store.py` (메모리 → DB) |

자세한 절차는 `docs/INTEGRATION_GUIDE.md` 참고.

## 실행

```bash
cp .env.example .env       # 값 채우기 (LLM_API_KEY 필수, 없으면 부팅 실패)
pip install -e ".[dev]"    # 또는 uv sync --extra dev
python main.py             # http://localhost:8000/ui
pytest                     # 코어 테스트
```

## 코드 규약

- 타입 명시(파라미터·반환·변수), 변수명은 `{type}_{의미}` snake_case 접두어(`str_`·`list_`·`dict_`·`model_` 등).
- 의존성은 진입점에서 생성해 주입한다. 하위 계층에서 설정·클라이언트·연결을 직접 만들지 않는다.
- 전역 상태·싱글톤 금지, `print` 대신 로거 사용, 계층 간 raw dict 대신 명시적 모델 전달.

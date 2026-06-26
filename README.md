# Agent AI Skeleton

외부 AI 모델 API를 **도구(tool)로 호출하는** LangGraph + FastAPI 에이전트 서비스 골격.
코어(`aiagent/`)는 DB·프레임워크·모델 구현을 모르고, 모든 의존성은 진입점(`main.py`)에서 생성·주입한다.

## 핵심 흐름

```
사용자 → FastAPI(/chat) → AgentFacade → AgentOrchestrator(LangGraph)
            의도분류 → 도구계획 → 도구실행 → 답변생성
                                  └ run_model_inference → InferenceModelClient(포트) → 외부 모델 API
```

## 폴더 구조

```
main.py                         조립 루트 (lifespan에서 모든 의존성 생성·주입)
aiagent/                            프레임워크 독립 코어
├─ config/{schemas,loader}.py   추론 LLM 설정 (LLM_* 환경변수)
├─ ports/inference_model.py     ★ 외부 모델 API 호출 경계(포트)
├─ tools/tool_registry.py       포트를 도구로 노출 (build_tools)
├─ agents/                      orchestrator · facade · prompt_builder · state
├─ schemas/agent_io.py          계층 간 데이터 계약
└─ core/factory.py              assemble_orchestrator(...)
app/                            FastAPI 어댑터 (transport 계층)
├─ api/routes.py                /health · /chat
├─ core/{config,dependencies}.py 서버/모델 API 설정 · DI
├─ agent/conversation_store.py  대화 이력 보관 경계
└─ adapters/http_inference_client.py  ★ 포트의 HTTP 구현
frontend/index.html             데모 채팅 UI (/ui)
tests/test_agent_core.py        코어 순수 로직 테스트
```

## 실행

```bash
cp .env.example .env       # 값 채우기
uv sync --extra dev        # 또는 pip install -e ".[dev]"
python main.py             # http://localhost:8000/ui
pytest                     # 코어 테스트
```

## 교체 지점 (이 부분만 갈아끼우면 된다)

| 무엇 | 어디 | 어떻게 |
|------|------|--------|
| 외부 모델 API 호출 | `app/adapters/http_inference_client.py` | 실제 API 요청/응답 스키마에 맞춰 `_build_payload`·`_parse_response` 수정 |
| 도구 추가 | `aiagent/tools/tool_registry.py` | 포트를 주입받아 `build_tools` 목록에 도구 등록 |
| 의도 종류 | `aiagent/constants.py` + `agent_orchestrator.DICT_SCENARIO_TOOL_NAMES` | `IntentType`에 의도 추가 후 시나리오 도구 매핑 |
| 대화 이력 백엔드 | `app/agent/conversation_store.py` | `ConversationStore`를 DB 구현으로 교체 |

자세한 핸드오프 절차는 [docs/INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md) 참고.
